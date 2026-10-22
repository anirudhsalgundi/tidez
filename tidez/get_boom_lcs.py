"""
Step 3: crossmatch every DECam object with ZTF and LSST in BOOM, and fetch their light curves.

For each chunk of DECam objects, one cone search per survey on BOOM's *_alerts_aux catalog
(one document per ZTF/LSST object, holding its whole history). The nearest BOOM object
within --radius-arcsec is the match (every object within the radius is listed too), and
its whole history becomes the light curve:

  prv_candidates     alerts (positive and negative)         kind = "alert"
  prv_nondetections  upper limits (ZTF only)                kind = "nondetection"
  fp_hists           forced photometry                      kind = "forced_phot"

NO DATA IS REMOVED OR JUDGED HERE. Every point BOOM has is kept, in every filter, with
BOOM's values as they are: no SNR, sign, procstatus or quality cuts, and no detection flag.
Those cuts are left for later, so the raw quantities they need (isdiffpos, procstatus,
snr_psf) are kept alongside the photometry. mags/magerrs are BOOM's magpsf/sigmapsf as-is
(for negative subtractions BOOM gives the magnitude of |flux|; isdiffpos says the sign).
On top of that, each survey's untouched BOOM history (every field of every point) is
stored as JSON in *_raw_history, so nothing BOOM returned is lost.

The only transformations: jd -> mjd, and psfFlux -> microJy. BOOM stores psfFlux in nJy for
LSST but in 1e-6 nJy for ZTF (checked: magpsf = -2.5 log10(psfFlux) + 31.4 for LSST and
+ 46.4 for ZTF). Points are sorted by time (any point without a jd goes last, mjd = NaN).

Outputs:
  data/boom_lightcurves.parquet   one row per DECam object with a ZTF and/or LSST match:
      decam_objid,
      ztfname,  ztf_sep_arcsec,  ztf_all_match_names,  ztf_all_match_seps_arcsec,
      ztf_mjds,  ztf_filters,  ztf_mags,  ztf_magerrs,  ztf_limiting_mags,
      ztf_fluxes_ujy,  ztf_flux_errs_ujy,  ztf_snr_psfs,  ztf_isdiffpos,  ztf_procstatus,
      ztf_kinds,  ztf_raw_history,
      lsstname, lsst_sep_arcsec, ... (same as ztf_),
      lsst_ztf_aliases (BOOM's own ZTF names for the LSST object, a cross-check on ztfname)
    every per-point list column has one entry per point, in time order; a value BOOM
    doesn't have for a point is NaN/None.
  data/all_objects.parquet        gains ztfname, ztf_sep_arcsec, lsstname, lsst_sep_arcsec
    (None/NaN where there is no match). create_masterdb.py rewrites the master without
    them, so re-run this step after it.

Resumable: each chunk is saved to data/boom_lightcurves_parts/ as it finishes, and a re-run
skips objects already saved. Once every object is done, the parts are merged into the two
outputs and the parts directory is removed.

Needs BOOM_USERNAME and BOOM_PASSWORD in the environment.

Usage:
    python tidez/get_boom_lcs.py
    python tidez/get_boom_lcs.py --radius-arcsec 3
"""

import argparse
import json
import math
import os
import pathlib
import shutil

import numpy as np
import pandas as pd
from boom import BoomClient
from tqdm import tqdm

from tidez.logger import get_logger
logger = get_logger("get_boom_lcs")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MASTER_FILE = DATA_DIR / "all_objects.parquet"
LIGHTCURVES_FILE = DATA_DIR / "boom_lightcurves.parquet"

CHUNK_SIZE = 100  # DECam objects per cone search; small, since dense ZTF fields return long histories
DEFAULT_RADIUS_ARCSEC = 2.0
JD_TO_MJD = 2400000.5

# survey prefix used in column names -> BOOM catalog holding one document per object
SURVEY_CATALOGS = {"ztf": "ZTF_alerts_aux", "lsst": "LSST_alerts_aux"}
# BOOM psfFlux -> microJy (LSST stores nJy, ZTF stores 1e-6 nJy)
PSF_FLUX_TO_UJY = {"ztf": 1e-9, "lsst": 1e-3}
# BOOM history array -> kind written for its points
HISTORY_KINDS = {"prv_candidates": "alert", "prv_nondetections": "nondetection", "fp_hists": "forced_phot"}
PROJECTION = {  # whole history arrays, every field
    **{history: 1 for history in HISTORY_KINDS},
    "coordinates.radec_geojson": 1,
    "aliases": 1,
}
LIGHTCURVE_FIELDS = ["mjds", "filters", "mags", "magerrs", "limiting_mags", "fluxes_ujy", "flux_errs_ujy",
                     "snr_psfs", "isdiffpos", "procstatus", "kinds"]
CROSSMATCH_COLUMNS = ["ztfname", "ztf_sep_arcsec", "lsstname", "lsst_sep_arcsec"]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def to_float(value) -> float:
    return math.nan if value is None else float(value)


def boom_object_position(boom_doc: dict) -> tuple:
    """Median alert position, falling back to the catalog's GeoJSON point (lon = ra - 180)."""
    alerts = [a for a in boom_doc.get("prv_candidates") or [] if a.get("ra") is not None and a.get("dec") is not None]
    if alerts:
        return float(np.median([a["ra"] for a in alerts])), float(np.median([a["dec"] for a in alerts]))
    lon, lat = boom_doc["coordinates"]["radec_geojson"]["coordinates"]
    return lon + 180.0, lat


def separation_arcsec(ra1, dec1, ra2, dec2) -> float:
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    a = (math.sin((dec2 - dec1) / 2) ** 2
         + math.cos(dec1) * math.cos(dec2) * math.sin((ra2 - ra1) / 2) ** 2)
    return math.degrees(2 * math.asin(math.sqrt(min(1.0, a)))) * 3600.0


def sort_by_separation(decam_ra: float, decam_dec: float, boom_docs: list) -> list:
    """[(BOOM document, separation in arcsec)] for every match, nearest first."""
    separations = [separation_arcsec(decam_ra, decam_dec, *boom_object_position(d)) for d in boom_docs]
    return sorted(zip(boom_docs, separations), key=lambda pair: pair[1])


# ---------------------------------------------------------------------------
# Light curves
# ---------------------------------------------------------------------------

def extract_lightcurve(survey: str, boom_doc: dict) -> dict:
    """Every point of one BOOM object's history as {field: list}, sorted by time. Nothing is
    dropped and no value is judged; a value BOOM doesn't have for a point is NaN/None."""
    flux_to_ujy = PSF_FLUX_TO_UJY[survey]
    points = []
    for history, kind in HISTORY_KINDS.items():
        for entry in boom_doc.get(history) or []:
            points.append({
                "mjds": to_float(entry.get("jd")) - JD_TO_MJD,
                "filters": entry.get("band"),
                "mags": to_float(entry.get("magpsf")),
                "magerrs": to_float(entry.get("sigmapsf")),
                "limiting_mags": to_float(entry.get("diffmaglim")),
                "fluxes_ujy": to_float(entry.get("psfFlux")) * flux_to_ujy,
                "flux_errs_ujy": to_float(entry.get("psfFluxErr")) * flux_to_ujy,
                "snr_psfs": to_float(entry.get("snr_psf")),
                "isdiffpos": entry.get("isdiffpos"),
                "procstatus": None if entry.get("procstatus") is None else str(entry["procstatus"]),
                "kinds": kind,
            })
    points.sort(key=lambda p: (math.isnan(p["mjds"]), p["mjds"]))  # points without a jd go last
    return {field: [p[field] for p in points] for field in LIGHTCURVE_FIELDS}


def raw_history_json(boom_doc: dict) -> str:
    """The BOOM history arrays exactly as returned, every field of every point."""
    return json.dumps({history: boom_doc.get(history) or [] for history in HISTORY_KINDS}, default=str)


def empty_survey_columns(survey: str) -> dict:
    columns = {f"{survey}name": None, f"{survey}_sep_arcsec": math.nan,
               f"{survey}_all_match_names": [], f"{survey}_all_match_seps_arcsec": [],
               **{f"{survey}_{field}": [] for field in LIGHTCURVE_FIELDS},
               f"{survey}_raw_history": None}
    if survey == "lsst":
        columns["lsst_ztf_aliases"] = []
    return columns


def process_chunk(boom: BoomClient, decam_positions: dict, radius_arcsec: float) -> pd.DataFrame:
    """One row per DECam object in the chunk (matched or not) with its ZTF and LSST columns."""
    rows = {objid: {"decam_objid": objid} for objid in decam_positions}
    for survey, catalog in SURVEY_CATALOGS.items():
        matches_by_objid = boom.cone_search(catalog, decam_positions, radius_arcsec, unit="arcsec",
                                            projection=PROJECTION)
        for objid, (decam_ra, decam_dec) in decam_positions.items():
            boom_docs = matches_by_objid.get(objid) or []
            survey_columns = empty_survey_columns(survey)
            if boom_docs:
                matches = sort_by_separation(decam_ra, decam_dec, boom_docs)
                boom_doc, sep_arcsec = matches[0]
                survey_columns[f"{survey}name"] = str(boom_doc["_id"])
                survey_columns[f"{survey}_sep_arcsec"] = sep_arcsec
                survey_columns[f"{survey}_all_match_names"] = [str(doc["_id"]) for doc, _ in matches]
                survey_columns[f"{survey}_all_match_seps_arcsec"] = [sep for _, sep in matches]
                for field, values in extract_lightcurve(survey, boom_doc).items():
                    survey_columns[f"{survey}_{field}"] = values
                survey_columns[f"{survey}_raw_history"] = raw_history_json(boom_doc)
                if survey == "lsst":
                    survey_columns["lsst_ztf_aliases"] = list((boom_doc.get("aliases") or {}).get("ZTF") or [])
            rows[objid].update(survey_columns)
        del matches_by_objid  # free the response before querying the next survey
    return pd.DataFrame(list(rows.values()))


# ---------------------------------------------------------------------------
# Inputs and outputs
# ---------------------------------------------------------------------------

def load_decam_positions(master: pathlib.Path) -> dict:
    """{objid: (ra, dec)} for every master object with a position (first row wins for duplicated objids)."""
    if not master.exists():
        raise FileNotFoundError(f"{master} not found -- run create_masterdb.py first")
    master_df = pd.read_parquet(master, columns=["objid", "ra", "dec"])
    master_df = master_df.dropna(subset=["ra", "dec"]).drop_duplicates("objid")
    logger.info(f"{len(master_df)} objects with RA/Dec in {master}")
    return {objid: (float(ra), float(dec)) for objid, ra, dec in zip(master_df["objid"], master_df["ra"], master_df["dec"])}


def load_done_objids(parts_dir: pathlib.Path) -> set:
    """DECam objids already saved in a part (every queried object has a row, matched or not)."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for part in parts_dir.glob("chunk_*.parquet"):
        done.update(pd.read_parquet(part, columns=["decam_objid"])["decam_objid"])
    return done


def write_atomically(df: pd.DataFrame, path: pathlib.Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)  # a file is either complete or untouched


def add_crossmatches_to_master(crossmatches: pd.DataFrame, master: pathlib.Path) -> None:
    """Replace the master's ztfname/lsstname columns (if any) with this run's crossmatches."""
    master_df = pd.read_parquet(master)
    master_df = master_df.drop(columns=[c for c in CROSSMATCH_COLUMNS if c in master_df.columns])
    master_df = master_df.merge(crossmatches.rename(columns={"decam_objid": "objid"}), on="objid", how="left")
    write_atomically(master_df, master)
    logger.info(f"Added {', '.join(CROSSMATCH_COLUMNS)} to {master}: "
                f"{master_df['ztfname'].notna().sum()} rows with a ZTF match, "
                f"{master_df['lsstname'].notna().sum()} with an LSST match, out of {len(master_df)}")


def finish(parts_dir: pathlib.Path, lightcurves_file: pathlib.Path, master: pathlib.Path) -> None:
    """Merge the parts into the light-curve file and the master's crossmatch columns, then remove them."""
    all_objects = pd.concat([pd.read_parquet(p) for p in sorted(parts_dir.glob("chunk_*.parquet"))],
                            ignore_index=True)
    add_crossmatches_to_master(all_objects[["decam_objid"] + CROSSMATCH_COLUMNS], master)

    has_match = all_objects["ztfname"].notna() | all_objects["lsstname"].notna()
    lightcurves = all_objects[has_match].sort_values("decam_objid", ignore_index=True)
    n_ztf_points = lightcurves["ztf_mjds"].apply(len).sum()
    n_lsst_points = lightcurves["lsst_mjds"].apply(len).sum()
    write_atomically(lightcurves, lightcurves_file)
    logger.info(f"Wrote {len(lightcurves)} matched objects to {lightcurves_file} "
                f"({n_ztf_points} ZTF points, {n_lsst_points} LSST points)")
    shutil.rmtree(parts_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def connect_to_boom() -> BoomClient:
    if not os.environ.get("BOOM_USERNAME") or not os.environ.get("BOOM_PASSWORD"):
        raise SystemExit("Set BOOM_USERNAME and BOOM_PASSWORD in the environment.")
    return BoomClient(username=os.environ["BOOM_USERNAME"], password=os.environ["BOOM_PASSWORD"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Crossmatch master-file objects with ZTF/LSST in BOOM and fetch their light curves.")
    parser.add_argument("--master", type=pathlib.Path, default=MASTER_FILE)
    parser.add_argument("--out", type=pathlib.Path, default=LIGHTCURVES_FILE)
    parser.add_argument("--radius-arcsec", type=float, default=DEFAULT_RADIUS_ARCSEC)
    args = parser.parse_args()

    decam_positions = load_decam_positions(args.master)
    parts_dir = args.out.parent / (args.out.stem + "_parts")
    done_objids = load_done_objids(parts_dir)
    todo_objids = [objid for objid in decam_positions if objid not in done_objids]
    chunks = [todo_objids[i:i + CHUNK_SIZE] for i in range(0, len(todo_objids), CHUNK_SIZE)]
    logger.info(f"{len(done_objids)} objects already done in {parts_dir}; {len(todo_objids)} to query "
                f"in {len(chunks)} chunks of {CHUNK_SIZE} (radius {args.radius_arcsec}\")")

    boom = connect_to_boom()
    first_chunk_number = len(list(parts_dir.glob("chunk_*.parquet")))
    n_failed_objects = n_ztf_matches = n_lsst_matches = 0
    progress = tqdm(chunks, desc="BOOM chunks", unit="chunk")
    for chunk_number, chunk_objids in enumerate(progress, start=first_chunk_number):
        chunk_positions = {objid: decam_positions[objid] for objid in chunk_objids}
        try:
            chunk_df = process_chunk(boom, chunk_positions, args.radius_arcsec)
        except Exception as e:
            logger.error(f"Chunk {chunk_number} ({len(chunk_objids)} objects) failed, "
                         f"will be retried on the next run: {type(e).__name__}: {e}")
            n_failed_objects += len(chunk_objids)
            continue
        write_atomically(chunk_df, parts_dir / f"chunk_{chunk_number:06d}.parquet")
        n_ztf_matches += chunk_df["ztfname"].notna().sum()
        n_lsst_matches += chunk_df["lsstname"].notna().sum()
        progress.set_postfix(ztf=n_ztf_matches, lsst=n_lsst_matches, failed=n_failed_objects)
    logger.info(f"This run: {n_ztf_matches} ZTF and {n_lsst_matches} LSST matches")

    if n_failed_objects:
        logger.error(f"{n_failed_objects} objects failed; finished chunks are kept in {parts_dir}. "
                     f"Re-run to retry them -- the outputs are written once everything is done.")
        return
    finish(parts_dir, args.out, args.master)


if __name__ == "__main__":
    main()
