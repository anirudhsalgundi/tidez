"""
Step 6: build the scanning bundle for the fast-transient candidates.

One small H5 file, in the layout the web scanner (webapp.py) already reads, holding only
the candidates in data/fast_transients.csv, so it can be copied to a laptop and scanned
offline:

  /<objid>/                        attrs: objid, ra, dec, program, decam_on_tns, the TNS
                                   match (tns_name, tns_type, ...) and the filter's numbers
                                   (rise_rate, rise_band, fade_rate, age_days, ...)
  /<objid>/lightcurve              every DECam epoch: MJD_OBS, FILTER, MAG_FPHOT, MAGERR_FPHOT,
                                   SNR_FPHOT, STATUS_FPHOT, LIM_MAG3, LIM_MAG5, CNNSCORE, and
                                   DETECTED (the same detection rule as get_lcs.py)
  /<objid>/cutouts/{template,science,diff}
                                   float16 (n, 121, 121) stamps for detection epochs only
                                   (DETECTED, or STATUS_FPHOT == 'p'); each stamp is divided
                                   by its <key>_scale (>= 1, float32) to fit float16's range,
                                   so pixel values = stamp * scale
  /<objid>/cutouts/epoch_idx       which lightcurve rows those stamps belong to
  /<objid>/crossmatch/{ztf,lsst}/  lightcurve (jd, mag, magerr, limmag, band, detected),
                                   attrs objectId, separation_arcsec, n_alerts -- from
                                   data/boom_lc.parquet
  /<objid>/legacy_survey/          Legacy Survey DR10 JPEG cutout (fetched once)
  /scan_order                      candidate objids in the filter's ranking (fastest first)

decam_on_tns is True when the matched TNS object lists this DECam objid among its
internal names, i.e. the DECam data has been reported.

Resumable: objects already in the bundle are skipped, so a re-run after a crash or a
failed Legacy Survey fetch only does what's missing (objects whose cutout fetch failed
are written without one; --retry-legacy fetches those).

Usage:
    python tidez/make_scan_bundle.py
    python tidez/make_scan_bundle.py --candidates data/fast_transients.csv --out data/scan_bundle.h5
"""

import argparse
import pathlib
import sys
import time

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from tidez.build_h5 import STAMP_SIDE, append_to_catalog, ensure_catalog_dataset
from tidez.crossmatch_legacy_survey import (DEFAULT_LAYER, DEFAULT_PIXSCALE, DEFAULT_SIZE_PIX, DEFAULT_SLEEP,
                                            fetch_cutout_bytes, write_legacy_cutout)
from tidez.build_lcs import GOOD_STATUS, MIN_CNN, MIN_SNR
from tidez.logger import get_logger
logger = get_logger("make_scan_bundle")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CANDIDATES_FILE = DATA_DIR / "fast_transients.csv"
MASTER_FILE = DATA_DIR / "all_objects.csv"
BOOM_FILE = DATA_DIR / "boom_lc.parquet"
OUT_FILE = DATA_DIR / "fast_transients.h5"

JD_TO_MJD = 2400000.5
FLOAT16_TARGET = 60000.0  # stamps whose |values| exceed this are scaled down into float16's range (max 65504)

LIGHTCURVE_DTYPE = np.dtype([
    ("MJD_OBS", "f8"), ("FILTER", "S3"), ("MAG_FPHOT", "f4"), ("MAGERR_FPHOT", "f4"),
    ("SNR_FPHOT", "f4"), ("STATUS_FPHOT", "S1"), ("LIM_MAG3", "f4"), ("LIM_MAG5", "f4"),
    ("CNNSCORE", "f4"), ("DETECTED", "?"),
])
CROSSMATCH_DTYPE = np.dtype([("jd", "f8"), ("mag", "f4"), ("magerr", "f4"), ("limmag", "f4"),
                             ("band", "S1"), ("detected", "?")])
# candidate-CSV columns that are text even when every value is missing (pandas then reads them as float)
TEXT_COLUMNS = {"rise_band", "rise_pair", "fade_band", "fade_pair", "peak_band", "bands_detected",
                "surveys_detected", "tns_name", "tns_type", "tns_discoverydate", "tns_internal_names"}
STAMP_COLUMNS = {"template": "PixA_THUMB_TEMP", "science": "PixA_THUMB_SCI", "diff": "PixA_THUMB_DIFF"}
BOOM_KEYS = {"ZTF": ("ztf", "ztfname", "ztf_sep_arcsec"), "LSST": ("lsst", "lsstname", "lsst_sep_arcsec")}

# In a SLURM log (not a terminal) every redraw is appended to the file, so update rarely.
BAR_KWARGS = {} if sys.stderr.isatty() else {"mininterval": 30, "ascii": True}


def _strip(values) -> np.ndarray:
    return np.char.strip(np.asarray(values).astype(str))


def read_decam(file_path: str) -> tuple:
    """(lightcurve record array, {key: float16 stamps}, epoch_idx of those stamps) from one
    candidate FITS file. Only the stamps of detection epochs are read off disk (memmap)."""
    with fits.open(file_path, memmap=True) as hdul:
        t = hdul[1].data
        lc = np.empty(len(t), dtype=LIGHTCURVE_DTYPE)
        for col in ("MJD_OBS", "MAG_FPHOT", "MAGERR_FPHOT", "SNR_FPHOT", "LIM_MAG3", "LIM_MAG5"):
            lc[col] = t[col]
        lc["FILTER"] = np.char.encode(_strip(t["FILTER"]), "ascii")
        status = _strip(t["STATUS_FPHOT"])
        lc["STATUS_FPHOT"] = np.char.encode(status, "ascii")
        lc["CNNSCORE"] = t["CNNSCORE_ALT"]
        # same rule as get_lcs.py (NaN comparisons are False, so NaNs never count)
        lc["DETECTED"] = ((status == GOOD_STATUS) & (lc["SNR_FPHOT"] > MIN_SNR) & (lc["CNNSCORE"] > MIN_CNN)
                          & (lc["MAG_FPHOT"] <= lc["LIM_MAG5"]))
        epoch_idx = np.flatnonzero(lc["DETECTED"] | (status == GOOD_STATUS)).astype("i4")
        stamps = {}
        for key, col in STAMP_COLUMNS.items():
            raw = np.asarray(t[col][epoch_idx], dtype="f4").reshape(len(epoch_idx), STAMP_SIDE, STAMP_SIDE)
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            # per-stamp scale so even huge-valued stamps keep float16's ~0.1% precision
            peak = np.abs(raw).max(axis=(1, 2)) if len(raw) else np.zeros(0, "f4")
            scale = np.maximum(1.0, peak / FLOAT16_TARGET).astype("f4")
            stamps[key] = (raw / scale[:, None, None]).astype("f2")
            stamps[f"{key}_scale"] = scale
    return lc, stamps, epoch_idx


def crossmatch_arrays(boom_rows: pd.DataFrame) -> dict:
    """{'ztf'/'lsst': (lightcurve, attrs)} from one object's rows of boom_lc.parquet."""
    out = {}
    for survey, (key, name_col, sep_col) in BOOM_KEYS.items():
        rows = boom_rows[boom_rows["survey"] == survey].sort_values("mjd")
        if rows.empty:
            continue
        lc = np.empty(len(rows), dtype=CROSSMATCH_DTYPE)
        lc["jd"] = rows["mjd"].to_numpy() + JD_TO_MJD
        for col in ("mag", "magerr", "limmag", "detected"):
            lc[col] = rows[col].to_numpy()
        lc["band"] = np.char.encode(rows["filter"].fillna("").str[:1].to_numpy().astype(str), "ascii")
        attrs = {"objectId": str(rows[name_col].iloc[0]), "separation_arcsec": float(rows[sep_col].iloc[0]),
                 "n_alerts": int((rows["kind"] == "alert").sum()) if "kind" in rows else len(rows)}
        out[key] = (lc, attrs)
    return out


def decam_on_tns(objid: str, internal_names) -> bool:
    if not isinstance(internal_names, str):
        return False
    return objid in {n.strip() for n in internal_names.split(",")}


def _attr_value(value, is_text: bool = False):
    """H5 attribute-friendly version of a CSV cell (numbers as float, NaN kept; text as str,
    missing text as '')."""
    if value is None or (is_text and isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return str(value)


def write_object(h5: h5py.File, objid: str, ra: float, dec: float, attrs: dict,
                 lc: np.ndarray, stamps: dict, epoch_idx: np.ndarray, crossmatch: dict) -> None:
    """Write one object's group; everything is built in memory first, so a group is either
    complete or absent."""
    group = h5.create_group(objid)
    group.attrs.update({"objid": objid, "ra": ra, "dec": dec, **attrs})
    group.create_dataset("lightcurve", data=lc, compression="gzip", compression_opts=4)
    cut = group.create_group("cutouts")
    cut.create_dataset("epoch_idx", data=epoch_idx)
    for key, arr in stamps.items():
        cut.create_dataset(key, data=arr, compression="gzip", compression_opts=4, shuffle=True)
    if crossmatch:
        cm = group.create_group("crossmatch")
        for key, (xlc, xattrs) in crossmatch.items():
            sub = cm.create_group(key)
            sub.attrs.update(xattrs)
            sub.create_dataset("lightcurve", data=xlc, compression="gzip", compression_opts=4)
    append_to_catalog(h5, objid, ra, dec, len(lc))


def add_legacy(h5: h5py.File, objid: str, layer: str, pixscale: float, size_pix: int) -> bool:
    ra, dec = float(h5[objid].attrs["ra"]), float(h5[objid].attrs["dec"])
    try:
        jpg = fetch_cutout_bytes(ra, dec, layer, pixscale, size_pix)
    except Exception as exc:
        logger.warning(f"{objid}: Legacy Survey cutout failed ({exc}); re-run with --retry-legacy")
        return False
    write_legacy_cutout(h5, objid, jpg, layer, pixscale, size_pix)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the scanning bundle for the fast-transient candidates.")
    parser.add_argument("--candidates", type=pathlib.Path, default=CANDIDATES_FILE,
                        help="find_fast_transients.py output CSV (default: data/fast_transients.csv).")
    parser.add_argument("--master", type=pathlib.Path, default=MASTER_FILE)
    parser.add_argument("--boom", type=pathlib.Path, default=BOOM_FILE)
    parser.add_argument("--out", type=pathlib.Path, default=OUT_FILE)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--no-legacy", action="store_true", help="Skip the Legacy Survey cutouts.")
    parser.add_argument("--retry-legacy", action="store_true",
                        help="Also fetch Legacy Survey cutouts for objects already in the bundle without one.")
    args = parser.parse_args()

    cand = pd.read_csv(args.candidates, index_col=0)
    text_cols = {c: cand[c].dtype == object or c in TEXT_COLUMNS for c in cand.columns}
    master = pd.read_csv(args.master)
    master = master.dropna(subset=["ra", "dec"])
    programs = master.groupby("objid")["program"].agg(lambda p: ", ".join(dict.fromkeys(p)))
    first = master.drop_duplicates("objid").set_index("objid")
    boom = pd.read_parquet(args.boom) if args.boom.exists() else pd.DataFrame(columns=["decam_objid"])
    boom = boom[boom["decam_objid"].isin(cand.index)]
    boom_by_obj = dict(tuple(boom.groupby("decam_objid")))
    logger.info(f"{len(cand)} candidates from {args.candidates}; {len(boom_by_obj)} with ZTF/LSST data")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_new = n_fail = n_legacy = 0
    with h5py.File(args.out, "a") as h5:
        ensure_catalog_dataset(h5)
        todo = [o for o in cand.index if o not in h5]
        logger.info(f"{len(cand) - len(todo)} already in {args.out}; adding {len(todo)}")
        for objid in tqdm(todo, desc="Bundling candidates", unit="obj", **BAR_KWARGS):
            if objid not in first.index:
                logger.warning(f"{objid}: not in {args.master}, skipped")
                n_fail += 1
                continue
            row = cand.loc[objid]
            try:
                lc, stamps, epoch_idx = read_decam(first.at[objid, "file_path"])
            except Exception as exc:
                logger.warning(f"{objid}: could not read {first.at[objid, 'file_path']}: {exc}")
                n_fail += 1
                continue
            attrs = {c: _attr_value(row[c], text_cols[c]) for c in cand.columns}
            attrs["program"] = programs.get(objid, "")
            attrs["decam_on_tns"] = decam_on_tns(objid, row.get("tns_internal_names"))
            write_object(h5, objid, float(first.at[objid, "ra"]), float(first.at[objid, "dec"]), attrs,
                         lc, stamps, epoch_idx, crossmatch_arrays(boom_by_obj.get(objid, boom.iloc[:0])))
            n_new += 1
            if not args.no_legacy:
                n_legacy += add_legacy(h5, objid, args.layer, DEFAULT_PIXSCALE, DEFAULT_SIZE_PIX)
                time.sleep(DEFAULT_SLEEP)
            h5.flush()

        if args.retry_legacy and not args.no_legacy:
            missing = [o for o in cand.index if o in h5 and "legacy_survey" not in h5[o]]
            for objid in tqdm(missing, desc="Retrying Legacy Survey", unit="obj", **BAR_KWARGS):
                n_legacy += add_legacy(h5, objid, args.layer, DEFAULT_PIXSCALE, DEFAULT_SIZE_PIX)
                time.sleep(DEFAULT_SLEEP)

        order = [o for o in cand.index if o in h5]
        if "scan_order" in h5:
            del h5["scan_order"]
        h5.create_dataset("scan_order", data=np.array(order, dtype=h5py.string_dtype()))

    size_mb = args.out.stat().st_size / 1e6
    logger.info(f"Added {n_new} objects ({n_legacy} Legacy Survey cutouts), {n_fail} failed; "
                f"{len(order)} objects in {args.out} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
