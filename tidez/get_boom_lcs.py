"""
Step 3: fetch ZTF and LSST light curves from BOOM for every object in the master file.

For each batch of objects in data/all_objects.csv, one cone search per survey on
BOOM's *_alerts_aux catalog returns each matched object's full history in one document:

  prv_candidates     alert detections           kind = "alert"
                     (negative subtractions)    kind = "alert_negative"
  prv_nondetections  ZTF upper limits           kind = "nondetection"
  fp_hists           forced photometry          kind = "fp"

The nearest BOOM object within --radius-arcsec is the match. Rows follow the DECam
light-curve convention: detected = True rows carry mag/magerr; every other row is an
upper limit at limmag (= diffmaglim) with mag = magerr = NaN. Forced photometry counts
as a detection at SNR >= 5 with a positive difference (same 5-sigma bar as DECam);
ZTF forced photometry with procstatus != 0 is dropped.

Output: data/boom_lc.parquet, one row per ZTF/LSST measurement:
    decam_objid, ztfname, lsstname, survey, kind, mjd, filter, mag, magerr, limmag,
    snr, detected, ztf_sep_arcsec, lsst_sep_arcsec
ztfname/lsstname (and their separations) are the object's matches, repeated on every
row; survey says which of the two a row's measurement came from. Objects with no
match in either survey have no rows.

Resumable: each batch is saved to data/boom_lc_parts/ as it finishes, and a re-run
skips finished batches. Once every object is done, the parts are merged into
boom_lc.parquet and the parts directory is removed (so the next run starts fresh,
picking up new alerts).

Needs BOOM_USERNAME and BOOM_PASSWORD in the environment.

Usage:
    python tidez/get_boom_lcs.py
    python tidez/get_boom_lcs.py --radius-arcsec 3
"""

import argparse
import csv
import json
import math
import os
import pathlib
import re
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from tidez.logger import get_logger
logger = get_logger("get_boom_lcs")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MASTER_FILE = DATA_DIR / "all_objects.csv"
OUT_FILE = DATA_DIR / "boom_lc.parquet"

BOOM_BASE = "https://api.kaboom.caltech.edu"
REQUEST_TIMEOUT = 180
MAX_RETRIES = 4
BATCH_SIZE = 100  # small enough that one batch's JSON (dense ZTF fields, long histories) fits in memory
SLEEP_BETWEEN_BATCHES = 0.5  # seconds
DEFAULT_RADIUS_ARCSEC = 2.0
MIN_FP_SNR = 5.0
JD_TO_MJD = 2400000.5

SURVEYS = {"ZTF": "ZTF_alerts_aux", "LSST": "LSST_alerts_aux"}
NAME_COLUMN = {"ZTF": "ztfname", "LSST": "lsstname"}
SEP_COLUMN = {"ZTF": "ztf_sep_arcsec", "LSST": "lsst_sep_arcsec"}
HISTORY_FIELDS = ("jd", "band", "magpsf", "sigmapsf", "diffmaglim", "snr", "snr_psf",
                  "isdiffpos", "isNegative", "procstatus", "ra", "dec")
PROJECTION = {
    **{f"{array}.{field}": 1
       for array in ("prv_candidates", "prv_nondetections", "fp_hists") for field in HISTORY_FIELDS},
    "coordinates.radec_geojson": 1,
}

SCHEMA = pa.schema([
    ("decam_objid", pa.string()),
    ("ztfname", pa.string()),
    ("lsstname", pa.string()),
    ("survey", pa.string()),
    ("kind", pa.string()),
    ("mjd", pa.float64()),
    ("filter", pa.string()),
    ("mag", pa.float32()),
    ("magerr", pa.float32()),
    ("limmag", pa.float32()),
    ("snr", pa.float32()),
    ("detected", pa.bool_()),
    ("ztf_sep_arcsec", pa.float32()),
    ("lsst_sep_arcsec", pa.float32()),
])
DONE_KEY = b"queried_objids"  # Parquet metadata key: every objid a part covers, matched or not

# In a SLURM log (not a terminal) every redraw is appended to the file, so update rarely.
BAR_KWARGS = {} if sys.stderr.isatty() else {"mininterval": 30, "ascii": True}


# ---------------------------------------------------------------------------
# BOOM client
# ---------------------------------------------------------------------------

_session = requests.Session()
_token = None


def _login() -> None:
    global _token
    username, password = os.getenv("BOOM_USERNAME"), os.getenv("BOOM_PASSWORD")
    if not username or not password:
        raise RuntimeError("Set BOOM_USERNAME and BOOM_PASSWORD environment variables.")
    resp = _session.post(f"{BOOM_BASE}/auth", data={"username": username, "password": password},
                         timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    _token = resp.json()["access_token"]


def cone_search_aux(coords: dict, survey: str, radius_arcsec: float) -> dict:
    """{objid: [aux docs within radius]} for one batch; objids with no match are absent."""
    payload = {
        "catalog_name": SURVEYS[survey],
        "object_coordinates": {objid: [ra, dec] for objid, (ra, dec) in coords.items()},
        "radius": radius_arcsec,
        "unit": "Arcseconds",
        "projection": PROJECTION,
    }
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if _token is None:
                _login()
            resp = _session.post(f"{BOOM_BASE}/queries/cone_search", json=payload, timeout=REQUEST_TIMEOUT,
                                 headers={"Authorization": f"Bearer {_token}"})
            if resp.status_code == 401:
                logger.warning("BOOM token expired/rejected, re-authenticating.")
                _login()
                continue
            resp.raise_for_status()
            return resp.json().get("data", {})
        except requests.RequestException as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(f"{survey} cone search failed (attempt {attempt}/{MAX_RETRIES}, "
                           f"{len(coords)} objects): {exc}. Retrying in {wait}s.")
            time.sleep(wait)
    raise RuntimeError(f"{survey} cone search failed after {MAX_RETRIES} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# Matching + photometry rows
# ---------------------------------------------------------------------------

def _num(value) -> float:
    return math.nan if value is None else float(value)


def _doc_position(doc: dict) -> tuple:
    """Median alert position, falling back to the catalog's GeoJSON point (lon = ra - 180)."""
    alerts = [a for a in doc.get("prv_candidates") or [] if a.get("ra") is not None and a.get("dec") is not None]
    if alerts:
        return float(np.median([a["ra"] for a in alerts])), float(np.median([a["dec"] for a in alerts]))
    lon, lat = doc["coordinates"]["radec_geojson"]["coordinates"]
    return lon + 180.0, lat


def _sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    a = (math.sin((dec2 - dec1) / 2) ** 2
         + math.cos(dec1) * math.cos(dec2) * math.sin((ra2 - ra1) / 2) ** 2)
    return math.degrees(2 * math.asin(math.sqrt(min(1.0, a)))) * 3600.0


def _photometry_columns(survey: str, doc: dict) -> dict:
    """One BOOM object's history as {column: numpy array} (compact; the JSON can be freed)."""
    kind, mjd, band, mag, magerr, limmag, snr, detected = ([] for _ in range(8))

    def add(k, e, det):
        kind.append(k)
        mjd.append(_num(e.get("jd")) - JD_TO_MJD)
        band.append(e.get("band") or "")
        mag.append(_num(e.get("magpsf")) if det else math.nan)
        magerr.append(_num(e.get("sigmapsf")) if det else math.nan)
        limmag.append(_num(e.get("diffmaglim")))
        snr.append(_num(e.get("snr", e.get("snr_psf"))))
        detected.append(det)

    for e in doc.get("prv_candidates") or []:
        positive = bool(e.get("isdiffpos")) and not e.get("isNegative", False)
        add("alert" if positive else "alert_negative", e, positive and e.get("magpsf") is not None)
    for e in doc.get("prv_nondetections") or []:
        add("nondetection", e, False)
    for e in doc.get("fp_hists") or []:
        if survey == "ZTF" and str(e.get("procstatus", "0")) != "0":
            continue
        s = e.get("snr", e.get("snr_psf"))
        add("fp", e, bool(e.get("isdiffpos")) and e.get("magpsf") is not None
            and s is not None and s >= MIN_FP_SNR)

    cols = {
        "kind": np.array(kind, dtype=object), "mjd": np.array(mjd, dtype="f8"),
        "filter": np.array(band, dtype=object), "mag": np.array(mag, dtype="f4"),
        "magerr": np.array(magerr, dtype="f4"), "limmag": np.array(limmag, dtype="f4"),
        "snr": np.array(snr, dtype="f4"), "detected": np.array(detected, dtype=bool),
    }
    ok = np.isfinite(cols["mjd"])
    return {k: v[ok] for k, v in cols.items()}


def process_batch(coords: dict, radius_arcsec: float) -> pa.Table:
    """Query every survey for one batch and return its photometry as an Arrow table.

    Each survey's response is reduced to the best match's compact columns right away and
    the raw JSON is dropped before the next survey is queried, so peak memory stays near
    one survey response for one batch.
    """
    matches = {objid: {} for objid in coords}  # objid -> {survey: (name, sep, columns)}
    for survey in SURVEYS:
        found = cone_search_aux(coords, survey, radius_arcsec)
        for objid, (ra, dec) in coords.items():
            docs = found.pop(objid, None) or []
            if docs:
                seps = [_sep_arcsec(ra, dec, *_doc_position(d)) for d in docs]
                best = int(np.argmin(seps))
                matches[objid][survey] = (docs[best]["_id"], seps[best], _photometry_columns(survey, docs[best]))
        del found

    pieces = []
    for objid, by_survey in matches.items():
        names = {s: by_survey[s][0] if s in by_survey else "" for s in SURVEYS}
        seps = {s: by_survey[s][1] if s in by_survey else math.nan for s in SURVEYS}
        for survey, (_, _, cols) in by_survey.items():
            n = len(cols["mjd"])
            if n == 0:
                continue
            pieces.append(pa.table({
                "decam_objid": pa.array([objid] * n, pa.string()),
                "ztfname": pa.array([names["ZTF"]] * n, pa.string()),
                "lsstname": pa.array([names["LSST"]] * n, pa.string()),
                "survey": pa.array([survey] * n, pa.string()),
                **{c: pa.array(cols[c], SCHEMA.field(c).type) for c in
                   ("kind", "mjd", "filter", "mag", "magerr", "limmag", "snr", "detected")},
                "ztf_sep_arcsec": pa.array(np.full(n, seps["ZTF"], dtype="f4")),
                "lsst_sep_arcsec": pa.array(np.full(n, seps["LSST"], dtype="f4")),
            }, schema=SCHEMA))
    return pa.concat_tables(pieces) if pieces else SCHEMA.empty_table()


# ---------------------------------------------------------------------------
# Resumable parts -> one Parquet file
# ---------------------------------------------------------------------------

def _write_part(table: pa.Table, objids: list, path: pathlib.Path) -> None:
    table = table.replace_schema_metadata({DONE_KEY: json.dumps(objids).encode()})
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)  # atomic: a part is either complete or absent


def load_progress(parts_dir: pathlib.Path) -> tuple:
    """(objids already done, next batch number) from the parts written so far."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    done, numbers = set(), []
    for p in parts_dir.glob("batch-*.parquet"):
        done.update(json.loads(pq.read_schema(p).metadata[DONE_KEY]))
        numbers.append(int(re.match(r"batch-(\d+)", p.name).group(1)))
    return done, max(numbers, default=0) + 1


def merge_parts(parts_dir: pathlib.Path, out_file: pathlib.Path) -> int:
    """Concatenate every part into out_file (atomically), then remove the parts. Returns rows written."""
    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    n_rows = 0
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer:
        for p in sorted(parts_dir.glob("batch-*.parquet")):
            table = pq.read_table(p).replace_schema_metadata(None)
            if table.num_rows:
                writer.write_table(table)
                n_rows += table.num_rows
    os.replace(tmp, out_file)
    shutil.rmtree(parts_dir)
    return n_rows


def load_targets(master: pathlib.Path) -> dict:
    """{objid: (ra, dec)} for every object with a position (first row wins for duplicated objids)."""
    coords = {}
    with open(master, newline="") as f:
        for r in csv.DictReader(f):
            if r["ra"] and r["dec"] and r["objid"] not in coords:
                coords[r["objid"]] = (float(r["ra"]), float(r["dec"]))
    logger.info(f"{len(coords)} objects with RA/Dec in {master}")
    return coords


def run(targets: dict, out_file: pathlib.Path, radius_arcsec: float) -> None:
    parts_dir = out_file.parent / (out_file.stem + "_parts")
    done, batch_no = load_progress(parts_dir)
    todo = [o for o in targets if o not in done]
    logger.info(f"{len(done)} objects already done in {parts_dir}; "
                f"{len(todo)} to query in batches of {BATCH_SIZE} (radius {radius_arcsec}\")")

    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    n_failed = 0
    for i, batch in enumerate(tqdm(batches, desc="BOOM batches", unit="batch", **BAR_KWARGS)):
        try:
            table = process_batch({o: targets[o] for o in batch}, radius_arcsec)
        except Exception as exc:
            logger.error(f"Batch of {len(batch)} objects failed, will be retried on the next run: {exc}")
            n_failed += len(batch)
            continue
        _write_part(table, batch, parts_dir / f"batch-{batch_no:05d}.parquet")
        del table
        batch_no += 1
        if i < len(batches) - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    if n_failed:
        logger.error(f"{n_failed} objects failed; finished batches are kept in {parts_dir}. "
                     f"Re-run to retry them -- {out_file} is written once everything is done.")
        return

    n_rows = merge_parts(parts_dir, out_file)
    table = pq.read_table(out_file, columns=["decam_objid", "ztfname", "lsstname"])
    n_ztf = len({o for o, z in zip(table["decam_objid"].to_pylist(), table["ztfname"].to_pylist()) if z})
    n_lsst = len({o for o, z in zip(table["decam_objid"].to_pylist(), table["lsstname"].to_pylist()) if z})
    logger.info(f"Wrote {n_rows} rows to {out_file}: {n_ztf} objects matched in ZTF, "
                f"{n_lsst} in LSST, out of {len(targets)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ZTF/LSST light curves from BOOM for every master-file object.")
    parser.add_argument("--master", type=pathlib.Path, default=MASTER_FILE)
    parser.add_argument("--out", type=pathlib.Path, default=OUT_FILE)
    parser.add_argument("--radius-arcsec", type=float, default=DEFAULT_RADIUS_ARCSEC)
    args = parser.parse_args()

    run(load_targets(args.master), args.out, args.radius_arcsec)


if __name__ == "__main__":
    main()