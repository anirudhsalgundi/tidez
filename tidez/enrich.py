"""
Step 4: crossmatch every master-file object against all of BOOM's reference catalogs.

For each batch of objects in data/all_objects.csv, one cone search per catalog
(every catalog BOOM marks as a crossmatch catalog: Gaia_DR3, LSPSC, LSDR10, PS1_DR2,
2MASS_PSC, AllWISE, CatWISE2020, GALEX, DESI_DR1, NED, VSX, milliquas_v8, TNS) returns
every source within --radius-arcsec. *All* matches and *all* their fields are kept,
so new cuts (tighter radii, other catalogs, other columns) never need a re-query;
basic_filtering.py picks what it needs.

These are positional matches on BOOM's copies of the reference catalogs, so they work
for every DECam object, whether or not ZTF/LSST ever saw it.

Output: data/crossmatches/<catalog>.parquet, one row per (DECam object, catalog source):
    decam_objid, rank (0 = nearest), n_matches, sep_arcsec, id, <every catalog field>
Objects with no match in a catalog have no rows in that file. Nested fields are stored
as JSON strings; the GeoJSON 'coordinates' field is dropped (ra/dec are kept).

Resumable: each batch is saved under data/crossmatches_parts/ as it finishes and skipped on
a re-run. Once every object is done the parts are merged into data/crossmatches/ and the
parts directory is removed.

Needs BOOM_USERNAME and BOOM_PASSWORD in the environment.

Usage:
    python tidez/enrich.py
    python tidez/enrich.py --radius-arcsec 10 --catalogs Gaia_DR3 milliquas_v8
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

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from tidez.logger import get_logger
logger = get_logger("enrich")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MASTER_FILE = DATA_DIR / "all_objects.csv"
OUT_DIR = DATA_DIR / "crossmatches"

BOOM_BASE = "https://api.kaboom.caltech.edu"
REQUEST_TIMEOUT = 180
MAX_RETRIES = 4
BATCH_SIZE = 100
SLEEP_BETWEEN_BATCHES = 0.5  # seconds
DEFAULT_RADIUS_ARCSEC = 5.0

DROP_FIELDS = {"coordinates"}  # GeoJSON point, duplicates ra/dec
LEAD_TYPES = {"decam_objid": pa.string(), "rank": pa.int64(), "n_matches": pa.int64(),
              "sep_arcsec": pa.float64(), "id": pa.string(), "ra": pa.float64(), "dec": pa.float64()}
LEAD_COLUMNS = list(LEAD_TYPES)

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


def _request(method: str, path: str, **kwargs):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if _token is None:
                _login()
            resp = _session.request(method, f"{BOOM_BASE}{path}", timeout=REQUEST_TIMEOUT,
                                    headers={"Authorization": f"Bearer {_token}"}, **kwargs)
            if resp.status_code == 401:
                logger.warning("BOOM token expired/rejected, re-authenticating.")
                _login()
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(f"BOOM {path} failed (attempt {attempt}/{MAX_RETRIES}): {exc}. Retrying in {wait}s.")
            time.sleep(wait)
    raise RuntimeError(f"BOOM {path} failed after {MAX_RETRIES} attempts: {last_exc}")


def crossmatch_catalogs() -> list:
    return [c["name"] for c in _request("GET", "/catalogs")["data"] if c.get("crossmatch")]


def cone_search(coords: dict, catalog: str, radius_arcsec: float) -> dict:
    """{objid: [catalog docs within radius]}; objids with no match are absent."""
    payload = {
        "catalog_name": catalog,
        "object_coordinates": {objid: [ra, dec] for objid, (ra, dec) in coords.items()},
        "radius": radius_arcsec,
        "unit": "Arcseconds",
    }
    return _request("POST", "/queries/cone_search", json=payload).get("data", {})


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def _sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    a = (math.sin((dec2 - dec1) / 2) ** 2
         + math.cos(dec1) * math.cos(dec2) * math.sin((ra2 - ra1) / 2) ** 2)
    return math.degrees(2 * math.asin(math.sqrt(min(1.0, a)))) * 3600.0


def _doc_radec(doc: dict) -> tuple:
    if doc.get("ra") is not None and doc.get("dec") is not None:
        return float(doc["ra"]), float(doc["dec"])
    lon, lat = doc["coordinates"]["radec_geojson"]["coordinates"]  # GeoJSON lon = ra - 180
    return lon + 180.0, lat


def match_rows(coords: dict, found: dict) -> list:
    """Every match of every object, nearest first, with all catalog fields."""
    rows = []
    for objid, docs in found.items():
        if objid not in coords or not docs:
            continue
        ra, dec = coords[objid]
        seps = [_sep_arcsec(ra, dec, *_doc_radec(d)) for d in docs]
        for rank, i in enumerate(sorted(range(len(docs)), key=seps.__getitem__)):
            doc = docs[i]
            row = {"decam_objid": objid, "rank": rank, "n_matches": len(docs), "sep_arcsec": seps[i],
                   "id": doc.get("_id")}
            for key, value in doc.items():
                if key == "_id" or key in DROP_FIELDS:
                    continue
                row[key] = json.dumps(value, default=str) if isinstance(value, (dict, list)) else value
            rows.append(row)
    return rows


def _rows_to_table(rows: list) -> pa.Table:
    """Arrow table from rows whose fields vary by catalog; a column whose values mix
    incompatible types (e.g. str and int) is stored as strings."""
    names = list(dict.fromkeys(LEAD_COLUMNS + [k for r in rows for k in r]))
    columns = {}
    for name in names:
        values = [r.get(name) for r in rows]
        try:
            columns[name] = pa.array(values)
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, OverflowError):
            columns[name] = pa.array([None if v is None else str(v) for v in values], pa.string())
    return pa.table(columns)


# ---------------------------------------------------------------------------
# Resumable parts -> one Parquet file per catalog
# ---------------------------------------------------------------------------

def _write_atomic(table: pa.Table, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def load_progress(parts_dir: pathlib.Path) -> tuple:
    """(done objids, next batch number). A batch counts only once its .done marker exists;
    catalog parts of an unfinished batch (crash mid-batch) are removed so it is redone."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    done, finished = set(), set()
    for marker in parts_dir.glob("batch-*.done"):
        done.update(json.loads(marker.read_text()))
        finished.add(marker.stem)
    for part in parts_dir.glob("*/batch-*.parquet"):
        if part.stem not in finished:
            logger.warning(f"Removing part of unfinished batch: {part}")
            part.unlink()
    numbers = [int(re.match(r"batch-(\d+)", name).group(1)) for name in finished]
    return done, max(numbers, default=0) + 1


def _common_type(types: list) -> pa.DataType:
    types = [t for t in types if not pa.types.is_null(t)]
    if not types:
        return pa.null()
    if all(t == types[0] for t in types):
        return types[0]
    if all(pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t) for t in types):
        return pa.float64()
    return pa.string()


def merge_parts(parts_dir: pathlib.Path, out_dir: pathlib.Path, catalogs: list, radius_arcsec: float) -> dict:
    """Merge each catalog's parts into out_dir/<catalog>.parquet -- an empty file (lead columns
    only) for a catalog with no matches at all. Returns {catalog: rows}."""
    counts = {}
    for catalog in catalogs:
        cat_dir = parts_dir / catalog
        parts = sorted(cat_dir.glob("batch-*.parquet"))
        schemas = [pq.read_schema(p) for p in parts]
        names = list(dict.fromkeys(LEAD_COLUMNS + [n for s in schemas for n in s.names]))
        types = {n: _common_type([s.field(n).type for s in schemas if n in s.names]) for n in names}
        target = pa.schema([(n, LEAD_TYPES[n] if pa.types.is_null(t) and n in LEAD_TYPES else t)
                            for n, t in types.items()])
        tables = []
        for p in parts:
            t = pq.read_table(p)
            tables.append(pa.table({
                f.name: (t[f.name].cast(f.type) if f.name in t.column_names
                         else pa.nulls(t.num_rows, f.type)) for f in target}))
        merged = pa.concat_tables(tables) if tables else target.empty_table()
        merged = merged.replace_schema_metadata({b"radius_arcsec": str(radius_arcsec).encode()})
        _write_atomic(merged, out_dir / f"{catalog}.parquet")
        counts[catalog] = merged.num_rows
    shutil.rmtree(parts_dir)
    return counts


def load_targets(master: pathlib.Path) -> dict:
    """{objid: (ra, dec)} for every object with a position (first row wins for duplicated objids)."""
    coords = {}
    with open(master, newline="") as f:
        for r in csv.DictReader(f):
            if r["ra"] and r["dec"] and r["objid"] not in coords:
                coords[r["objid"]] = (float(r["ra"]), float(r["dec"]))
    logger.info(f"{len(coords)} objects with RA/Dec in {master}")
    return coords


def run(targets: dict, out_dir: pathlib.Path, catalogs: list, radius_arcsec: float) -> None:
    parts_dir = out_dir.parent / (out_dir.name + "_parts")
    done, batch_no = load_progress(parts_dir)
    todo = [o for o in targets if o not in done]
    logger.info(f"Catalogs: {', '.join(catalogs)}")
    logger.info(f"{len(done)} objects already done in {parts_dir}; "
                f"{len(todo)} to query in batches of {BATCH_SIZE} (radius {radius_arcsec}\")")

    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    n_failed = 0
    for i, batch in enumerate(tqdm(batches, desc="BOOM catalog batches", unit="batch", **BAR_KWARGS)):
        coords = {o: targets[o] for o in batch}
        name = f"batch-{batch_no:05d}"
        try:
            for catalog in catalogs:
                rows = match_rows(coords, cone_search(coords, catalog, radius_arcsec))
                if rows:
                    _write_atomic(_rows_to_table(rows), parts_dir / catalog / f"{name}.parquet")
        except Exception as exc:
            logger.error(f"Batch of {len(batch)} objects failed, will be retried on the next run: {exc}")
            for part in parts_dir.glob(f"*/{name}.parquet"):
                part.unlink()
            n_failed += len(batch)
            continue
        (parts_dir / f"{name}.done").write_text(json.dumps(batch))  # marks the whole batch finished
        batch_no += 1
        if i < len(batches) - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    if n_failed:
        logger.error(f"{n_failed} objects failed; finished batches are kept in {parts_dir}. "
                     f"Re-run to retry them -- {out_dir} is written once everything is done.")
        return

    counts = merge_parts(parts_dir, out_dir, catalogs, radius_arcsec)
    for catalog in catalogs:
        path = out_dir / f"{catalog}.parquet"
        n_obj = len(set(pq.read_table(path, columns=["decam_objid"])["decam_objid"].to_pylist()))
        logger.info(f"  {catalog:14s} {counts.get(catalog, 0):>9d} matches for {n_obj:>7d} of {len(targets)} objects")
    logger.info(f"Wrote {len(counts)} catalog files to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crossmatch master-file objects against BOOM's reference catalogs.")
    parser.add_argument("--master", type=pathlib.Path, default=MASTER_FILE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--radius-arcsec", type=float, default=DEFAULT_RADIUS_ARCSEC)
    parser.add_argument("--catalogs", nargs="+", default=None,
                        help="Catalogs to query (default: every catalog BOOM marks as a crossmatch catalog).")
    args = parser.parse_args()

    targets = load_targets(args.master)
    catalogs = args.catalogs or crossmatch_catalogs()
    run(targets, args.out_dir, catalogs, args.radius_arcsec)


if __name__ == "__main__":
    main()