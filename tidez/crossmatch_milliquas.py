"""
Cross-match every object already in the master H5 (see build_h5.py) against
the Milliquas (Million Quasars) catalog via the CDS X-Match service, and
write the result back into the same H5 file. Like crossmatch_boom.py, this
is a separate, resumable, network-bound pass -- the AGN filter
(afterglow_hunt.py) just reads whatever is already stored.

Unlike crossmatch_boom.py's per-batch cone searches, CDS X-Match matches an
entire uploaded table against a VizieR catalog in a single HTTP request, so
this queries all objects in --chunk-size sized chunks rather than
individually -- far fewer requests than a per-object cone search (this is
exactly the pattern WISE crossmatching lacked, which is why it was too slow
to use).

--milliquas-catalog defaults to the current Milliquas VizieR table id at the
time this was written; if CDS republishes a newer version under a new id,
override it on the command line rather than editing this file.

Written layout (added under each existing /<objid>/ group):
  /<objid>/milliquas/   attrs: matched (bool), and if matched:
                                separation_arcsec, plus whatever columns
                                the Milliquas VizieR table returns (Name,
                                Type, z, Qpct, Rmag, Bmag, ... stored as
                                string attrs, schema not hardcoded so a
                                catalog version change doesn't break this).

Resumable: an object is skipped if its /milliquas group already exists
(whether or not it matched).

Usage:
    python -m tidez.crossmatch_milliquas
    python -m tidez.crossmatch_milliquas --h5 data/tidez_master.h5 --radius-arcsec 3
"""

import argparse
import csv
import io
import pathlib
import time

import h5py
import requests
from tqdm import tqdm

from tidez.logger import get_logger

logger = get_logger("crossmatch_milliquas")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_H5 = DATA_DIR / "tidez_master.h5"

XMATCH_URL = "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync"
DEFAULT_MILLIQUAS_CATALOG = "vizier:VII/294/catalog"
DEFAULT_RADIUS_ARCSEC = 3.0
DEFAULT_CHUNK_SIZE = 5000
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3

# Columns from our own uploaded table / the xmatch machinery, not from the
# Milliquas catalog itself -- everything else in a response row is stored
# verbatim as a Milliquas attribute so a catalog schema change doesn't break
# this script.
_RESERVED_COLUMNS = {"angDist", "ra", "dec", "objid"}


def _xmatch_chunk(rows, radius_arcsec, catalog):
    """rows: list of (objid, ra, dec). Returns {objid: {col: value, ...}}
    for objids with a match (angDist included), nearest match only."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["objid", "ra", "dec"])
    for objid, ra, dec in rows:
        writer.writerow([objid, ra, dec])
    csv_bytes = buf.getvalue().encode("utf-8")

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                XMATCH_URL,
                data={
                    "request": "xmatch",
                    "distMaxArcsec": radius_arcsec,
                    "RESPONSEFORMAT": "csv",
                    "colRA1": "ra",
                    "colDec1": "dec",
                    "cat2": catalog,
                },
                files={"cat1": ("cat1.csv", csv_bytes, "text/csv")},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                f"CDS X-Match request failed (attempt {attempt}/{MAX_RETRIES}) "
                f"for {len(rows)} objects: {exc}. Retrying in {wait}s."
            )
            time.sleep(wait)
    else:
        raise RuntimeError(f"CDS X-Match request failed after {MAX_RETRIES} attempts: {last_exc}")

    best_by_objid = {}
    for row in csv.DictReader(io.StringIO(resp.text)):
        objid = row["objid"]
        ang_dist = float(row["angDist"])
        if objid not in best_by_objid or ang_dist < best_by_objid[objid]["angDist"]:
            best_by_objid[objid] = {**row, "angDist": ang_dist}

    return best_by_objid


def write_milliquas_match(h5file, objid, match):
    group = h5file[objid].create_group("milliquas")
    group.attrs["matched"] = match is not None
    if match is None:
        return
    group.attrs["separation_arcsec"] = match["angDist"]
    for key, value in match.items():
        if key in _RESERVED_COLUMNS:
            continue
        group.attrs[key] = value if value is not None else ""


def run(
    h5_path: pathlib.Path,
    radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
    catalog: str = DEFAULT_MILLIQUAS_CATALOG,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int = None,
):
    with h5py.File(h5_path, "a") as h5file:
        objids = sorted(k for k in h5file.keys() if k != "catalog")
        if limit is not None:
            objids = objids[:limit]

        already_done = {objid for objid in objids if "milliquas" in h5file[objid]}
        todo = [objid for objid in objids if objid not in already_done]
        logger.info(
            f"{len(objids)} objects in {h5_path}; "
            f"{len(already_done)} already crossmatched against Milliquas, will be skipped."
        )

        chunks = [todo[i:i + chunk_size] for i in range(0, len(todo), chunk_size)]
        n_matched = 0
        n_processed = 0
        for i, chunk_objids in enumerate(tqdm(chunks, desc="Crossmatching with Milliquas", unit="chunk"), start=1):
            rows = [
                (objid, float(h5file[objid].attrs["ra"]), float(h5file[objid].attrs["dec"]))
                for objid in chunk_objids
            ]
            try:
                matches = _xmatch_chunk(rows, radius_arcsec, catalog)
            except Exception:
                logger.error(f"Failed to crossmatch chunk {i}/{len(chunks)} ({len(chunk_objids)} objects)", exc_info=True)
                continue

            for objid in chunk_objids:
                match = matches.get(objid)
                write_milliquas_match(h5file, objid, match)
                n_processed += 1
                if match is not None:
                    n_matched += 1

            h5file.flush()
            logger.info(f"Flushed after chunk {i}/{len(chunks)} (processed={n_processed}, matched={n_matched})")

    logger.info(
        f"Done. processed={n_processed}, matched={n_matched}, "
        f"skipped={len(already_done)}, total_objects={len(objids)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cross-match master H5 objects against the Milliquas catalog via CDS X-Match."
    )
    parser.add_argument("--h5", type=pathlib.Path, default=DEFAULT_H5,
                         help="Master H5 file to read objects from and write matches into.")
    parser.add_argument("--radius-arcsec", type=float, default=DEFAULT_RADIUS_ARCSEC,
                         help="Cross-match radius in arcseconds (default: 3.0).")
    parser.add_argument("--milliquas-catalog", type=str, default=DEFAULT_MILLIQUAS_CATALOG,
                         dest="catalog",
                         help=f"VizieR catalog id for X-Match (default: {DEFAULT_MILLIQUAS_CATALOG}). "
                              "Override if CDS has republished Milliquas under a new id.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                         help="Objects per X-Match request (default: 5000).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N objects (for testing).")
    args = parser.parse_args()

    run(args.h5, args.radius_arcsec, args.catalog, args.chunk_size, args.limit)


if __name__ == "__main__":
    main()
