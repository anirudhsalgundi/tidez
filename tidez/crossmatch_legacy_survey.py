"""
Fetch a Legacy Survey (DR10) cutout image for every object already in the
master H5 (see build_h5.py) and write the raw image bytes back into the
same H5 file. Like crossmatch_boom.py, this is a separate, resumable,
network-bound pass so the slow part is cached once and the scanner webapp
can just read what's already stored -- no live network calls per page view.

Written layout (added under each existing /<objid>/ group):
  /<objid>/legacy_survey/            attrs: layer, pixscale, size_pix
  /<objid>/legacy_survey/cutout_jpg   1-D uint8 dataset: raw JPEG bytes

Resumable: an object is skipped if its /legacy_survey group already exists.

Usage:
    python -m tidez.crossmatch_legacy_survey
    python -m tidez.crossmatch_legacy_survey --h5 data/tidez_master.h5 --layer ls-dr10
"""

import argparse
import pathlib
import time

import h5py
import numpy as np
import requests
from tqdm import tqdm

from tidez.logger import get_logger

logger = get_logger("crossmatch_legacy_survey")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_H5 = DATA_DIR / "tidez_master.h5"

CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.jpg"
DEFAULT_LAYER = "ls-dr10"
DEFAULT_PIXSCALE = 0.262  # arcsec/pixel, native DECam scale
DEFAULT_SIZE_PIX = 150
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
DEFAULT_SLEEP = 0.2  # seconds between requests, polite to the shared service


def fetch_cutout_bytes(ra: float, dec: float, layer: str, pixscale: float, size_pix: int) -> bytes:
    params = {"ra": ra, "dec": dec, "layer": layer, "pixscale": pixscale, "size": size_pix}

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(CUTOUT_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                f"Legacy Survey cutout fetch failed (attempt {attempt}/{MAX_RETRIES}) "
                f"for ra={ra}, dec={dec}: {exc}. Retrying in {wait}s."
            )
            time.sleep(wait)

    raise RuntimeError(f"Legacy Survey cutout fetch failed after {MAX_RETRIES} attempts: {last_exc}")


def write_legacy_cutout(h5file, objid, jpg_bytes: bytes, layer: str, pixscale: float, size_pix: int):
    group = h5file[objid].create_group("legacy_survey")
    group.attrs["layer"] = layer
    group.attrs["pixscale"] = pixscale
    group.attrs["size_pix"] = size_pix
    group.create_dataset(
        "cutout_jpg",
        data=np.frombuffer(jpg_bytes, dtype="uint8"),
        compression="gzip", compression_opts=4,
    )


def run(
    h5_path: pathlib.Path,
    layer: str = DEFAULT_LAYER,
    pixscale: float = DEFAULT_PIXSCALE,
    size_pix: int = DEFAULT_SIZE_PIX,
    sleep: float = DEFAULT_SLEEP,
    limit: int = None,
):
    with h5py.File(h5_path, "a") as h5file:
        objids = sorted(k for k in h5file.keys() if k != "catalog")
        if limit is not None:
            objids = objids[:limit]

        already_done = {objid for objid in objids if "legacy_survey" in h5file[objid]}
        todo = [objid for objid in objids if objid not in already_done]
        logger.info(
            f"{len(objids)} objects in {h5_path}; "
            f"{len(already_done)} already have a Legacy Survey cutout, will be skipped."
        )

        n_done = 0
        n_failed = 0
        for objid in tqdm(todo, desc="Fetching Legacy Survey cutouts", unit="obj"):
            ra = float(h5file[objid].attrs["ra"])
            dec = float(h5file[objid].attrs["dec"])
            try:
                jpg_bytes = fetch_cutout_bytes(ra, dec, layer, pixscale, size_pix)
                write_legacy_cutout(h5file, objid, jpg_bytes, layer, pixscale, size_pix)
                n_done += 1
            except Exception:
                logger.error(f"Failed to fetch Legacy Survey cutout for {objid} (ra={ra}, dec={dec})", exc_info=True)
                n_failed += 1

            if n_done % 50 == 0:
                h5file.flush()

            time.sleep(sleep)

        h5file.flush()

    logger.info(
        f"Done. fetched={n_done}, failed={n_failed}, skipped={len(already_done)}, "
        f"total_objects={len(objids)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and store a Legacy Survey cutout for every object in the master H5."
    )
    parser.add_argument("--h5", type=pathlib.Path, default=DEFAULT_H5,
                         help="Master H5 file to read objects from and write cutouts into.")
    parser.add_argument("--layer", type=str, default=DEFAULT_LAYER,
                         help="Legacy Survey data release layer (default: ls-dr10).")
    parser.add_argument("--pixscale", type=float, default=DEFAULT_PIXSCALE,
                         help="Cutout pixel scale in arcsec/pixel (default: 0.262).")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE_PIX, dest="size_pix",
                         help="Cutout size in pixels (default: 150).")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                         help="Seconds to sleep between requests (default: 0.2).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N objects (for testing).")
    args = parser.parse_args()

    run(args.h5, args.layer, args.pixscale, args.size_pix, args.sleep, args.limit)


if __name__ == "__main__":
    main()
