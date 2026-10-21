"""
Step 1: build the master index of every candidate object.

Globs each survey directory for candidate FITS files (<n_detections>_<objid>.fits),
keeps the file with the most detections per (program, objid), and reads RA_OBJ / DEC_OBJ
from the HDU 1 header plus the MJD_OBS column of the HDU 1 table.

Output: data/all_objects.parquet with columns objid, program, file_path, ra, dec, mjds,
n_epochs, where program is DESIRT or DECAM_ZTF_EXPERIMENT depending on the search
directory. Every run re-reads all files and replaces the previous master.

Usage:
    python tidez/create_masterdb.py
    python tidez/create_masterdb.py --workers 32
"""

import argparse
import glob
import os
import pathlib
import re
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from tidez.logger import get_logger
logger = get_logger("create_masterdb")

# (program, search directory)
SEARCH_DIRS = [
    ("DESIRT", pathlib.Path("/ocean/projects/phy250012p/shared/3DTS/DECAM/DESIRT/SEARCH/3DTS/")),
    ("DESIRT", pathlib.Path("/ocean/projects/phy250012p/shared/3DTS/DECAM/DESIRT_21A_23B")),
    ("DECAM_ZTF_EXPERIMENT", pathlib.Path("/ocean/projects/phy250024p/shared/DECAM_ZTF/DECAM_FOLLOWUP/SEARCH/DECAM_ZTF_ALL_2025B")),
    ("DECAM_ZTF_EXPERIMENT", pathlib.Path("/ocean/projects/phy250024p/shared/DECAM_ZTF/DECAM_FOLLOWUP/SEARCH/DECAM_ZTF_ALL")),
    ("DECAM_ZTF_EXPERIMENT", pathlib.Path("/ocean/projects/phy250024p/shared/DECAM_ZTF/DECAM_FOLLOWUP/SEARCH/DECAM_ZTF")),
]
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MASTER_FILE = DATA_DIR / "all_objects.parquet"

FILENAME_RE = re.compile(r'^(\d+)_(.+)\.fits$')
COLUMNS = ["objid", "program", "file_path", "ra", "dec", "mjds"]


def fetch_all_files() -> list:
    """Glob every search dir one top-level subdirectory at a time, so there is a progress bar."""
    all_files = []
    for program, root in SEARCH_DIRS:
        if not root.is_dir():
            logger.warning(f"{program}: directory not found, skipping: {root}")
            continue
        subdirs = sorted(p for p in root.iterdir() if p.is_dir())
        logger.info(f"{program}: globbing {len(subdirs)} subdirectories of {root}")
        n_before = len(all_files)
        for subdir in tqdm(subdirs, desc=f"Globbing {root.name}", unit="dir"):
            all_files.extend((program, f) for f in glob.glob(str(subdir / "**/*.fits"), recursive=True))
        logger.info(f"{program}: found {len(all_files) - n_before} FITS files in {root}")
    return all_files


def get_ndet_objid(file_path: str) -> tuple:
    m = FILENAME_RE.match(os.path.basename(file_path))
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def parse_fits_filename(path):
    """Kept for build_h5.py: (n_detections, objid), or None if the name doesn't match."""
    n_detections, objid = get_ndet_objid(str(path))
    return None if objid is None else (n_detections, objid)


def get_complete_lightcurves(all_files: list) -> dict:
    """Keep the max-detections file per (program, objid). Returns {(program, objid): file_path}."""
    best = {}
    n_unparsed = 0
    for program, file_path in tqdm(all_files, desc="Picking max-detections file", unit="file"):
        n_detections, objid = get_ndet_objid(file_path)
        if objid is None:
            n_unparsed += 1
            continue
        key = (program, objid)
        if key not in best or n_detections > best[key][0]:
            best[key] = (n_detections, file_path)
    if n_unparsed:
        logger.warning(f"Skipped {n_unparsed} files whose names did not match <n>_<objid>.fits")
    return {key: file_path for key, (_, file_path) in best.items()}


def read_row(task: tuple) -> dict:
    """Worker: RA/DEC from the HDU 1 header, MJDs from the HDU 1 table. Unreadable files get None and []."""
    program, objid, file_path = task
    row = {"objid": str(objid), "program": program, "file_path": str(file_path), "ra": None, "dec": None, "mjds": []}
    try:
        with fits.open(file_path) as hdul:
            header = hdul[1].header
            data = hdul[1].data
            row["ra"] = header.get("RA_OBJ")
            row["dec"] = header.get("DEC_OBJ")
            if data is not None and "MJD_OBS" in data.columns.names:
                row["mjds"] = np.asarray(data["MJD_OBS"], dtype=float).tolist()
            else:
                logger.warning(f"No MJD_OBS column in {file_path}")
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {type(e).__name__}: {e}")
    return row


def build_rows(best_files: dict, workers: int) -> list:
    tasks = [(program, objid, file_path) for (program, objid), file_path in best_files.items()]
    logger.info(f"Reading {len(tasks)} files with {workers} workers")
    with Pool(workers) as pool:
        rows = list(tqdm(pool.imap_unordered(read_row, tasks, chunksize=64),
                         total=len(tasks), desc="Reading light curves", unit="file"))
    rows.sort(key=lambda r: (r["program"], r["objid"]))
    return rows


def write_to_parquet(rows: list, out_path: pathlib.Path) -> None:
    """Write rows to parquet atomically: write a .tmp file, then rename over the old one."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df["n_epochs"] = df["mjds"].apply(len)

    n_bad_coords = int((df["ra"].isna() | df["dec"].isna()).sum())
    n_no_epochs = int((df["n_epochs"] == 0).sum())
    logger.info(f"Rows per program: {df['program'].value_counts().to_dict()}")
    logger.info(f"Epochs per object: median {df['n_epochs'].median():.0f}, max {df['n_epochs'].max()}")
    if n_bad_coords or n_no_epochs:
        logger.warning(f"{n_bad_coords} rows missing ra/dec, {n_no_epochs} rows with no epochs")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)  # atomic: a crashed run never leaves a half-written master
    logger.info(f"Wrote {len(df)} rows to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the master index of candidate objects.")
    parser.add_argument("--out", type=pathlib.Path, default=MASTER_FILE)
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    args = parser.parse_args()

    t0 = time.time()
    all_files = fetch_all_files()
    logger.info(f"Total FITS files found: {len(all_files)} ({time.time() - t0:.0f}s)")
    if not all_files:  # don't replace a good master with an empty one (e.g. run off-cluster)
        raise SystemExit("No FITS files found in any search directory -- is /ocean mounted? Master not written.")
    best_files = get_complete_lightcurves(all_files)
    logger.info(f"Total unique objects: {len(best_files)}")

    rows = build_rows(best_files, args.workers)
    write_to_parquet(rows, args.out)
    logger.info(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
