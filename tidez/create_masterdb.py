"""
Step 1: build the master index of every candidate object.

Globs each survey directory for candidate FITS files (<n_detections>_<objid>.fits),
keeps the file with the most detections per (program, objid), and reads only the HDU 1
*header* of that file (RA_OBJ / DEC_OBJ) -- never the table or thumbnails.

Output: data/all_objects.csv with columns objid, program, file_path, ra, dec,
where program is DESIRT or DECAM_ZTF_EXPERIMENT depending on the search directory.
Re-runs reuse the previous row for any object whose file_path is unchanged
(the filename changes when an object gains detections); --full re-reads all.

Usage:
    python tidez/create_masterdb.py
    python tidez/create_masterdb.py --workers 32 --full
"""

import argparse
import csv
import glob
import os
import pathlib
import re
from multiprocessing import Pool
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import pandas as pd
from pathlib import Path
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
MASTER_FILE = DATA_DIR / "all_objects.csv"

FILENAME_RE = re.compile(r'^(\d+)_(.+)\.fits$')
FIELDNAMES = ["objid", "program", "file_path", "ra", "dec"]


def fetch_all_files() -> list:
    all_files = []
    for program, root in SEARCH_DIRS:
        if not root.is_dir():
            logger.warning(f"{program}: directory not found, skipping: {root}")
            continue
        logger.info(f"{program}: globbing {root} (no progress until it finishes)...")
        files = glob.glob(str(root / "*/**/*.fits"), recursive=True)
        logger.info(f"{program}: found {len(files)} FITS files in {root}")
        all_files.extend((program, f) for f in files)
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



def build_rows(best_files: dict, previous: dict, workers: int, full: bool) -> list:
    rows, tasks = [], []
    for (program, objid), file_path in best_files.items():
        old = previous.get((program, objid))
        reusable = (
            not full
            and old is not None
            and old["file_path"] == str(file_path)
            and pd.notna(old["ra"])
            and pd.notna(old["dec"])
            and "mjds" in old
            and len(old["mjds"]) > 0
        )
        if reusable:
            old["mjds"] = list(old["mjds"])
            rows.append(old)
        else:
            tasks.append((program, objid, file_path))

    logger.info(f"Reusing {len(rows)} unchanged rows; reading {len(tasks)} files with {workers} workers")
    with Pool(workers) as pool:
        rows.extend(tqdm(pool.imap_unordered(read_row, tasks, chunksize=64),
                         total=len(tasks), desc="Reading light curves", unit="file"))
    rows.sort(key=lambda r: (r["program"], r["objid"]))
    return rows


def read_row(task: tuple) -> dict:
    """Worker: RA/DEC from the HDU 1 header, MJDs from the HDU 1 table. Unreadable files get None and []."""
    program, objid, file_path = task
    row = {"objid": str(objid), "program": program, "file_path": str(file_path), "ra": None, "dec": None, "mjds": []}
    try:
        with fits.open(file_path, memmap=False) as hdul:
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


def write_to_parquet(rows: list, out_path: Path):
    """Write rows to parquet atomically: write a .tmp file, then rename over the old one."""
    df = pd.DataFrame(rows, columns=["objid", "program", "file_path", "ra", "dec", "mjds"])
    df["objid"] = df["objid"].astype(str)
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df["mjds"] = df["mjds"].apply(lambda m: [] if m is None else [float(x) for x in m])
    df["n_epochs"] = df["mjds"].apply(len)

    n_bad_coords = int(df["ra"].isna().sum() | 0) if False else int((df["ra"].isna() | df["dec"].isna()).sum())
    n_no_epochs = int((df["n_epochs"] == 0).sum())
    logger.info(f"Writing {len(df)} rows to {out_path} ({n_bad_coords} missing ra/dec, {n_no_epochs} with no epochs)")

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the master index of candidate objects.")
    parser.add_argument("--out", type=pathlib.Path, default=MASTER_FILE)
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    parser.add_argument("--full", action="store_true",
                        help="Re-read every header, even for objects unchanged since the last run.")
    args = parser.parse_args()

    previous = load_previous_master(args.out)
    all_files = fetch_all_files()
    logger.info(f"Total FITS files found: {len(all_files)}")
    best_files = get_complete_lightcurves(all_files)
    logger.info(f"Total unique objects: {len(best_files)}")

    rows = build_rows(best_files, previous, args.workers, args.full)
    write_to_csv(rows, args.out)
    create_paraquet_from_csv(args.out, args.out.with_suffix(".parquet"))


if __name__ == "__main__":
    main()
