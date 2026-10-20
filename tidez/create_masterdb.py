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


def read_row(task: tuple) -> dict:
    """Worker: read RA/DEC from the HDU 1 header. Unreadable files get empty ra/dec."""
    program, objid, file_path = task
    row = {"objid": objid, "program": program, "file_path": file_path, "ra": "", "dec": ""}
    try:
        header = fits.getheader(file_path, ext=1)
        row["ra"], row["dec"] = header.get("RA_OBJ", ""), header.get("DEC_OBJ", "")
    except Exception as e:
        logger.warning(f"Could not read header of {file_path}: {type(e).__name__}: {e}")
    return row


def load_previous_master(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {(r.get("program", ""), r["objid"]): r for r in csv.DictReader(f)}


def build_rows(best_files: dict, previous: dict, workers: int, full: bool) -> list:
    rows, tasks = [], []
    for (program, objid), file_path in best_files.items():
        old = previous.get((program, objid))
        if not full and old and old["file_path"] == file_path and old["ra"] and old["dec"]:
            rows.append(old)
        else:
            tasks.append((program, objid, file_path))

    logger.info(f"Reusing {len(rows)} unchanged rows; reading {len(tasks)} headers with {workers} workers")
    with Pool(workers) as pool:
        rows.extend(tqdm(pool.imap_unordered(read_row, tasks, chunksize=64),
                         total=len(tasks), desc="Reading headers", unit="file"))
    rows.sort(key=lambda r: (r["program"], r["objid"]))
    return rows


def write_to_csv(rows: list, output_file: pathlib.Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with open(tmp_file, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_file, output_file)  # atomic: a crashed run never leaves a half-written master
    logger.info(f"Wrote {len(rows)} rows to {output_file}")


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


if __name__ == "__main__":
    main()
