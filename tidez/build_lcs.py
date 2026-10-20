"""
Step 2: build the light-curve Parquet files for every object in the master file.

For each FITS file in data/all_objects.csv (from get_data.py), read only the
light-curve columns of the HDU 1 table (memory-mapped, so the 121x121 thumbnails
are never read off disk). Every epoch with a real filter and a finite LIM_MAG5 is
kept, and is either a detection or an upper limit. One pass writes two versions
that differ only in the detection rule:

  data/lightcurves.parquet              detection = STATUS_FPHOT == 'p', SNR > 5,
                                        CNN > 0.7, MAG_FPHOT <= LIM_MAG5
  data/lightcurves_any_status.parquet   detection = SNR > 5, CNN > 0.7,
                                        MAG_FPHOT <= LIM_MAG5 (status ignored)

Every other epoch is an upper limit: mag = magerr = NaN, limmag5 kept, detected = False.
Even a bad ('q') subtraction still records that the field was observed then, at that depth;
the status column says which upper limits came from which status.

Dropped: epochs with a blank/'N/A' filter or no LIM_MAG5 (in those the photometry failed
outright -- ZP, SNR, mag and even MJD_OBS are NaN).

Columns: objid, mjd, filter, mag, magerr, limmag5, detected, status, science_name

Usage:
    python tidez/get_lcs.py
    python tidez/get_lcs.py --master data/all_objects.csv --out-dir data --workers 32
"""

import argparse
import csv
import os
import pathlib
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from tqdm import tqdm

from tidez.logger import get_logger
logger = get_logger("get_lcs")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MASTER_FILE = DATA_DIR / "all_objects.csv"

MIN_SNR = 5.0
MIN_CNN = 0.7
GOOD_STATUS = "p"

# version name -> (output filename, whether a detection must have STATUS_FPHOT == 'p')
VERSIONS = {
    "status_p": ("lightcurves.parquet", True),
    "any_status": ("lightcurves_any_status.parquet", False),
}

SCHEMA = pa.schema([
    ("objid", pa.string()),
    ("mjd", pa.float64()),
    ("filter", pa.string()),
    ("mag", pa.float32()),
    ("magerr", pa.float32()),
    ("limmag5", pa.float32()),
    ("detected", pa.bool_()),
    ("status", pa.string()),
    ("science_name", pa.string()),
])
ROWS_PER_WRITE = 500_000  # buffer this many epochs before writing a Parquet row group

# In a SLURM log (not a terminal) every redraw is appended to the file, so update rarely.
BAR_KWARGS = {} if sys.stderr.isatty() else {"mininterval": 30, "ascii": True}


def _strip(values) -> np.ndarray:
    return np.char.strip(np.asarray(values).astype(str))


def read_lightcurve(task: tuple):
    """Worker: return (objid, {version: {column: array}}) for one FITS file, or (objid, error string)."""
    objid, file_path = task
    try:
        with fits.open(file_path, memmap=True) as hdul:
            table = hdul[1].data
            mjd = np.array(table["MJD_OBS"], dtype="f8")
            band = _strip(table["FILTER"])
            mag = np.array(table["MAG_FPHOT"], dtype="f4")
            magerr = np.array(table["MAGERR_FPHOT"], dtype="f4")
            snr = np.array(table["SNR_FPHOT"], dtype="f4")
            status = _strip(table["STATUS_FPHOT"])
            cnn = np.array(table["CNNSCORE_ALT"], dtype="f4")
            limmag5 = np.array(table["LIM_MAG5"], dtype="f4")
            science_name = _strip(table["SCIENCE_NAME"])
    except Exception as e:
        return objid, f"{type(e).__name__}: {e}"

    keep = (band != "") & (band != "N/A") & np.isfinite(limmag5)
    order = np.flatnonzero(keep)[np.argsort(mjd[keep], kind="stable")]

    # NaN comparisons are False, so NaN SNR/CNN/mag never count as a detection.
    # A "detection" fainter than its own 5-sigma limit isn't really above it -> upper limit.
    quality = (snr > MIN_SNR) & (cnn > MIN_CNN) & (mag <= limmag5)

    common = {
        "mjd": mjd[order],
        "filter": band[order],
        "limmag5": limmag5[order],
        "status": status[order],
        "science_name": science_name[order],
    }
    versions = {}
    for name, (_, require_p) in VERSIONS.items():
        detected = (quality & (status == GOOD_STATUS) if require_p else quality)[order]
        versions[name] = {
            **common,
            "mag": np.where(detected, mag[order], np.nan),
            "magerr": np.where(detected, magerr[order], np.nan),
            "detected": detected,
        }
    return objid, versions


def load_master(path: pathlib.Path) -> list:
    """(objid, file_path) for every readable object in the master file."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    tasks = [(r["objid"], r["file_path"]) for r in rows if r["ra"] and r["dec"]]
    n_skipped = len(rows) - len(tasks)
    if n_skipped:
        logger.warning(f"Skipping {n_skipped} master-file rows with no RA/Dec (header was unreadable)")
    n_dup = len(tasks) - len({objid for objid, _ in tasks})
    if n_dup:
        logger.warning(f"{n_dup} objids appear under more than one program; their epochs will share an objid")
    return tasks


def _to_table(buffer: list) -> pa.Table:
    columns = {name: np.concatenate([lc[name] for _, lc in buffer]) for name in SCHEMA.names if name != "objid"}
    columns["objid"] = np.concatenate([np.full(len(lc["mjd"]), objid, dtype=object) for objid, lc in buffer])
    return pa.table({name: columns[name] for name in SCHEMA.names}, schema=SCHEMA)


class _VersionWriter:
    """Buffers one version's epochs and writes them to its own Parquet file."""

    def __init__(self, out_file: pathlib.Path):
        self.out_file = out_file
        self.tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
        self.writer = pq.ParquetWriter(self.tmp_file, SCHEMA, compression="zstd")
        self.buffer, self.n_buffered = [], 0
        self.n_det = self.n_ul = self.n_objects = 0

    def add(self, objid: str, lc: dict) -> None:
        n_det = int(lc["detected"].sum())
        self.n_det += n_det
        self.n_ul += len(lc["detected"]) - n_det
        self.n_objects += 1
        self.buffer.append((objid, lc))
        self.n_buffered += len(lc["mjd"])
        if self.n_buffered >= ROWS_PER_WRITE:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.writer.write_table(_to_table(self.buffer))
            self.buffer, self.n_buffered = [], 0

    def close(self) -> None:
        self.flush()
        self.writer.close()
        os.replace(self.tmp_file, self.out_file)  # atomic: a crashed run never leaves a half-written file
        logger.info(f"Wrote {self.n_det + self.n_ul} epochs ({self.n_det} detections, {self.n_ul} upper limits) "
                    f"for {self.n_objects} objects to {self.out_file}")


def build_lightcurves(tasks: list, out_dir: pathlib.Path, workers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {name: _VersionWriter(out_dir / filename) for name, (filename, _) in VERSIONS.items()}

    n_empty = 0
    errors = []
    with Pool(workers) as pool:
        results = pool.imap(read_lightcurve, tasks, chunksize=32)
        for objid, versions in tqdm(results, total=len(tasks), desc="Reading light curves", unit="file", **BAR_KWARGS):
            if isinstance(versions, str):
                errors.append((objid, versions))
                continue
            if len(versions["status_p"]["mjd"]) == 0:
                n_empty += 1
                continue
            for name, lc in versions.items():
                writers[name].add(objid, lc)
    for writer in writers.values():
        writer.close()

    for objid, err in errors[:20]:
        logger.warning(f"Could not read {objid}: {err}")
    if len(errors) > 20:
        logger.warning(f"... and {len(errors) - 20} more unreadable files")
    logger.info(f"{n_empty} objects had no usable epochs; {len(errors)} files could not be read")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the light-curve Parquet files from the master file.")
    parser.add_argument("--master", type=pathlib.Path, default=MASTER_FILE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DATA_DIR)
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    args = parser.parse_args()

    tasks = load_master(args.master)
    logger.info(f"Loaded {len(tasks)} objects from {args.master}; reading with {args.workers} workers")
    build_lightcurves(tasks, args.out_dir, args.workers)


if __name__ == "__main__":
    main()