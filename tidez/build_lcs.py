"""
Step 2: build the DECam light curve of every object in the master file.

Reads each object's candidate FITS file (the max-detections one, from create_masterdb.py)
and keeps EVERY epoch as it is in the file: no SNR, STATUS_FPHOT, CNN or filter cuts, and
no upper-limit / detection decision. Those cuts are left for later, so the raw quantities
they need are kept with the photometry. Points are sorted by MJD (NaN MJDs last).

Per point, from the HDU 1 table (forced photometry, except the CNN score):
    mjds           MJD_OBS
    filters        FILTER        (includes 'N/A' where the file says so)
    mags           MAG_FPHOT     (NaN where forced photometry gave no mag)
    magerrs        MAGERR_FPHOT
    snrs           SNR_FPHOT
    statuses       STATUS_FPHOT  ('p', 'm', 'q', ... as written by the pipeline)
    limiting_mags  LIM_MAG5      (5-sigma limiting magnitude)
    cnn_scores     CNNSCORE_ALT

Plus the science / template / difference cutouts (121 x 121, float32) of the brightest point,
the one with the smallest MAG_FPHOT whatever its status or SNR; brightest_index says which
point that is. Objects with no finite MAG_FPHOT have no cutouts (brightest_index = -1).

Output: data/decam_lightcurves.parquet, one row per object:
    objid, program, file_path, ra, dec, n_points,
    mjds, filters, mags, magerrs, snrs, statuses, limiting_mags, cnn_scores,
    brightest_index, brightest_mjd, brightest_filter, brightest_mag,
    cutout_science, cutout_template, cutout_difference,   (flat lists; reshape(121, 121))
    read_error   (None, or why the FITS file could not be read; then every list is empty)

Resumable: each chunk is saved to data/decam_lightcurves_parts/ as it finishes, and a re-run
skips objects already saved. Once every object is done, the parts are merged into the output
and the parts directory is removed.

Usage:
    python tidez/build_lcs.py                       # every object
    python tidez/build_lcs.py --objid-prefix T      # only objects whose objid starts with T
    python tidez/build_lcs.py --workers 32
"""

import argparse
import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from astropy.io import fits
from multiprocessing import Pool
from tqdm import tqdm

from tidez.logger import get_logger
logger = get_logger("build_lcs")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MASTER_FILE = DATA_DIR / "all_objects.parquet"
LIGHTCURVES_FILE = DATA_DIR / "decam_lightcurves.parquet"

CHUNK_SIZE = 500  # objects per saved part; each carries three 121 x 121 cutouts
DEFAULT_OBJID_PREFIX = ""  # every object; pass --objid-prefix T for transients only
CUTOUT_SIDE = 121

# light-curve field -> FITS column
LIGHTCURVE_COLUMNS = {
    "mjds": "MJD_OBS",
    "filters": "FILTER",
    "mags": "MAG_FPHOT",
    "magerrs": "MAGERR_FPHOT",
    "snrs": "SNR_FPHOT",
    "statuses": "STATUS_FPHOT",
    "limiting_mags": "LIM_MAG5",
    "cnn_scores": "CNNSCORE_ALT",
}
TEXT_FIELDS = ("filters", "statuses")
# cutout name -> FITS column
CUTOUT_COLUMNS = {
    "cutout_science": "PixA_THUMB_SCI",
    "cutout_template": "PixA_THUMB_TEMP",
    "cutout_difference": "PixA_THUMB_DIFF",
}
# Fixed column types, so every part (even one where e.g. no object has cutouts) matches and
# the parts can be streamed into one file without loading them all.
SCHEMA = pa.schema([
    ("objid", pa.string()), ("program", pa.string()), ("file_path", pa.string()),
    ("ra", pa.float64()), ("dec", pa.float64()), ("n_points", pa.int64()),
    *[(field, pa.list_(pa.string() if field in ("filters", "statuses") else pa.float64()))
      for field in LIGHTCURVE_COLUMNS],
    ("brightest_index", pa.int64()), ("brightest_mjd", pa.float64()),
    ("brightest_filter", pa.string()), ("brightest_mag", pa.float64()),
    *[(name, pa.list_(pa.float32())) for name in CUTOUT_COLUMNS],
    ("read_error", pa.string()),
])


# ---------------------------------------------------------------------------
# One object
# ---------------------------------------------------------------------------

def empty_lightcurve_row(task: dict, read_error: str | None) -> dict:
    return {
        **task, "n_points": 0,
        **{field: [] for field in LIGHTCURVE_COLUMNS},
        "brightest_index": -1, "brightest_mjd": np.nan, "brightest_filter": None, "brightest_mag": np.nan,
        **{name: [] for name in CUTOUT_COLUMNS},
        "read_error": read_error,
    }


def as_text(values) -> np.ndarray:
    """FITS fixed-width strings (bytes or str) -> stripped str."""
    if values.dtype.kind == "S":
        values = np.char.decode(values, "ascii")
    return np.char.strip(np.asarray(values, dtype=str))


def read_lightcurve(task: dict) -> dict:
    """Worker: every epoch of one object's FITS file, plus the brightest point's cutouts."""
    try:
        with fits.open(task["file_path"]) as hdul:
            table = hdul[1].data
            missing = [col for col in [*LIGHTCURVE_COLUMNS.values(), *CUTOUT_COLUMNS.values()]
                       if col not in table.columns.names]
            if missing:
                return empty_lightcurve_row(task, f"missing columns {missing}")

            lightcurve = {}
            for field, col in LIGHTCURVE_COLUMNS.items():
                lightcurve[field] = as_text(table[col]) if field in TEXT_FIELDS else np.asarray(table[col], dtype=float)

            time_order = np.argsort(lightcurve["mjds"], kind="stable")  # NaN MJDs sort last
            lightcurve = {field: values[time_order] for field, values in lightcurve.items()}

            row = empty_lightcurve_row(task, None)
            row["n_points"] = len(time_order)
            row.update({field: values.tolist() for field, values in lightcurve.items()})

            mags = lightcurve["mags"]
            if np.isfinite(mags).any():
                brightest = int(np.nanargmin(mags))
                row["brightest_index"] = brightest
                row["brightest_mjd"] = float(lightcurve["mjds"][brightest])
                row["brightest_filter"] = str(lightcurve["filters"][brightest])
                row["brightest_mag"] = float(mags[brightest])
                row_in_file = int(time_order[brightest])
                for name, col in CUTOUT_COLUMNS.items():
                    # a numpy float32 array (not a list) so Parquet stores float32, as in the FITS file
                    row[name] = np.array(table[col][row_in_file], dtype="f4").ravel()
            return row
    except Exception as e:
        return empty_lightcurve_row(task, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Inputs and outputs
# ---------------------------------------------------------------------------

def load_tasks(master: pathlib.Path, objid_prefix: str) -> list:
    """[{objid, program, file_path, ra, dec}] for every master object whose objid starts with objid_prefix."""
    if not master.exists():
        raise FileNotFoundError(f"{master} not found -- run create_masterdb.py first")
    master_df = pd.read_parquet(master, columns=["objid", "program", "file_path", "ra", "dec"])
    logger.info(f"Loaded {len(master_df)} objects from {master}")
    if objid_prefix:
        master_df = master_df[master_df["objid"].str.startswith(objid_prefix)]
        logger.info(f"Kept {len(master_df)} objects whose objid starts with {objid_prefix!r}")
    return master_df.to_dict("records")


def task_key(row: dict) -> tuple:
    return row["program"], row["objid"]


def load_done_keys(parts_dir: pathlib.Path) -> set:
    """(program, objid) of every object already saved in a part."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for part in parts_dir.glob("chunk_*.parquet"):
        saved = pd.read_parquet(part, columns=["program", "objid"])
        done.update(zip(saved["program"], saved["objid"]))
    return done


def write_part(df: pd.DataFrame, path: pathlib.Path) -> None:
    df = df.sort_values(["program", "objid"], ignore_index=True)
    tmp_path = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pandas(df[SCHEMA.names], schema=SCHEMA, preserve_index=False), tmp_path)
    os.replace(tmp_path, path)  # a part is either complete or absent


def finish(parts_dir: pathlib.Path, out_file: pathlib.Path) -> None:
    """Stream the parts into the output file one at a time (they don't all fit in memory), then remove them."""
    parts = sorted(parts_dir.glob("chunk_*.parquet"))
    n_objects = n_points = n_cutouts = n_errors = 0
    statuses = {}
    tmp_file = out_file.with_name(out_file.name + ".tmp")
    with pq.ParquetWriter(tmp_file, SCHEMA) as writer:
        for part in tqdm(parts, desc="Merging parts", unit="part"):
            table = pq.read_table(part, schema=SCHEMA)
            writer.write_table(table)
            n_objects += table.num_rows
            n_points += pc.sum(table["n_points"]).as_py() or 0
            n_cutouts += pc.sum(pc.greater_equal(table["brightest_index"], 0)).as_py() or 0
            n_errors += table.num_rows - table["read_error"].null_count
            for status, count in pd.Series(pc.list_flatten(table["statuses"]).to_pylist()).value_counts().items():
                statuses[status] = statuses.get(status, 0) + int(count)
            del table
    os.replace(tmp_file, out_file)
    logger.info(f"Wrote {n_objects} objects, {n_points} points to {out_file} "
                f"({out_file.stat().st_size / 1e6:.0f} MB); {n_cutouts} with cutouts, {n_errors} unreadable")
    logger.info(f"Points per STATUS_FPHOT: {statuses}")
    shutil.rmtree(parts_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build every master-file object's DECam light curve, keeping every point.")
    parser.add_argument("--master", type=pathlib.Path, default=MASTER_FILE)
    parser.add_argument("--out", type=pathlib.Path, default=LIGHTCURVES_FILE)
    parser.add_argument("--objid-prefix", default=DEFAULT_OBJID_PREFIX,
                        help="Only objects whose objid starts with this, e.g. T (default: every object).")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    args = parser.parse_args()

    tasks = load_tasks(args.master, args.objid_prefix)
    parts_dir = args.out.parent / (args.out.stem + "_parts")
    done_keys = load_done_keys(parts_dir)
    todo = [task for task in tasks if task_key(task) not in done_keys]
    chunks = [todo[i:i + CHUNK_SIZE] for i in range(0, len(todo), CHUNK_SIZE)]
    logger.info(f"{len(done_keys)} objects already done in {parts_dir}; {len(todo)} to read "
                f"in {len(chunks)} chunks of {CHUNK_SIZE} with {args.workers} workers")

    first_chunk_number = len(list(parts_dir.glob("chunk_*.parquet")))
    n_errors = 0
    with Pool(args.workers) as pool:
        for chunk_number, chunk in enumerate(chunks, start=first_chunk_number):
            rows = list(tqdm(pool.imap_unordered(read_lightcurve, chunk, chunksize=16), total=len(chunk),
                             desc=f"Chunk {chunk_number - first_chunk_number + 1}/{len(chunks)}", unit="file"))
            chunk_df = pd.DataFrame(rows)
            write_part(chunk_df, parts_dir / f"chunk_{chunk_number:06d}.parquet")
            chunk_errors = chunk_df[chunk_df["read_error"].notna()]
            n_errors += len(chunk_errors)
            for _, bad in chunk_errors.head(5).iterrows():
                logger.warning(f"Could not read {bad['file_path']}: {bad['read_error']}")
    if n_errors:
        logger.warning(f"{n_errors} files could not be read this run; they are in the output with read_error set")

    finish(parts_dir, args.out)


if __name__ == "__main__":
    main()
