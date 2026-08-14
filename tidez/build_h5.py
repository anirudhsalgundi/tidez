"""
Build/append the master H5 file from candidate FITS files.

One group per object ID:
  /<objid>/                 attrs: objid, ra, dec
  /<objid>/lightcurve        compound dataset, one row per epoch (all
                              STATUS_FPHOT values kept, not just 'p')
  /<objid>/cutouts/template   (n_epochs, 121, 121) float32, gzip
  /<objid>/cutouts/science    (n_epochs, 121, 121) float32, gzip
  /<objid>/cutouts/diff       (n_epochs, 121, 121) float32, gzip

Plus a top-level extendable catalog table for fast objid/cone-search lookup:
  /catalog   compound dataset: objid, ra, dec, n_epochs

Resumable by design: an object's group is only created after its FITS file
is fully read and its arrays are built in memory, so a group is either
absent or complete. The file is flushed to disk every FLUSH_EVERY objects,
so a crash loses at most the objects processed since the last flush, not
anything already on disk. Re-running with the same --output simply skips
any objid whose group already exists.
"""

import argparse
import pathlib
import traceback

import h5py
import numpy as np
from astropy.io import fits
from tqdm import tqdm

from tidez.logger import get_logger
from tidez.get_data import parse_fits_filename

logger = get_logger("build_h5")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_INPUT_DIR = DATA_DIR / "candidates"
DEFAULT_OUTPUT = DATA_DIR / "tidez_master.h5"
FLUSH_EVERY = 50
STAMP_SIDE = 121  # sqrt(14641)

LIGHTCURVE_COLUMNS = [
    "MJD_OBS",
    "FILTER",
    "MAG_FPHOT",
    "MAGERR_FPHOT",
    "SNR_FPHOT",
    "STATUS_FPHOT",
    "LIM_MAG3",
    "LIM_MAG5",
]

CATALOG_DTYPE = np.dtype(
    [("objid", "S32"), ("ra", "f8"), ("dec", "f8"), ("n_epochs", "i4")]
)

LIGHTCURVE_DTYPE = np.dtype(
    [
        ("MJD_OBS", "f8"),
        ("FILTER", "S3"),
        ("MAG_FPHOT", "f4"),
        ("MAGERR_FPHOT", "f4"),
        ("SNR_FPHOT", "f4"),
        ("STATUS_FPHOT", "S1"),
        ("LIM_MAG3", "f4"),
        ("LIM_MAG5", "f4"),
    ]
)


def read_object_fits(fits_path):
    """Read one candidate FITS file, returning (ra, dec, lightcurve, cutouts).

    RA/Dec come from the RA_OBJ/DEC_OBJ keywords in the table extension's
    header (not parsed from the objid string, which is lossy/ambiguous).
    """
    with fits.open(fits_path) as hdul:
        header = hdul[1].header
        ra = float(header["RA_OBJ"])
        dec = float(header["DEC_OBJ"])

        table = hdul[1].data
        n_epochs = len(table)

        lightcurve = np.empty(n_epochs, dtype=LIGHTCURVE_DTYPE)
        for col in LIGHTCURVE_COLUMNS:
            lightcurve[col] = table[col]

        cutouts = {}
        for key, col in (
            ("template", "PixA_THUMB_TEMP"),
            ("science", "PixA_THUMB_SCI"),
            ("diff", "PixA_THUMB_DIFF"),
        ):
            stamps = np.asarray(table[col], dtype="f4")
            cutouts[key] = stamps.reshape(n_epochs, STAMP_SIDE, STAMP_SIDE)

    return ra, dec, lightcurve, cutouts


def ensure_catalog_dataset(h5file: h5py.File):
    if "catalog" not in h5file:
        h5file.create_dataset(
            "catalog",
            shape=(0,),
            maxshape=(None,),
            dtype=CATALOG_DTYPE,
            chunks=True,
        )
    return h5file["catalog"]


def append_to_catalog(h5file: h5py.File, objid: str, ra: float, dec: float, n_epochs: int):
    catalog = h5file["catalog"]
    idx = catalog.shape[0]
    catalog.resize((idx + 1,))
    catalog[idx] = (objid.encode("ascii"), ra, dec, n_epochs)


def write_object_group(h5file: h5py.File, objid: str, ra: float, dec: float, lightcurve, cutouts):
    group = h5file.create_group(objid)
    group.attrs["objid"] = objid
    group.attrs["ra"] = ra
    group.attrs["dec"] = dec

    group.create_dataset(
        "lightcurve", data=lightcurve, compression="gzip", compression_opts=4
    )

    cutout_group = group.create_group("cutouts")
    for key, stamps in cutouts.items():
        cutout_group.create_dataset(
            key, data=stamps, compression="gzip", compression_opts=4, shuffle=True
        )


def gather_input_files(input_dir: pathlib.Path, input_list: pathlib.Path):
    if input_list is not None:
        with open(input_list) as f:
            paths = [pathlib.Path(line.strip()) for line in f if line.strip()]
        logger.info(f"Read {len(paths)} candidate file paths from {input_list}")
        return paths

    paths = sorted(input_dir.glob("*.fits"))
    logger.info(f"Found {len(paths)} candidate FITS files under {input_dir}")
    return paths


def build(input_dir: pathlib.Path, input_list: pathlib.Path, output: pathlib.Path, limit: int = None):
    files = gather_input_files(input_dir, input_list)
    if limit is not None:
        files = files[:limit]

    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Opening output H5 file (append mode): {output}")

    n_processed = 0
    n_skipped = 0
    n_failed = 0

    with h5py.File(output, "a") as h5file:
        ensure_catalog_dataset(h5file)
        already_present = set(h5file.keys()) - {"catalog"}
        logger.info(f"{len(already_present)} objects already present in {output}; will be skipped.")

        pbar = tqdm(files, desc="Building master H5", unit="file")
        for i, fits_path in enumerate(pbar, start=1):
            pbar.set_postfix_str(f"{i}/{len(files)}")

            parsed = parse_fits_filename(fits_path)
            if parsed is None:
                n_failed += 1
                continue
            n_detections, objid = parsed

            if objid in already_present:
                n_skipped += 1
                logger.debug(f"Skipping already-ingested object: {objid}")
                continue

            try:
                ra, dec, lightcurve, cutouts = read_object_fits(fits_path)

                write_object_group(h5file, objid, ra, dec, lightcurve, cutouts)
                append_to_catalog(h5file, objid, ra, dec, len(lightcurve))
                already_present.add(objid)
                n_processed += 1

            except Exception:
                n_failed += 1
                logger.error(f"Failed to process {fits_path} (objid={objid}):\n{traceback.format_exc()}")
                continue

            if n_processed % FLUSH_EVERY == 0:
                h5file.flush()
                logger.info(
                    f"Flushed at file {i}/{len(files)} "
                    f"(processed={n_processed}, skipped={n_skipped}, failed={n_failed})"
                )

        h5file.flush()

    logger.info(
        f"Done. processed={n_processed}, skipped={n_skipped}, failed={n_failed}, "
        f"total_input_files={len(files)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build/append the tidez master H5 file.")
    parser.add_argument("--input-dir", type=pathlib.Path, default=DEFAULT_INPUT_DIR,
                         help="Directory of candidate FITS files to glob (local testing mode).")
    parser.add_argument("--input-list", type=pathlib.Path, default=None,
                         help="Text file of candidate FITS paths, one per line "
                              "(e.g. max_detections_candidate_fits_files.txt on the remote).")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT,
                         help="Output H5 file path (created if missing, appended to if present).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N input files (for testing).")
    args = parser.parse_args()

    build(args.input_dir, args.input_list, args.output, args.limit)


if __name__ == "__main__":
    main()
