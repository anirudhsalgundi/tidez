"""
Flag fast risers/faders in the master H5 and write their objids to a
candidates file for the web scanner.

Rate is computed on the combined multi-survey (DECam+ZTF+LSST) lightcurve:
detections from all three surveys are pooled *within each band* (so e.g.
DECam-g and ZTF-g share one timeline), sorted by time, and the steepest
consecutive-pair slope |delta_mag| / delta_t (mag/day) anywhere in any
band's timeline is the object's rate. Rates are kept within a single band
because comparing different filters (e.g. g vs r) as if they were the same
brightness conflates color with real variability and produces bogus rates
for same-night multi-band epochs. An object passes the filter if that rate
is at or above --threshold (default 0.1 mag/day).

Usage:
    python -m tidez.find_fast_transients
    python -m tidez.find_fast_transients --h5 data/tidez_master.h5 --threshold 0.1
"""

import argparse
import itertools
import pathlib

import h5py
from tqdm import tqdm

from tidez.logger import get_logger
from tidez.plot_lightcurve import DEFAULT_H5, load_object_rows

logger = get_logger("find_fast_transients")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "fast_transient_candidates.txt"
DEFAULT_THRESHOLD = 0.1  # mag/day

# Same-visit duplicate detections (e.g. two ZTF alerts ~40s apart) have a
# near-zero baseline that turns ordinary noise into a huge bogus rate (one
# real case: 0.19 mag apart in 41s -> "392 mag/day"). Require pairs to span
# at least this long before trusting their slope.
DEFAULT_MIN_DT_DAYS = 0.25  # 6 hours


def max_rate_mag_per_day(rows, min_dt_days=DEFAULT_MIN_DT_DAYS):
    """Steepest consecutive-pair |delta_mag|/delta_t within any single band
    (pooled across surveys, e.g. DECam-g + ZTF-g share one timeline), or
    None if no band has a pair spanning >= min_dt_days.

    Returns (rate, row_a, row_b) for the pair achieving that rate.
    """
    # DECam subtraction quality: only trust STATUS_FPHOT == 'p' ("positive",
    # SNR >~5.8); 'q'/'m' are marginal/non-detections that shouldn't drive a
    # rate. ZTF/LSST rows have no STATUS_FPHOT (status is None) and use
    # their own `detected` flag as-is.
    detected = [
        r for r in rows
        if r["detected"] and (r["survey"] != "DECam" or r["status"] == "p")
    ]

    best = None
    for band, band_rows in itertools.groupby(
        sorted(detected, key=lambda r: (r["band"], r["jd"])), key=lambda r: r["band"]
    ):
        band_rows = sorted(band_rows, key=lambda r: r["jd"])
        for a, b in zip(band_rows, band_rows[1:]):
            dt = b["jd"] - a["jd"]
            if dt < min_dt_days:
                continue
            rate = abs(b["mag"] - a["mag"]) / dt
            if best is None or rate > best[0]:
                best = (rate, a, b)
    return best


def find_fast_transients(h5_path: pathlib.Path, threshold: float, min_dt_days: float = DEFAULT_MIN_DT_DAYS):
    """Return a list of (objid, rate, row_a, row_b) for objects at/above threshold."""
    with h5py.File(h5_path, "r") as h5file:
        objids = sorted(k for k in h5file.keys() if k != "catalog")

        candidates = []
        for objid in tqdm(objids, desc="Scanning for fast transients", unit="obj"):
            rows = load_object_rows(h5file, objid)
            result = max_rate_mag_per_day(rows, min_dt_days=min_dt_days)
            if result is None:
                continue
            rate, row_a, row_b = result
            if rate >= threshold:
                candidates.append((objid, rate, row_a, row_b))

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Flag objects rising/fading >= threshold mag/day on their combined lightcurve."
    )
    parser.add_argument("--h5", type=pathlib.Path, default=DEFAULT_H5,
                         help="Master H5 file to read from.")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT,
                         help="Output text file of qualifying objids, one per line.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                         help="Minimum |mag/day| rate to qualify (default: 0.1).")
    parser.add_argument("--min-dt-days", type=float, default=DEFAULT_MIN_DT_DAYS,
                         help="Minimum time baseline between two epochs before trusting "
                              "their slope (default: 0.25d = 6h; guards against same-visit "
                              "duplicate detections producing bogus huge rates).")
    args = parser.parse_args()

    candidates = find_fast_transients(args.h5, args.threshold, args.min_dt_days)
    candidates.sort(key=lambda c: c[1], reverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for objid, rate, row_a, row_b in candidates:
            f.write(objid + "\n")

    logger.info(f"{len(candidates)} objects at/above {args.threshold} mag/day -> {args.output}")
    for objid, rate, row_a, row_b in candidates[:20]:
        logger.info(
            f"  {objid}: {rate:.3f} mag/day "
            f"({row_a['band']}/{row_a['survey']} -> {row_b['band']}/{row_b['survey']}, "
            f"dt={row_b['jd'] - row_a['jd']:.2f}d)"
        )


if __name__ == "__main__":
    main()
