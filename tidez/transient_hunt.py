"""
A stricter, multi-cut successor to find_fast_transients.py aimed at
rejecting subtraction/photometry artifacts that pass a bare rate
threshold, keeping only fast risers/faders that also look astrophysically
real.

Rate is computed the same way as find_fast_transients.py: pool detections
within each band (DECam+ZTF+LSST share a band timeline), sort by time, and
look at the steepest consecutive-pair |delta_mag| / delta_t. But a pair is
only accepted as a genuine candidate if it additionally survives all of:

  1. DECam quality gate: only STATUS_FPHOT == 'p' ("good", SNR>~5.8) DECam
     epochs are used to build rates. 'q' (marginal) and 'm' (non-detection)
     rows are dropped outright rather than trusted, unlike the plain
     find_fast_transients filter which also allows 'q'.
  2. SNR floor: any DECam endpoint of the pair must also clear --min-snr
     (default 5.0) even though 'p' already implies ~5.8 -- this is a second,
     independently-tunable knob.
  3. Minimum baseline: pairs must span >= --min-dt-days (default 6h),
     guarding against same-visit duplicate detections with near-zero
     baselines (one real case: 0.19 mag apart in 41s -> "392 mag/day").
  4. Minimum amplitude: |delta_mag| must be >= --min-delta-mag (default
     0.3 mag). A "fast" rate built from a tiny magnitude wobble over a
     tiny time baseline is usually just noise riding a short dt, not a
     real fast transient.
  5. Statistical significance: |delta_mag| / sqrt(magerr_a^2 + magerr_b^2)
     must be >= --sigma-thresh (default 4.0), i.e. the brightness change
     must be resolved well above the combined photometric error bars.
  6. Anti-blip check: reject a pair if the very next point in that band's
     timeline reverts back close to the pre-pair baseline within a short
     window (--blip-window-days, default 3d) -- a real fast transient
     doesn't discretely snap back to baseline a few points later; a
     transient subtraction artifact or bad reference often does.
  7. Support requirement: the object must have >= --min-detections
     (default 3) total good detections across all bands/surveys combined,
     so a rate isn't accepted from an object with literally nothing else
     in its lightcurve to sanity-check it against.

Candidates that pass all cuts are ranked by score = rate * significance
(so a fast AND well-resolved change ranks above a fast-but-marginal one).

Usage:
    python -m tidez.transient_hunt
    python -m tidez.transient_hunt --h5 data/tidez_master.h5 --threshold 0.1
"""

import argparse
import itertools
import math
import pathlib

import h5py
from tqdm import tqdm

from tidez.logger import get_logger
from tidez.plot_lightcurve import DEFAULT_H5, load_object_rows

logger = get_logger("transient_hunt")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "transient_hunt_candidates.txt"
DEFAULT_CROSSMATCH_OUTPUT = DATA_DIR / "transient_hunt_candidates_crossmatched.txt"

DEFAULT_THRESHOLD = 0.1        # mag/day
DEFAULT_MIN_DT_DAYS = 0.25     # 6 hours
DEFAULT_MAX_DT_DAYS = None     # no cap by default
DEFAULT_MIN_DELTA_MAG = 0.3    # mag
DEFAULT_SIGMA_THRESH = 4.0     # combined-error significance
DEFAULT_MIN_SNR = 5.0          # DECam SNR_FPHOT floor
DEFAULT_BLIP_WINDOW_DAYS = 3.0
DEFAULT_BLIP_REVERT_TOL = 0.15  # mag, "back near baseline" tolerance
DEFAULT_MIN_DETECTIONS = 3


def has_crossmatch(h5file, objid):
    """True if the object has a ZTF or LSST/Rubin crossmatch lightcurve."""
    cm = h5file[objid].get("crossmatch")
    if cm is None:
        return False
    return "ztf" in cm or "lsst" in cm


def _is_blip(band_rows, a, b, window_days, revert_tol):
    """True if the point right after b in this band's timeline snaps back
    close to a's baseline within `window_days` -- a signature of a
    subtraction/photometry glitch rather than a real fast transient."""
    idx_b = band_rows.index(b)
    if idx_b + 1 >= len(band_rows):
        return False
    c = band_rows[idx_b + 1]
    if (c["jd"] - b["jd"]) > window_days:
        return False
    return abs(c["mag"] - a["mag"]) < revert_tol


def find_transients(
    h5_path: pathlib.Path,
    threshold: float,
    min_dt_days: float = DEFAULT_MIN_DT_DAYS,
    max_dt_days: float | None = DEFAULT_MAX_DT_DAYS,
    min_delta_mag: float = DEFAULT_MIN_DELTA_MAG,
    sigma_thresh: float = DEFAULT_SIGMA_THRESH,
    min_snr: float = DEFAULT_MIN_SNR,
    blip_window_days: float = DEFAULT_BLIP_WINDOW_DAYS,
    blip_revert_tol: float = DEFAULT_BLIP_REVERT_TOL,
    min_detections: int = DEFAULT_MIN_DETECTIONS,
):
    """Return [(objid, rate, score, significance, row_a, row_b, crossmatched)]
    for objects with a pair passing every quality cut, best-scoring pair
    per object, sorted by score descending."""
    with h5py.File(h5_path, "r") as h5file:
        objids = sorted(k for k in h5file.keys() if k != "catalog")

        candidates = []
        for objid in tqdm(objids, desc="Hunting for real fast transients", unit="obj"):
            # good_only=True: only STATUS_FPHOT == 'p' DECam epochs feed the
            # rate search; ZTF/LSST rows come through as-is (their own
            # `detected` flag is trusted, same as find_fast_transients.py).
            rows = load_object_rows(h5file, objid, good_only=True)
            detected = [r for r in rows if r["detected"]]
            if len(detected) < min_detections:
                continue

            best = None  # (score, rate, significance, row_a, row_b)
            for band, group in itertools.groupby(
                sorted(detected, key=lambda r: (r["band"], r["jd"])), key=lambda r: r["band"]
            ):
                band_rows = sorted(group, key=lambda r: r["jd"])
                for a, b in zip(band_rows, band_rows[1:]):
                    dt = b["jd"] - a["jd"]
                    if dt < min_dt_days:
                        continue
                    if max_dt_days is not None and dt > max_dt_days:
                        continue

                    delta_mag = abs(b["mag"] - a["mag"])
                    if delta_mag < min_delta_mag:
                        continue

                    combined_err = math.sqrt(a["magerr"] ** 2 + b["magerr"] ** 2)
                    significance = delta_mag / combined_err if combined_err > 0 else math.inf
                    if significance < sigma_thresh:
                        continue

                    if a["survey"] == "DECam" and a["snr"] is not None and a["snr"] < min_snr:
                        continue
                    if b["survey"] == "DECam" and b["snr"] is not None and b["snr"] < min_snr:
                        continue

                    rate = delta_mag / dt
                    if rate < threshold:
                        continue

                    if _is_blip(band_rows, a, b, blip_window_days, blip_revert_tol):
                        continue

                    score = rate * significance
                    if best is None or score > best[0]:
                        best = (score, rate, significance, a, b)

            if best is not None:
                score, rate, significance, row_a, row_b = best
                candidates.append((objid, rate, score, significance, row_a, row_b, has_crossmatch(h5file, objid)))

    candidates.sort(key=lambda c: c[2], reverse=True)
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Hunt for fast risers/faders that also survive quality cuts meant "
                     "to reject subtraction/photometry artifacts (see module docstring)."
    )
    parser.add_argument("--h5", type=pathlib.Path, default=DEFAULT_H5,
                         help="Master H5 file to read from.")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT,
                         help="Output text file of all qualifying objids, one per line.")
    parser.add_argument("--crossmatch-output", type=pathlib.Path, default=DEFAULT_CROSSMATCH_OUTPUT,
                         help="Output text file of qualifying objids that also have a "
                              "ZTF/Rubin crossmatch, one per line.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                         help="Minimum |mag/day| rate to qualify (default: 0.1).")
    parser.add_argument("--min-dt-days", type=float, default=DEFAULT_MIN_DT_DAYS,
                         help="Minimum time baseline between two epochs before trusting "
                              "their slope (default: 0.25d = 6h).")
    parser.add_argument("--max-dt-days", type=float, default=DEFAULT_MAX_DT_DAYS,
                         help="Maximum time baseline between two epochs (default: no cap). "
                              "Set this to exclude slow, long-baseline trends from counting "
                              "as 'fast'.")
    parser.add_argument("--min-delta-mag", type=float, default=DEFAULT_MIN_DELTA_MAG,
                         help="Minimum |delta mag| for a pair to qualify (default: 0.3).")
    parser.add_argument("--sigma-thresh", type=float, default=DEFAULT_SIGMA_THRESH,
                         help="Minimum delta-mag significance in combined-error sigma "
                              "(default: 4.0).")
    parser.add_argument("--min-snr", type=float, default=DEFAULT_MIN_SNR,
                         help="Minimum DECam SNR_FPHOT for either endpoint of a pair "
                              "(default: 5.0).")
    parser.add_argument("--blip-window-days", type=float, default=DEFAULT_BLIP_WINDOW_DAYS,
                         help="Reject a pair if the next point in-band reverts near the "
                              "pre-pair baseline within this many days (default: 3.0).")
    parser.add_argument("--blip-revert-tol", type=float, default=DEFAULT_BLIP_REVERT_TOL,
                         help="Mag tolerance for the anti-blip revert-to-baseline check "
                              "(default: 0.15).")
    parser.add_argument("--min-detections", type=int, default=DEFAULT_MIN_DETECTIONS,
                         help="Minimum total good detections (any band/survey) an object "
                              "must have before its rate is trusted (default: 3).")
    args = parser.parse_args()

    candidates = find_transients(
        args.h5,
        args.threshold,
        min_dt_days=args.min_dt_days,
        max_dt_days=args.max_dt_days,
        min_delta_mag=args.min_delta_mag,
        sigma_thresh=args.sigma_thresh,
        min_snr=args.min_snr,
        blip_window_days=args.blip_window_days,
        blip_revert_tol=args.blip_revert_tol,
        min_detections=args.min_detections,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for objid, rate, score, significance, row_a, row_b, crossmatched in candidates:
            f.write(objid + "\n")

    crossmatched_candidates = [c for c in candidates if c[6]]
    args.crossmatch_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.crossmatch_output, "w") as f:
        for objid, rate, score, significance, row_a, row_b, crossmatched in crossmatched_candidates:
            f.write(objid + "\n")

    logger.info(f"{len(candidates)} objects passed all cuts -> {args.output}")
    logger.info(
        f"{len(crossmatched_candidates)} of those have a ZTF/Rubin crossmatch -> {args.crossmatch_output}"
    )
    for objid, rate, score, significance, row_a, row_b, crossmatched in candidates[:20]:
        logger.info(
            f"  {objid}: {rate:.3f} mag/day, {significance:.1f}sigma "
            f"({row_a['band']}/{row_a['survey']} -> {row_b['band']}/{row_b['survey']}, "
            f"dt={row_b['jd'] - row_a['jd']:.2f}d)"
            f"{'  [XM]' if crossmatched else ''}"
        )


if __name__ == "__main__":
    main()
