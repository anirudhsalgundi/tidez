"""
A three-stage filter aimed specifically at orphan afterglows, TDEs, GRB
afterglows, FBOTs, and other fast extragalactic transients -- built on top
of transient_hunt.py's quality gates, with two extra rejection stages meant
to cut the AGN contamination that otherwise dominates the sample:

  Stage 1 -- Milliquas veto: reject any object with a Milliquas catalog
  match (run crossmatch_milliquas.py first; objects that haven't been
  crossmatched yet are passed through with a warning rather than silently
  rejected or accepted).

  Stage 2 -- Stochastic-variability veto: reject objects whose lightcurve
  looks like ongoing AGN-style variability rather than a single transient
  event, using two signals computed on the pooled, good-quality detections
  in the same band as the best rate pair (see below):
    - reversal fraction: how often consecutive delta-mag flips sign. A
      single rise-or-fade event is close to monotonic (low reversal
      fraction); stochastic AGN variability flips sign often.
    - persistent-detection fraction: what fraction of epochs *outside* a
      window around the fast-rate event are still detections rather than
      upper limits. A real one-off transient should be undetected most of
      the time outside its flare; an AGN is typically detected most of the
      time throughout its whole baseline.
  Either signal above its threshold rejects the object. Both checks are
  skipped (not rejected) when there isn't enough data to compute them
  reliably, since "not enough information" isn't evidence of AGN-hood.

  Stage 3 -- Asymmetric rate cut: unlike transient_hunt.py's single
  |rate| >= threshold, real fast transients typically rise faster than
  they fade (as do FBOTs/afterglows/TDEs), so rise and fade get separate,
  independently-tunable thresholds:
    - rising (getting brighter) >= --rise-threshold mag/day (default 0.5)
    - fading (getting fainter)  >= --fade-threshold mag/day (default 0.2)
  All of transient_hunt.py's per-pair quality cuts (DECam 'p'-only, SNR
  floor, minimum baseline, minimum amplitude, significance, anti-blip
  check) still apply pair-by-pair before a rate is trusted.

Run crossmatch_milliquas.py (and ideally crossmatch_boom.py, for the
crossmatched-output split) before this.

Usage:
    python -m tidez.afterglow_hunt
    python -m tidez.afterglow_hunt --h5 data/tidez_master.h5 --rise-threshold 0.5 --fade-threshold 0.2
"""

import argparse
import itertools
import math
import pathlib

import h5py
from tqdm import tqdm

from tidez.logger import get_logger
from tidez.plot_lightcurve import DEFAULT_H5, load_object_rows

logger = get_logger("afterglow_hunt")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "afterglow_candidates.txt"
DEFAULT_CROSSMATCH_OUTPUT = DATA_DIR / "afterglow_candidates_crossmatched.txt"

# Stage 3 (asymmetric rate cut)
DEFAULT_RISE_THRESHOLD = 0.5  # mag/day, brightening
DEFAULT_FADE_THRESHOLD = 0.2  # mag/day, dimming

# Per-pair quality cuts, same rationale/defaults as transient_hunt.py
DEFAULT_MIN_DT_DAYS = 0.25
DEFAULT_MAX_DT_DAYS = None
DEFAULT_MIN_DELTA_MAG = 0.3
DEFAULT_SIGMA_THRESH = 4.0
DEFAULT_MIN_SNR = 5.0
DEFAULT_BLIP_WINDOW_DAYS = 3.0
DEFAULT_BLIP_REVERT_TOL = 0.15
DEFAULT_MIN_DETECTIONS = 3

# Stage 2 (stochastic-variability veto)
DEFAULT_MAX_REVERSAL_FRAC = 0.4
DEFAULT_MAX_PERSISTENT_FRAC = 0.5
DEFAULT_EVENT_WINDOW_DAYS = 7.0
DEFAULT_MIN_OUTSIDE_POINTS = 4


def has_crossmatch(h5file, objid):
    """True if the object has a ZTF or LSST/Rubin crossmatch lightcurve."""
    cm = h5file[objid].get("crossmatch")
    if cm is None:
        return False
    return "ztf" in cm or "lsst" in cm


_warned_no_milliquas = False


def milliquas_matched(h5file, objid):
    """True/False if crossmatch_milliquas.py has run for this object, else
    None (unknown -- treated as "not matched" by the caller, with a
    one-time warning) if it hasn't been crossmatched yet."""
    global _warned_no_milliquas
    group = h5file[objid].get("milliquas")
    if group is None:
        if not _warned_no_milliquas:
            logger.warning(
                "No Milliquas crossmatch found (run crossmatch_milliquas.py first) -- "
                "objects without a crossmatch are passed through this veto, not rejected."
            )
            _warned_no_milliquas = True
        return None
    return bool(group.attrs["matched"])


def _is_blip(band_rows, a, b, window_days, revert_tol):
    idx_b = band_rows.index(b)
    if idx_b + 1 >= len(band_rows):
        return False
    c = band_rows[idx_b + 1]
    if (c["jd"] - b["jd"]) > window_days:
        return False
    return abs(c["mag"] - a["mag"]) < revert_tol


def _reversal_fraction(band_rows):
    """Fraction of consecutive delta-mag sign flips in a time-sorted band
    timeline. None if too few points to judge (<4, i.e. <3 deltas)."""
    if len(band_rows) < 4:
        return None
    deltas = [band_rows[i + 1]["mag"] - band_rows[i]["mag"] for i in range(len(band_rows) - 1)]
    reversals = sum(
        1 for i in range(len(deltas) - 1)
        if deltas[i] != 0 and deltas[i + 1] != 0 and (deltas[i] > 0) != (deltas[i + 1] > 0)
    )
    return reversals / (len(deltas) - 1)


def _persistent_detection_fraction(all_rows, event_jd_lo, event_jd_hi, min_outside_points):
    """Fraction of epochs outside [event_jd_lo, event_jd_hi] (across all
    bands/surveys) that are detections rather than upper limits. None if
    fewer than min_outside_points epochs fall outside the window."""
    outside = [r for r in all_rows if not (event_jd_lo <= r["jd"] <= event_jd_hi)]
    if len(outside) < min_outside_points:
        return None
    return sum(1 for r in outside if r["detected"]) / len(outside)


def is_stochastic(all_rows, band_rows, event_a, event_b, max_reversal_frac, max_persistent_frac,
                   event_window_days, min_outside_points):
    """True if either stochastic-variability signal fires for this object's
    best-rate band/event pair."""
    reversal_frac = _reversal_fraction(band_rows)
    if reversal_frac is not None and reversal_frac >= max_reversal_frac:
        return True

    event_jd_lo = event_a["jd"] - event_window_days
    event_jd_hi = event_b["jd"] + event_window_days
    persistent_frac = _persistent_detection_fraction(all_rows, event_jd_lo, event_jd_hi, min_outside_points)
    if persistent_frac is not None and persistent_frac >= max_persistent_frac:
        return True

    return False


def find_afterglow_candidates(
    h5_path: pathlib.Path,
    rise_threshold: float = DEFAULT_RISE_THRESHOLD,
    fade_threshold: float = DEFAULT_FADE_THRESHOLD,
    min_dt_days: float = DEFAULT_MIN_DT_DAYS,
    max_dt_days: float | None = DEFAULT_MAX_DT_DAYS,
    min_delta_mag: float = DEFAULT_MIN_DELTA_MAG,
    sigma_thresh: float = DEFAULT_SIGMA_THRESH,
    min_snr: float = DEFAULT_MIN_SNR,
    blip_window_days: float = DEFAULT_BLIP_WINDOW_DAYS,
    blip_revert_tol: float = DEFAULT_BLIP_REVERT_TOL,
    min_detections: int = DEFAULT_MIN_DETECTIONS,
    max_reversal_frac: float = DEFAULT_MAX_REVERSAL_FRAC,
    max_persistent_frac: float = DEFAULT_MAX_PERSISTENT_FRAC,
    event_window_days: float = DEFAULT_EVENT_WINDOW_DAYS,
    min_outside_points: int = DEFAULT_MIN_OUTSIDE_POINTS,
):
    """Return (candidates, stage_counts). candidates is a list of
    (objid, rate, direction, score, significance, row_a, row_b, crossmatched)
    sorted by score descending. stage_counts is a dict of how many objects
    were dropped at each stage, for the summary log."""
    stage_counts = {"total": 0, "milliquas_rejected": 0, "stochastic_rejected": 0, "no_rate_pair": 0, "passed": 0}

    with h5py.File(h5_path, "r") as h5file:
        objids = sorted(k for k in h5file.keys() if k != "catalog")
        stage_counts["total"] = len(objids)

        candidates = []
        for objid in tqdm(objids, desc="Hunting for afterglow-like transients", unit="obj"):
            # Stage 1: Milliquas veto
            if milliquas_matched(h5file, objid):
                stage_counts["milliquas_rejected"] += 1
                continue

            rows = load_object_rows(h5file, objid, good_only=True)
            detected = [r for r in rows if r["detected"]]
            if len(detected) < min_detections:
                stage_counts["no_rate_pair"] += 1
                continue

            # Stage 3 (quality cuts + asymmetric rate): find the best-scoring
            # pair per object across all bands, same structure as
            # transient_hunt.py but tracking rise/fade direction separately.
            best = None  # (score, rate, direction, significance, band_rows, a, b)
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

                    signed_delta = b["mag"] - a["mag"]
                    delta_mag = abs(signed_delta)
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
                    direction = "rise" if signed_delta < 0 else "fade"
                    threshold = rise_threshold if direction == "rise" else fade_threshold
                    if rate < threshold:
                        continue

                    if _is_blip(band_rows, a, b, blip_window_days, blip_revert_tol):
                        continue

                    score = rate * significance
                    if best is None or score > best[0]:
                        best = (score, rate, direction, significance, band_rows, a, b)

            if best is None:
                stage_counts["no_rate_pair"] += 1
                continue

            score, rate, direction, significance, band_rows, row_a, row_b = best

            # Stage 2: stochastic-variability veto, evaluated against the
            # band/event that produced the winning rate.
            if is_stochastic(rows, band_rows, row_a, row_b, max_reversal_frac, max_persistent_frac,
                              event_window_days, min_outside_points):
                stage_counts["stochastic_rejected"] += 1
                continue

            stage_counts["passed"] += 1
            candidates.append((objid, rate, direction, score, significance, row_a, row_b, has_crossmatch(h5file, objid)))

    candidates.sort(key=lambda c: c[3], reverse=True)
    return candidates, stage_counts


def main():
    parser = argparse.ArgumentParser(
        description="Hunt for orphan afterglows/TDEs/GRB afterglows/FBOTs: Milliquas veto -> "
                     "stochastic-variability veto -> asymmetric rise/fade rate cut (see module docstring)."
    )
    parser.add_argument("--h5", type=pathlib.Path, default=DEFAULT_H5,
                         help="Master H5 file to read from.")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT,
                         help="Output text file of all qualifying objids, one per line.")
    parser.add_argument("--crossmatch-output", type=pathlib.Path, default=DEFAULT_CROSSMATCH_OUTPUT,
                         help="Output text file of qualifying objids that also have a "
                              "ZTF/Rubin crossmatch, one per line.")
    parser.add_argument("--rise-threshold", type=float, default=DEFAULT_RISE_THRESHOLD,
                         help="Minimum brightening rate (mag/day) to qualify (default: 0.5).")
    parser.add_argument("--fade-threshold", type=float, default=DEFAULT_FADE_THRESHOLD,
                         help="Minimum fading rate (mag/day) to qualify (default: 0.2).")
    parser.add_argument("--min-dt-days", type=float, default=DEFAULT_MIN_DT_DAYS,
                         help="Minimum time baseline between two epochs (default: 0.25d = 6h).")
    parser.add_argument("--max-dt-days", type=float, default=DEFAULT_MAX_DT_DAYS,
                         help="Maximum time baseline between two epochs (default: no cap).")
    parser.add_argument("--min-delta-mag", type=float, default=DEFAULT_MIN_DELTA_MAG,
                         help="Minimum |delta mag| for a pair to qualify (default: 0.3).")
    parser.add_argument("--sigma-thresh", type=float, default=DEFAULT_SIGMA_THRESH,
                         help="Minimum delta-mag significance in combined-error sigma (default: 4.0).")
    parser.add_argument("--min-snr", type=float, default=DEFAULT_MIN_SNR,
                         help="Minimum DECam SNR_FPHOT for either endpoint of a pair (default: 5.0).")
    parser.add_argument("--blip-window-days", type=float, default=DEFAULT_BLIP_WINDOW_DAYS,
                         help="Reject a pair if the next point in-band reverts near the "
                              "pre-pair baseline within this many days (default: 3.0).")
    parser.add_argument("--blip-revert-tol", type=float, default=DEFAULT_BLIP_REVERT_TOL,
                         help="Mag tolerance for the anti-blip revert-to-baseline check (default: 0.15).")
    parser.add_argument("--min-detections", type=int, default=DEFAULT_MIN_DETECTIONS,
                         help="Minimum total good detections an object must have (default: 3).")
    parser.add_argument("--max-reversal-frac", type=float, default=DEFAULT_MAX_REVERSAL_FRAC,
                         help="Reject if the fraction of sign-flipping consecutive deltas in the "
                              "event band is >= this (default: 0.4).")
    parser.add_argument("--max-persistent-frac", type=float, default=DEFAULT_MAX_PERSISTENT_FRAC,
                         help="Reject if the fraction of detections (vs. upper limits) outside the "
                              "event window is >= this (default: 0.5).")
    parser.add_argument("--event-window-days", type=float, default=DEFAULT_EVENT_WINDOW_DAYS,
                         help="Days padded around the winning event pair, excluded from the "
                              "persistent-detection-fraction calculation (default: 7.0).")
    parser.add_argument("--min-outside-points", type=int, default=DEFAULT_MIN_OUTSIDE_POINTS,
                         help="Minimum epochs required outside the event window before the "
                              "persistent-detection check is trusted (default: 4).")
    args = parser.parse_args()

    candidates, stage_counts = find_afterglow_candidates(
        args.h5,
        rise_threshold=args.rise_threshold,
        fade_threshold=args.fade_threshold,
        min_dt_days=args.min_dt_days,
        max_dt_days=args.max_dt_days,
        min_delta_mag=args.min_delta_mag,
        sigma_thresh=args.sigma_thresh,
        min_snr=args.min_snr,
        blip_window_days=args.blip_window_days,
        blip_revert_tol=args.blip_revert_tol,
        min_detections=args.min_detections,
        max_reversal_frac=args.max_reversal_frac,
        max_persistent_frac=args.max_persistent_frac,
        event_window_days=args.event_window_days,
        min_outside_points=args.min_outside_points,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for objid, rate, direction, score, significance, row_a, row_b, crossmatched in candidates:
            f.write(objid + "\n")

    crossmatched_candidates = [c for c in candidates if c[7]]
    args.crossmatch_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.crossmatch_output, "w") as f:
        for objid, rate, direction, score, significance, row_a, row_b, crossmatched in crossmatched_candidates:
            f.write(objid + "\n")

    logger.info(
        f"{stage_counts['total']} objects scanned -> "
        f"{stage_counts['milliquas_rejected']} rejected (Milliquas match), "
        f"{stage_counts['stochastic_rejected']} rejected (stochastic variability), "
        f"{stage_counts['no_rate_pair']} had no qualifying rate pair, "
        f"{len(candidates)} passed -> {args.output}"
    )
    logger.info(
        f"{len(crossmatched_candidates)} of those have a ZTF/Rubin crossmatch -> {args.crossmatch_output}"
    )
    for objid, rate, direction, score, significance, row_a, row_b, crossmatched in candidates[:20]:
        logger.info(
            f"  {objid}: {direction} {rate:.3f} mag/day, {significance:.1f}sigma "
            f"({row_a['band']}/{row_a['survey']} -> {row_b['band']}/{row_b['survey']}, "
            f"dt={row_b['jd'] - row_a['jd']:.2f}d)"
            f"{'  [XM]' if crossmatched else ''}"
        )


if __name__ == "__main__":
    main()
