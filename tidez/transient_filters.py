"""
Shared filtering logic used by both afterglow_hunt.py (assumes crossmatch
data already exists for every object) and find_real_fast_transients.py
(does the rate cut first, then crossmatches only the survivors). Kept here
so the two don't drift out of sync on what "passes" means.

See afterglow_hunt.py's module docstring for the full rationale behind
each cut -- this module just holds the mechanics.
"""

import itertools
import math

# Per-pair quality cuts (DECam 'p'-only detections, applied by the caller
# via load_object_rows(good_only=True))
DEFAULT_MIN_DT_DAYS = 0.25
DEFAULT_MAX_DT_DAYS = None
DEFAULT_MIN_DELTA_MAG = 0.3
DEFAULT_SIGMA_THRESH = 4.0
DEFAULT_MIN_SNR = 5.0
DEFAULT_BLIP_WINDOW_DAYS = 3.0
DEFAULT_BLIP_REVERT_TOL = 0.15
DEFAULT_MIN_DETECTIONS = 3

# Asymmetric rise/fade rate cut
DEFAULT_RISE_THRESHOLD = 0.5  # mag/day, brightening
DEFAULT_FADE_THRESHOLD = 0.2  # mag/day, dimming

# Stochastic-variability veto
DEFAULT_MAX_REVERSAL_FRAC = 0.4
DEFAULT_MAX_PERSISTENT_FRAC = 0.5
DEFAULT_EVENT_WINDOW_DAYS = 7.0
DEFAULT_MIN_OUTSIDE_POINTS = 4


def has_crossmatch(h5file, objid):
    """True if the object has a ZTF or LSST/Rubin crossmatch lightcurve
    (see crossmatch_boom.py)."""
    cm = h5file[objid].get("crossmatch")
    if cm is None:
        return False
    return "ztf" in cm or "lsst" in cm


_warned_no_milliquas = False


def milliquas_matched(h5file, objid, logger=None):
    """True/False if crossmatch_milliquas.py has run for this object, else
    None (unknown -- treated as "not matched" by the caller, with a
    one-time warning if a logger is given) if it hasn't been crossmatched
    yet."""
    global _warned_no_milliquas
    group = h5file[objid].get("milliquas")
    if group is None:
        if logger is not None and not _warned_no_milliquas:
            logger.warning(
                "No Milliquas crossmatch found for at least one object -- "
                "objects without a crossmatch are passed through this veto, not rejected."
            )
            _warned_no_milliquas = True
        return None
    return bool(group.attrs["matched"])


def find_best_rate_pair(
    rows,
    rise_threshold: float = DEFAULT_RISE_THRESHOLD,
    fade_threshold: float = DEFAULT_FADE_THRESHOLD,
    min_dt_days: float = DEFAULT_MIN_DT_DAYS,
    max_dt_days: float | None = DEFAULT_MAX_DT_DAYS,
    min_delta_mag: float = DEFAULT_MIN_DELTA_MAG,
    sigma_thresh: float = DEFAULT_SIGMA_THRESH,
    min_snr: float = DEFAULT_MIN_SNR,
    blip_window_days: float = DEFAULT_BLIP_WINDOW_DAYS,
    blip_revert_tol: float = DEFAULT_BLIP_REVERT_TOL,
):
    """Steepest quality-cut-passing pair across all bands in `rows`
    (already-detected rows only expected), tracking rise/fade direction
    separately with independent thresholds. Returns
    (score, rate, direction, significance, band_rows, row_a, row_b) for the
    best-scoring pair, or None if nothing qualifies. score = rate *
    significance, used to rank candidates against each other."""
    best = None
    for band, group in itertools.groupby(
        sorted(rows, key=lambda r: (r["band"], r["jd"])), key=lambda r: r["band"]
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

    return best


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


def is_stochastic(
    all_rows, band_rows, event_a, event_b,
    max_reversal_frac: float = DEFAULT_MAX_REVERSAL_FRAC,
    max_persistent_frac: float = DEFAULT_MAX_PERSISTENT_FRAC,
    event_window_days: float = DEFAULT_EVENT_WINDOW_DAYS,
    min_outside_points: int = DEFAULT_MIN_OUTSIDE_POINTS,
):
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
