"""
Multi-survey photometry statistics shared by the transient filters (not a script).

    from tidez import multisurvey_photstats as photostats
    lc = photostats.combine(decam_lc, boom_lc)     # one DECam + ZTF + LSST stream (times in UT)
    stats = photostats.object_stats(lc).join(photostats.rates(lc))

Combined stream columns: objid, survey, filter, mjd, mag, magerr, limmag, detected.
Detections carry mag/magerr; upper limits have mag = NaN and limmag set (the
convention of both build_lcs.py and get_boom_lcs.py).

Rates are computed per *band*, where a band is a (survey, filter) pair -- DECam g,
ZTF g and LSST g are separate sequences -- so filter/zero-point differences between
surveys never produce a rate. On each band's epochs, sorted by time, neighbouring pairs
at least min_dt_days apart give:

  detection -> detection      rise or fade = |delta mag| / delta t, if the change is
                              >= min_delta_mag and >= sigma_thresh x its error
  upper limit -> detection    rise >= (limmag - mag) / delta t   (lower bound: the object
                              was fainter than the limit before)
  detection -> upper limit    fade >= (limmag - mag) / delta t   (lower bound)

Each epoch is paired with the first epoch in its band >= min_dt_days later. An upper limit
contradicted by a detection brighter than it within min_dt_days (a bad exposure) gives no
bound.

Rates are in mag/day and positive for both rising (brightening) and fading.

peak_rates() is the alternative, ZTFReST-style method: per band, rates are measured from
the peak (brightest detection) over the continuous rising run before it and the continuous
fading run after it (>= 3 epochs linked by consecutive significant steps). Both functions return the same columns.
"""

import numpy as np
import pandas as pd

MIN_DT_DAYS = 0.25
MIN_DELTA_MAG = 0.3
SIGMA_THRESH = 3.0
# peak_rates(): a rise/fade must be continuous -- >= MIN_RUN_POINTS points (peak included)
# linked by consecutive steps of >= STEP_MIN_MAG and >= STEP_SIGMA x error -- so one blip
# can't produce a rate.
MIN_RUN_POINTS = 3
STEP_MIN_MAG = 0.1
STEP_SIGMA = 2.0
# variability_stats(): window around the event outside which a real transient should be undetected
EVENT_WINDOW_DAYS = 7.0
MIN_OUTSIDE_POINTS = 4

LC_COLUMNS = ["objid", "survey", "filter", "mjd", "mag", "magerr", "limmag", "detected"]
BAND_KEYS = ["objid", "survey", "filter"]


# ---------------------------------------------------------------------------
# Combined stream
# ---------------------------------------------------------------------------

def combine(decam_lc: pd.DataFrame, boom_lc: pd.DataFrame | None = None) -> pd.DataFrame:
    """One light-curve stream from build_lcs.py (DECam) and get_boom_lcs.py (ZTF/LSST) output."""
    decam = decam_lc.rename(columns={"limmag5": "limmag"}).assign(survey="DECam")
    parts = [decam[LC_COLUMNS]]
    if boom_lc is not None and len(boom_lc):
        parts.append(boom_lc.rename(columns={"decam_objid": "objid"})[LC_COLUMNS])
    lc = pd.concat(parts, ignore_index=True)
    lc = lc[np.isfinite(lc["mjd"]) & (lc["filter"].fillna("") != "")]
    return lc.sort_values(["objid", "survey", "filter", "mjd"], ignore_index=True)


# ---------------------------------------------------------------------------
# Per-object statistics
# ---------------------------------------------------------------------------

def object_stats(lc: pd.DataFrame) -> pd.DataFrame:
    """Per objid: n_det, first/last detection mjd, age_days (first -> last detection, all
    surveys), the same for DECam alone, peak mag and its band, and the bands detected."""
    det = lc[lc["detected"]].assign(band=lambda d: d["survey"] + "-" + d["filter"])
    g = det.groupby("objid")
    stats = pd.DataFrame({
        "n_det": g["mjd"].size(),
        "first_det_mjd": g["mjd"].min(),
        "last_det_mjd": g["mjd"].max(),
        "bands_detected": g["band"].agg(lambda b: ",".join(sorted(set(b)))),
        "surveys_detected": g["survey"].agg(lambda s: ",".join(sorted(set(s)))),
    })
    stats["age_days"] = stats["last_det_mjd"] - stats["first_det_mjd"]
    peak = det.loc[det.groupby("objid")["mag"].idxmin(), ["objid", "mag", "band", "mjd"]].set_index("objid")
    stats = stats.join(peak.rename(columns={"mag": "peak_mag", "band": "peak_band", "mjd": "peak_mjd"}))
    decam = det[det["survey"] == "DECam"].groupby("objid")["mjd"]
    stats["decam_age_days"] = decam.max() - decam.min()
    return stats


def _contradicted_limits(lc: pd.DataFrame, window_days: float) -> np.ndarray:
    """True for an upper limit contradicted by a detection in the same band within
    window_days that is brighter than the limit (a bad exposure, not a non-detection).
    lc must be sorted by band then mjd."""
    det_mjd = lc["mjd"].where(lc["detected"])
    det_mag = lc["mag"].where(lc["detected"])
    prev = pd.DataFrame({"m": det_mjd, "g": det_mag}).groupby([lc[k] for k in BAND_KEYS], sort=False).ffill()
    nxt = pd.DataFrame({"m": det_mjd, "g": det_mag}).groupby([lc[k] for k in BAND_KEYS], sort=False).bfill()
    lim, mjd = lc["limmag"], lc["mjd"]
    near_prev = (mjd - prev["m"] <= window_days) & (prev["g"] < lim)
    near_next = (nxt["m"] - mjd <= window_days) & (nxt["g"] < lim)
    return (~lc["detected"] & (near_prev | near_next)).to_numpy()


def band_pairs(lc: pd.DataFrame, min_dt_days: float = MIN_DT_DAYS) -> pd.DataFrame:
    """Each epoch paired with the first epoch in the same band at least min_dt_days later,
    as rows with _a / _b columns. (Pairing with the immediate neighbour instead would lose
    the real change whenever a second exposure of the field follows within min_dt_days.)"""
    cols = ["mjd", "mag", "magerr", "limmag", "detected", "ul_contradicted"]
    lc = lc.sort_values(BAND_KEYS + ["mjd"], ignore_index=True)
    lc = lc.assign(ul_contradicted=_contradicted_limits(lc, min_dt_days))
    band_id = lc.groupby(BAND_KEYS, sort=False).ngroup().to_numpy()
    mjd = lc["mjd"].to_numpy()
    # bands are contiguous and time-sorted, so search each band's own slice of mjd
    start = np.r_[0, np.flatnonzero(np.diff(band_id)) + 1]
    end = np.r_[start[1:], len(lc)]
    j = np.empty(len(lc), dtype=np.int64)
    for s0, e0 in zip(start, end):
        j[s0:e0] = s0 + np.searchsorted(mjd[s0:e0], mjd[s0:e0] + min_dt_days, side="left")
    ok = j < np.repeat(end, end - start)
    a, b = lc[ok], lc.iloc[j[ok]]
    pairs = a[BAND_KEYS + cols].reset_index(drop=True).join(
        b[cols].reset_index(drop=True), lsuffix="_a", rsuffix="_b")
    return pairs.assign(dt=pairs["mjd_b"] - pairs["mjd_a"])


def pair_rates(pairs: pd.DataFrame, min_delta_mag: float = MIN_DELTA_MAG,
               sigma_thresh: float = SIGMA_THRESH) -> pd.DataFrame:
    """Add rise / fade rates (mag/day, NaN where the pair doesn't constrain them) and the
    pair type: 'det-det' (measured), 'ul-det' / 'det-ul' (lower bounds)."""
    dd = pairs["detected_a"] & pairs["detected_b"]
    # an upper limit contradicted by a nearby brighter detection gives no bound
    ud = ~pairs["detected_a"] & pairs["detected_b"] & ~pairs["ul_contradicted_a"]
    du = pairs["detected_a"] & ~pairs["detected_b"] & ~pairs["ul_contradicted_b"]

    brighten = pairs["mag_a"] - pairs["mag_b"]                       # > 0 = got brighter
    err = np.hypot(pairs["magerr_a"].fillna(0.0), pairs["magerr_b"].fillna(0.0))
    significant = (brighten.abs() >= min_delta_mag) & (brighten.abs() >= sigma_thresh * err)
    rise_ul = pairs["limmag_a"] - pairs["mag_b"]                     # was fainter than limmag_a
    fade_ul = pairs["limmag_b"] - pairs["mag_a"]                     # became fainter than limmag_b

    rise = np.select([dd & significant & (brighten > 0), ud & (rise_ul >= min_delta_mag)],
                     [brighten, rise_ul], default=np.nan) / pairs["dt"]
    fade = np.select([dd & significant & (brighten < 0), du & (fade_ul >= min_delta_mag)],
                     [-brighten, fade_ul], default=np.nan) / pairs["dt"]
    kind = np.select([dd, ud, du], ["det-det", "ul-det", "det-ul"], default="none")
    return pairs.assign(rise_rate=rise, fade_rate=fade, pair_type=kind)


def _best_per_object(candidates: pd.DataFrame) -> pd.DataFrame:
    """candidates: one row per possible rate with objid, side ('rise'/'fade'), rate, band,
    pair_type, dt, mjd. Returns per objid and side the fastest rate (+ band, pair type, dt,
    mjd) and the fastest *measured* ('det-det') rate."""
    out = []
    for side in ("rise", "fade"):
        c = candidates[(candidates["side"] == side) & candidates["rate"].notna()]
        best = c.loc[c.groupby("objid")["rate"].idxmax(), ["objid", "rate", "band", "pair_type", "dt", "mjd"]]
        out.append(best.set_index("objid").rename(columns={
            "rate": f"{side}_rate", "band": f"{side}_band", "pair_type": f"{side}_pair",
            "dt": f"{side}_dt", "mjd": f"{side}_mjd"}))
        measured = c[c["pair_type"] == "det-det"].groupby("objid")["rate"].max()
        out.append(measured.rename(f"{side}_rate_measured"))
    return pd.concat(out, axis=1)


def rates(lc: pd.DataFrame, min_dt_days: float = MIN_DT_DAYS, min_delta_mag: float = MIN_DELTA_MAG,
          sigma_thresh: float = SIGMA_THRESH) -> pd.DataFrame:
    """Consecutive-pair method. Per objid, for rise and fade separately: the fastest rate over
    all bands, which band and pair type gave it ('det-det' = measured, 'ul-det'/'det-ul' =
    lower bound), its delta t and mjd, plus the fastest *measured* rate on its own."""
    pairs = pair_rates(band_pairs(lc, min_dt_days), min_delta_mag, sigma_thresh)
    pairs = pairs.assign(band=pairs["survey"] + "-" + pairs["filter"])
    candidates = pd.concat([
        pairs.assign(side=side, rate=pairs[f"{side}_rate"], mjd=pairs["mjd_b"])
        [["objid", "side", "rate", "band", "pair_type", "dt", "mjd"]]
        for side in ("rise", "fade")])
    return _best_per_object(candidates)


def _run(order, mag, err, lim, det, k, step_min_mag, step_sigma):
    """Indices of the continuous run that ends (rise) or starts (fade) at the peak k, walking
    away from it in `order`: each next point must be fainter than the one before by
    >= step_min_mag and >= step_sigma x the step's error. An upper limit can end the run (if
    deeper than the last detection by >= step_min_mag) but nothing goes past it."""
    run = [k]
    for j in order:
        last = run[-1]
        if det[j]:
            step = mag[j] - mag[last]
            if step >= step_min_mag and step >= step_sigma * np.hypot(np.nan_to_num(err[j]), np.nan_to_num(err[last])):
                run.append(j)
                continue
        elif lim[j] - mag[last] >= step_min_mag:
            run.append(j)
        break
    return run


def peak_rates(lc: pd.DataFrame, min_dt_days: float = MIN_DT_DAYS, min_delta_mag: float = MIN_DELTA_MAG,
               sigma_thresh: float = SIGMA_THRESH, min_run_points: int = MIN_RUN_POINTS,
               step_min_mag: float = STEP_MIN_MAG, step_sigma: float = STEP_SIGMA) -> pd.DataFrame:
    """Peak-anchored method, after the simplified ZTFReST criterion (Andreoni & Coughlin et al.
    2021), requiring the evolution to be *continuous* so a single blip can't make a rate.

    In each band (survey + filter), anchor on the brightest detection (the peak). Walking back
    from it, the rise run is the consecutive points that each get fainter (going back in time)
    by >= step_min_mag and >= step_sigma x the step error; the fade run is the same walking
    forward. A deep enough upper limit can end a run. A side counts only if its run has
    >= min_run_points points including the peak (default 3: two consecutive significant steps),
    and its total change is >= min_delta_mag and >= sigma_thresh x its error. The rate is over
    the whole run, as in ZTFReST:

      run starts at a detection    (mag - peak_mag) / |t - t_peak|        pair 'det-det' (measured)
      run starts at an upper limit (limmag - peak_mag) / |t - t_peak|     'ul-det' / 'det-ul' (bound)

    Same output columns as rates(); *_rate_measured is the rate from the run's farthest detection.
    """
    epochs = lc.sort_values(BAND_KEYS + ["mjd"], ignore_index=True)
    band_start = np.flatnonzero(~epochs.duplicated(BAND_KEYS).to_numpy())
    bounds = np.append(band_start, len(epochs))
    mjd, mag, err = (epochs[c].to_numpy(dtype=float) for c in ("mjd", "mag", "magerr"))
    lim, det = epochs["limmag"].to_numpy(dtype=float), epochs["detected"].to_numpy(dtype=bool)
    objid, band = epochs["objid"].to_numpy(), (epochs["survey"] + "-" + epochs["filter"]).to_numpy()

    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        bdet = det[a:b]
        if not bdet.any():
            continue
        m, e, l, d, t = mag[a:b], err[a:b], lim[a:b], bdet, mjd[a:b]
        k = int(np.nanargmin(np.where(d, m, np.nan)))
        for side, order in (("rise", range(k - 1, -1, -1)), ("fade", range(k + 1, b - a))):
            run = _run(order, m, e, l, d, k, step_min_mag, step_sigma)
            if len(run) < min_run_points:
                continue
            end = run[-1]
            dt = abs(t[end] - t[k])
            if dt < min_dt_days:
                continue
            if d[end]:
                delta = m[end] - m[k]
                if delta < min_delta_mag or delta < sigma_thresh * np.hypot(np.nan_to_num(e[end]), np.nan_to_num(e[k])):
                    continue
                pair = "det-det"
            else:
                delta = l[end] - m[k]
                if delta < min_delta_mag:
                    continue
                pair = "ul-det" if side == "rise" else "det-ul"
            far_det = max((j for j in run if d[j]), key=lambda j: abs(t[j] - t[k]))
            measured = (m[far_det] - m[k]) / abs(t[far_det] - t[k]) if far_det != k else np.nan
            rows.append((objid[a], side, delta / dt, band[a], pair, dt, t[end], measured, len(run)))

    cand = pd.DataFrame(rows, columns=["objid", "side", "rate", "band", "pair_type", "dt", "mjd",
                                       "measured", "run_points"])
    best = _best_per_object(cand)
    # measured rate and run length of the run that gave each object's best rate
    for side in ("rise", "fade"):
        c = cand[cand["side"] == side]
        c = c.loc[c.groupby("objid")["rate"].idxmax()].set_index("objid") if len(c) else c.set_index("objid")
        best[f"{side}_rate_measured"] = c["measured"].reindex(best.index)
        best[f"{side}_run_points"] = c["run_points"].reindex(best.index)
    return best


# ---------------------------------------------------------------------------
# Stochastic variability (AGN-like flickering, persistent sources)
# ---------------------------------------------------------------------------

def variability_stats(lc: pd.DataFrame, rates_df: pd.DataFrame | None = None,
                      event_window_days: float = EVENT_WINDOW_DAYS,
                      min_outside_points: int = MIN_OUTSIDE_POINTS,
                      step_min_mag: float = STEP_MIN_MAG, step_sigma: float = STEP_SIGMA) -> pd.DataFrame:
    """Per objid, the two stochastic-variability signals of the old transient_filters.is_stochastic:

      reversal_frac    in each band, how often consecutive *significant* detection-to-detection
                       steps (>= step_min_mag and >= step_sigma x error) change direction; the
                       largest over bands with >= 3 significant steps. A single rise-and-fade
                       event flips once; flickering flips often. (Unlike the old version,
                       insignificant wiggles don't count, so noise alone doesn't look stochastic.)
      persistent_frac  fraction of epochs (all surveys/bands) *outside* the event window
                       that are detections, where the window is the rise run's start to the fade
                       run's end (from rates_df; the peak if absent) +- event_window_days. A real
                       transient is undetected away from its event; a variable source isn't.
                       NaN with fewer than min_outside_points points outside.
    """
    det = lc[lc["detected"]].sort_values(BAND_KEYS + ["mjd"])
    g = det.groupby(BAND_KEYS, sort=False)
    step = g["mag"].diff()
    err = np.hypot(det["magerr"].fillna(0.0), g["magerr"].shift().fillna(0.0))
    sig = det.assign(sign=np.sign(step))[(step.abs() >= step_min_mag) & (step.abs() >= step_sigma * err)]
    gs = sig.groupby(BAND_KEYS, sort=False)["sign"]
    flips = (sig["sign"] != gs.shift()) & gs.shift().notna()
    per_band = pd.DataFrame({"n": gs.size(), "flips": flips.groupby([sig[k] for k in BAND_KEYS]).sum()})
    per_band = per_band[per_band["n"] >= 3]
    reversal = (per_band["flips"] / (per_band["n"] - 1)).groupby(level="objid").max()

    peak_mjd = det.loc[det.groupby("objid")["mag"].idxmin()].set_index("objid")["mjd"]
    lo = hi = peak_mjd
    if rates_df is not None:
        lo = rates_df["rise_mjd"].reindex(peak_mjd.index).fillna(peak_mjd)
        hi = rates_df["fade_mjd"].reindex(peak_mjd.index).fillna(peak_mjd)
    window = pd.DataFrame({"lo": lo - event_window_days, "hi": hi + event_window_days})
    x = lc.join(window, on="objid")
    outside = x[(x["mjd"] < x["lo"]) | (x["mjd"] > x["hi"])].groupby("objid")["detected"].agg(["size", "mean"])
    persistent = outside["mean"].where(outside["size"] >= min_outside_points)

    return pd.DataFrame({"reversal_frac": reversal, "persistent_frac": persistent}).reindex(peak_mjd.index)
