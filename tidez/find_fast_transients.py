"""
Find fast transients on the combined DECam + ZTF + LSST light curves.

1. Basic cuts (tidez.basic_filtering): not a rock (>= 2 detections, or a galaxy nearby),
   not a star, not an AGN/quasar.
2. Age: first -> last detection across all surveys < --max-age-days (default 50).
3. Not stochastic: reversal_frac < --max-reversal-frac (0.4), i.e. the light curve doesn't
   keep changing direction (AGN-like flickering; only significant steps count) -- see
   multisurvey_photostats.variability_stats.
4. Rates (tidez.multisurvey_photostats), computed independently in each band (survey + filter)
   on the individual epochs (UT): rise >= --rise-threshold AND fade >= --fade-threshold
   (default 0.5 mag/day each; --require either to accept just one). A rate counts
   whether it is measured between two detections or is a lower bound from an upper
   limit next to a detection (non-detection -> detection for rises, detection ->
   non-detection for fades). --method picks how rates are measured:
     pairs  (default) between neighbouring epochs in a band: a pair counts if it is
            >= --min-dt-days apart (6 h) and changes by >= --min-delta-mag (0.5) and
            >= --sigma-thresh (4) x its error
     peak   ZTFReST-style: over the continuous rise into each band's peak and the
            continuous fade after it (>= --min-run-points points linked by consecutive
            significant steps, so a single blip can't make a rate)
   --filters restricts the rates to some filters (e.g. --filters i z y, like ZTFReST's
   selectRed).

The log prints the funnel: objects left after each cut, and how many each cut removes
on its own.

Outputs:
  data/fast_transients.csv          one row per candidate, fastest first (by the slower of
                                    rise and fade): flags, age, rates (which band, measured
                                    or bound, dt, run length), variability, peak, TNS match
  data/fast_transients.txt          candidate objids, one per line (for the scanner)
  data/fast_transients_stats.parquet  the same columns for *every* object (for tuning cuts)

Usage:
    python tidez/find_fast_transients.py
    python tidez/find_fast_transients.py --rise-threshold 1.0 --require either
    python tidez/find_fast_transients.py --method peak --filters i z y
    python tidez/find_fast_transients.py --min-delta-mag 0.3 --sigma-thresh 3
    python tidez/find_fast_transients.py --lc data/lightcurves_any_status.parquet --no-boom
"""

import argparse
import pathlib

import pandas as pd

from tidez import multisurvey_photstats as photostats
from tidez.basic_filtering import basic_flags, flag_summary
from tidez.logger import get_logger
logger = get_logger("find_fast_transients")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LC_FILE = DATA_DIR / "lightcurves.parquet"
BOOM_FILE = DATA_DIR / "boom_lc.parquet"
OUT_PREFIX = DATA_DIR / "fast_transients"

MAX_AGE_DAYS = 50.0
RISE_THRESHOLD = 1  # mag/day
FADE_THRESHOLD = 0.25  # mag/day
MIN_DT_DAYS = 1     # in fraction of days
MIN_DELTA_MAG = 0.5
SIGMA_THRESH = 4.0
MAX_REVERSAL_FRAC = 0.4

OUTPUT_COLUMNS = [
    "age_days", "decam_age_days", "n_det", "first_det_mjd", "last_det_mjd",
    "rise_rate", "rise_band", "rise_pair", "rise_dt", "rise_mjd", "rise_rate_measured", "rise_run_points",
    "fade_rate", "fade_band", "fade_pair", "fade_dt", "fade_mjd", "fade_rate_measured", "fade_run_points",
    "peak_mag", "peak_band", "peak_mjd", "bands_detected", "surveys_detected",
    "reversal_frac",
    "is_rock", "is_star", "is_agn", "is_stochastic", "is_variable", "is_on_tns",
    "tns_name", "tns_type", "tns_redshift", "tns_discoverydate", "tns_internal_names", "tns_sep_arcsec",
]


def load_lightcurves(lc_file: pathlib.Path, boom_file: pathlib.Path | None) -> pd.DataFrame:
    decam = pd.read_parquet(lc_file, columns=["objid", "mjd", "filter", "mag", "magerr", "limmag5", "detected"])
    logger.info(f"DECam: {len(decam)} epochs for {decam['objid'].nunique()} objects from {lc_file}")
    boom = None
    if boom_file is not None:
        if boom_file.exists():
            boom = pd.read_parquet(boom_file, columns=["decam_objid", "survey", "mjd", "filter", "mag",
                                                       "magerr", "limmag", "detected"])
            logger.info(f"BOOM: {len(boom)} ZTF/LSST rows for {boom['decam_objid'].nunique()} objects from {boom_file}")
        else:
            logger.warning(f"{boom_file} not found -- DECam only")
    return photostats.combine(decam, boom)


def find_fast_transients(lc: pd.DataFrame, max_age_days: float, rise_threshold: float,
                         fade_threshold: float, require: str, method: str = "pairs",
                         filters: list | None = None, min_run_points: int = photostats.MIN_RUN_POINTS,
                         max_reversal_frac: float = MAX_REVERSAL_FRAC, min_dt_days: float = MIN_DT_DAYS,
                         min_delta_mag: float = MIN_DELTA_MAG, sigma_thresh: float = SIGMA_THRESH) -> tuple:
    """Returns (stats for every object, boolean Series of candidates)."""
    # objids = pd.Index(lc["objid"].unique(), name="objid")
    objids = pd.Index(
    lc.loc[lc["objid"].str.startswith("T"), "objid"].unique(),
    name="objid"
)
    flags = basic_flags(objids, lc)
    logger.info("Basic flags:\n" + flag_summary(flags).to_string())

    rate_lc = lc if not filters else lc[lc["filter"].isin(filters)]
    cuts = dict(min_dt_days=min_dt_days, min_delta_mag=min_delta_mag, sigma_thresh=sigma_thresh)
    if method == "peak":
        band_rates = photostats.peak_rates(rate_lc, min_run_points=min_run_points, **cuts)
    else:
        band_rates = photostats.rates(rate_lc, **cuts)
    logger.info(f"Rates: {method} method, dt >= {min_dt_days:g} d, dmag >= {min_delta_mag:g}, >= {sigma_thresh:g} sigma"
                + (f", runs of >= {min_run_points} points" if method == "peak" else "")
                + (f", filters {' '.join(filters)}" if filters else ", all filters"))
    stats = (flags.join(photostats.object_stats(lc)).join(band_rates)
             .join(photostats.variability_stats(lc, band_rates)))
    stats["is_stochastic"] = stats["reversal_frac"] >= max_reversal_frac
    stats["score"] = stats[["rise_rate", "fade_rate"]].min(axis=1)

    steps = [
        ("not rock", ~stats["is_rock"]),
        ("not star", ~stats["is_star"]),
        ("not AGN/quasar", ~stats["is_agn"]),
        (f"age < {max_age_days:g} d", stats["age_days"] < max_age_days),
        ("not stochastic", ~stats["is_stochastic"]),
    ]
    rise_ok = stats["rise_rate"] >= rise_threshold
    fade_ok = stats["fade_rate"] >= fade_threshold
    if require == "both":
        steps.append((f"rise >= {rise_threshold:g} and fade >= {fade_threshold:g} mag/d", rise_ok & fade_ok))
    else:
        steps.append((f"rise >= {rise_threshold:g} or fade >= {fade_threshold:g} mag/d", rise_ok | fade_ok))

    passed = pd.Series(True, index=stats.index)
    lines = [f"Funnel: {len(passed)} objects", f"  {'cut':40s} {'left':>9s} {'removed alone':>14s}"]
    for name, cut in steps:
        cut = cut.fillna(False)
        passed &= cut
        lines.append(f"  {name:40s} {int(passed.sum()):>9d} {int((~cut).sum()):>14d}")
    lines.append(f"  => {int(passed.sum())} of {len(passed)} objects pass "
                 f"({100 * passed.mean():.3f}%)")
    logger.info("\n".join(lines))
    return stats, passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Find fast transients on the combined DECam/ZTF/LSST light curves.")
    parser.add_argument("--lc", type=pathlib.Path, default=LC_FILE, help="DECam light-curve Parquet (get_lcs.py).")
    parser.add_argument("--boom", type=pathlib.Path, default=BOOM_FILE, help="ZTF/LSST Parquet (get_boom_lcs.py).")
    parser.add_argument("--no-boom", action="store_true", help="Use the DECam light curves only.")
    parser.add_argument("--out-prefix", type=pathlib.Path, default=OUT_PREFIX)
    parser.add_argument("--max-age-days", type=float, default=MAX_AGE_DAYS)
    parser.add_argument("--rise-threshold", type=float, default=RISE_THRESHOLD, help="mag/day")
    parser.add_argument("--fade-threshold", type=float, default=FADE_THRESHOLD, help="mag/day")
    parser.add_argument("--require", choices=["both", "either"], default="both",
                        help="Require a fast rise AND a fast fade (default), or either one.")
    parser.add_argument("--method", choices=["pairs", "peak"], default="pairs",
                        help="pairs: between neighbouring epochs (default); "
                             "peak: ZTFReST-style continuous runs around each band's peak.")
    parser.add_argument("--min-dt-days", type=float, default=MIN_DT_DAYS,
                        help="Minimum time between the two points of a rate (default 0.25 d = 6 h).")
    parser.add_argument("--min-delta-mag", type=float, default=MIN_DELTA_MAG,
                        help="Minimum magnitude change for a rate to count (default 0.5).")
    parser.add_argument("--sigma-thresh", type=float, default=SIGMA_THRESH,
                        help="Minimum significance of that change, in sigma (default 4).")
    parser.add_argument("--min-run-points", type=int, default=photostats.MIN_RUN_POINTS,
                        help="peak method: points (peak included) a continuous rise/fade needs (default 3).")
    parser.add_argument("--max-reversal-frac", type=float, default=MAX_REVERSAL_FRAC)
    parser.add_argument("--filters", nargs="+", default=None,
                        help="Only use these filters for the rates (e.g. i z y). Default: all.")
    args = parser.parse_args()

    lc = load_lightcurves(args.lc, None if args.no_boom else args.boom)
    stats, passed = find_fast_transients(lc, args.max_age_days, args.rise_threshold,
                                         args.fade_threshold, args.require, args.method, args.filters,
                                         args.min_run_points, args.max_reversal_frac,
                                         args.min_dt_days, args.min_delta_mag, args.sigma_thresh)

    out = args.out_prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    stats.assign(passed=passed).to_parquet(out.with_name(out.name + "_stats.parquet"))
    candidates = stats.loc[passed].sort_values("score", ascending=False).reindex(columns=OUTPUT_COLUMNS)
    candidates.to_csv(out.with_suffix(".csv"), float_format="%.4f")
    out.with_suffix(".txt").write_text("".join(f"{o}\n" for o in candidates.index))
    logger.info(f"{len(candidates)} fast-transient candidates -> {out.with_suffix('.csv')} and .txt; "
                f"stats for all {len(stats)} objects -> {out.name}_stats.parquet")


if __name__ == "__main__":
    main()
