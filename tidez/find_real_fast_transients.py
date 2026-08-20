"""
End-to-end real-fast-transient pipeline that only pays for crossmatching on
objects that already look interesting, instead of crossmatching everything
in the master H5 up front.

By default (--enable-agn-veto not set) this runs two stages:

  Stage 1 -- Rate cut (cheap, no crossmatch data needed): the same
  quality-gated, asymmetric rise/fade rate cut as afterglow_hunt.py
  (transient_filters.find_best_rate_pair), run against every object's
  DECam('p'-only)+ZTF+LSST lightcurve.

  Stage 4 -- Legacy Survey cutout fetch for stage-1 survivors that don't
  already have a /legacy_survey group. Same approach as
  crossmatch_legacy_survey.py.

Pass --enable-agn-veto to also run the Milliquas/stochastic-variability
AGN-rejection stages in between (currently disabled by default -- under
investigation for a possible concurrent-access data consistency issue on
Bridges2's filesystem when other processes touch the H5 at the same time):

  Stage 2 -- Milliquas crossmatch, only for stage-1 survivors that don't
  already have a /milliquas group. Same bulk CDS X-Match approach as
  crossmatch_milliquas.py.

  Stage 3 -- AGN veto on stage-1 survivors: reject a Milliquas match, or
  stochastic-variability signals (transient_filters.is_stochastic) --
  same logic as afterglow_hunt.py's stages 1+2.

This assumes crossmatch_boom.py has already been run against the full H5
separately -- ZTF/LSST crossmatching isn't part of this pipeline, only used
(if present) to pool into each object's lightcurve and to decide the
crossmatched-output split below.

Writes two files: every final survivor, and the subset that also has a
ZTF/Rubin crossmatch (from a prior crossmatch_boom.py run).

Usage:
    python -m tidez.find_real_fast_transients
    python -m tidez.find_real_fast_transients --h5 data/tidez_master.h5 --rise-threshold 0.5
    python -m tidez.find_real_fast_transients --enable-agn-veto
"""

import argparse
import pathlib
import time

import h5py
from tqdm import tqdm

from tidez.crossmatch_legacy_survey import (
    DEFAULT_LAYER,
    DEFAULT_PIXSCALE,
    DEFAULT_SIZE_PIX,
    DEFAULT_SLEEP,
    fetch_cutout_bytes,
    write_legacy_cutout,
)
from tidez.build_h5 import CATALOG_DTYPE
from tidez.crossmatch_milliquas import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MILLIQUAS_CATALOG,
    DEFAULT_RADIUS_ARCSEC,
    _xmatch_chunk,
    write_milliquas_match,
)
from tidez.logger import get_logger
from tidez.plot_lightcurve import DEFAULT_H5, load_object_rows
from tidez.transient_filters import (
    DEFAULT_BLIP_REVERT_TOL,
    DEFAULT_BLIP_WINDOW_DAYS,
    DEFAULT_EVENT_WINDOW_DAYS,
    DEFAULT_FADE_THRESHOLD,
    DEFAULT_MAX_BASELINE_DAYS,
    DEFAULT_MAX_DT_DAYS,
    DEFAULT_MAX_PERSISTENT_FRAC,
    DEFAULT_MAX_REVERSAL_FRAC,
    DEFAULT_MIN_DELTA_MAG,
    DEFAULT_MIN_DETECTIONS,
    DEFAULT_MIN_DT_DAYS,
    DEFAULT_MIN_OUTSIDE_POINTS,
    DEFAULT_MIN_SNR,
    DEFAULT_RISE_THRESHOLD,
    DEFAULT_SIGMA_THRESH,
    detection_span_days,
    find_best_rate_pair,
    has_crossmatch,
    is_stochastic,
    milliquas_matched,
)

logger = get_logger("find_real_fast_transients")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "fast_real_transients.txt"
DEFAULT_CROSSMATCH_OUTPUT = DATA_DIR / "fast_real_transients_crossmatched.txt"
DEFAULT_SUBSET_H5 = DATA_DIR / "fast_real_transients.h5"


def write_subset_h5(source_h5file, objids, output_path: pathlib.Path):
    """Copy just the given objids' full groups (lightcurve, cutouts,
    crossmatch, milliquas, legacy_survey -- whatever each has) plus a
    matching /catalog table into a new, much smaller H5 file. Lets the
    webapp browse just the final candidates without loading/indexing the
    full ~100k-object master H5."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {len(objids)}-object candidate subset to {output_path} ...")
    with h5py.File(output_path, "w") as out_h5file:
        catalog = out_h5file.create_dataset(
            "catalog", shape=(len(objids),), maxshape=(None,), dtype=CATALOG_DTYPE, chunks=True,
        )
        for i, objid in enumerate(tqdm(objids, desc="Writing candidate subset H5", unit="obj")):
            source_h5file.copy(source_h5file[objid], out_h5file, name=objid)
            ra = float(source_h5file[objid].attrs["ra"])
            dec = float(source_h5file[objid].attrs["dec"])
            n_epochs = source_h5file[objid]["lightcurve"].shape[0]
            catalog[i] = (objid.encode("ascii"), ra, dec, n_epochs)
    logger.info(f"Done writing {output_path}.")


def run(
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
    milliquas_radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
    milliquas_catalog: str = DEFAULT_MILLIQUAS_CATALOG,
    milliquas_chunk_size: int = DEFAULT_CHUNK_SIZE,
    legacy_layer: str = DEFAULT_LAYER,
    legacy_pixscale: float = DEFAULT_PIXSCALE,
    legacy_size_pix: int = DEFAULT_SIZE_PIX,
    legacy_sleep: float = DEFAULT_SLEEP,
    enable_agn_veto: bool = False,
    max_baseline_days: float | None = DEFAULT_MAX_BASELINE_DAYS,
    subset_h5: pathlib.Path | None = DEFAULT_SUBSET_H5,
):
    stage_counts = {
        "total": 0, "rate_cut_passed": 0, "too_old_rejected": 0,
        "milliquas_rejected": 0, "stochastic_rejected": 0, "final_passed": 0,
    }

    logger.info(f"Opening {h5_path} ...")
    with h5py.File(h5_path, "a") as h5file:
        objids = sorted(k for k in h5file.keys() if k != "catalog")
        stage_counts["total"] = len(objids)
        logger.info(f"{len(objids)} objects in {h5_path}.")

        # Stage 1: rate cut on every object -- no crossmatch data needed.
        rate_survivors = []
        for objid in tqdm(objids, desc="Stage 1/4: rate cut", unit="obj"):
            rows = load_object_rows(h5file, objid, good_only=True)
            detected = [r for r in rows if r["detected"]]
            if len(detected) < min_detections:
                continue
            if max_baseline_days is not None and detection_span_days(detected) > max_baseline_days:
                # Detected over too long a span to be a single fast transient
                # event -- looks like a long-lived variable instead.
                stage_counts["too_old_rejected"] += 1
                continue
            best = find_best_rate_pair(
                detected, rise_threshold, fade_threshold, min_dt_days, max_dt_days,
                min_delta_mag, sigma_thresh, min_snr, blip_window_days, blip_revert_tol,
            )
            if best is None:
                continue
            score, rate, direction, significance, band_rows, row_a, row_b = best
            rate_survivors.append((objid, score, rate, direction, significance, band_rows, row_a, row_b, rows))

        stage_counts["rate_cut_passed"] = len(rate_survivors)
        logger.info(
            f"Stage 1/4 (rate cut): {len(rate_survivors)}/{len(objids)} objects passed "
            f"({stage_counts['too_old_rejected']} rejected for a >{max_baseline_days}-day detection span)."
        )

        if enable_agn_veto:
            # Stage 2: Milliquas crossmatch, only for survivors lacking it.
            need_milliquas = [
                c for c in tqdm(rate_survivors, desc="Checking existing /milliquas groups", unit="obj")
                if "milliquas" not in h5file[c[0]]
            ]
            logger.info(
                f"Stage 2/4 (Milliquas): crossmatching {len(need_milliquas)} rate-cut survivors "
                f"({len(rate_survivors) - len(need_milliquas)} already crossmatched)."
            )
            chunks = [
                need_milliquas[i:i + milliquas_chunk_size]
                for i in range(0, len(need_milliquas), milliquas_chunk_size)
            ]
            for i, chunk in enumerate(tqdm(chunks, desc="Stage 2/4: Milliquas crossmatch", unit="chunk"), start=1):
                coord_rows = [
                    (c[0], float(h5file[c[0]].attrs["ra"]), float(h5file[c[0]].attrs["dec"]))
                    for c in chunk
                ]
                logger.info(f"Requesting Milliquas chunk {i}/{len(chunks)} ({len(chunk)} objects) from CDS X-Match ...")
                try:
                    matches = _xmatch_chunk(coord_rows, milliquas_radius_arcsec, milliquas_catalog)
                except Exception:
                    logger.error(f"Failed to crossmatch a Milliquas chunk of {len(chunk)} objects", exc_info=True)
                    continue
                for c in chunk:
                    write_milliquas_match(h5file, c[0], matches.get(c[0]))
                h5file.flush()

            # Stage 3: AGN veto (Milliquas match, or stochastic variability).
            final_survivors = []
            for c in tqdm(rate_survivors, desc="Stage 3/4: AGN veto", unit="obj"):
                objid, score, rate, direction, significance, band_rows, row_a, row_b, rows = c
                if milliquas_matched(h5file, objid, logger=logger):
                    stage_counts["milliquas_rejected"] += 1
                    continue
                if is_stochastic(rows, band_rows, row_a, row_b, max_reversal_frac, max_persistent_frac,
                                  event_window_days, min_outside_points):
                    stage_counts["stochastic_rejected"] += 1
                    continue
                final_survivors.append(c)

            stage_counts["final_passed"] = len(final_survivors)
            logger.info(
                f"Stage 3/4 (AGN veto): {stage_counts['milliquas_rejected']} rejected (Milliquas match), "
                f"{stage_counts['stochastic_rejected']} rejected (stochastic variability), "
                f"{len(final_survivors)} passed."
            )
        else:
            logger.info("Stages 2/3 (Milliquas crossmatch + AGN veto) skipped (--enable-agn-veto not set).")
            final_survivors = rate_survivors
            stage_counts["final_passed"] = len(final_survivors)

        # Stage 4: Legacy Survey cutouts, only for final survivors lacking one.
        need_legacy = [
            c for c in tqdm(final_survivors, desc="Checking existing /legacy_survey groups", unit="obj")
            if "legacy_survey" not in h5file[c[0]]
        ]
        logger.info(
            f"Stage 4/4 (Legacy Survey): fetching cutouts for {len(need_legacy)} final survivors "
            f"({len(final_survivors) - len(need_legacy)} already fetched)."
        )
        n_legacy_failed = 0
        for c in tqdm(need_legacy, desc="Stage 4/4: Legacy Survey cutouts", unit="obj"):
            objid = c[0]
            ra = float(h5file[objid].attrs["ra"])
            dec = float(h5file[objid].attrs["dec"])
            try:
                jpg_bytes = fetch_cutout_bytes(ra, dec, legacy_layer, legacy_pixscale, legacy_size_pix)
                write_legacy_cutout(h5file, objid, jpg_bytes, legacy_layer, legacy_pixscale, legacy_size_pix)
            except Exception:
                logger.error(f"Failed to fetch Legacy Survey cutout for {objid} (ra={ra}, dec={dec})", exc_info=True)
                n_legacy_failed += 1
            time.sleep(legacy_sleep)
        h5file.flush()

        final_survivors.sort(key=lambda c: c[1], reverse=True)  # score descending
        crossmatched_objids = {
            c[0] for c in tqdm(final_survivors, desc="Checking ZTF/Rubin crossmatch", unit="obj")
            if has_crossmatch(h5file, c[0])
        }

        if subset_h5 is not None:
            write_subset_h5(h5file, [c[0] for c in final_survivors], subset_h5)

    return final_survivors, crossmatched_objids, stage_counts, n_legacy_failed


def main():
    parser = argparse.ArgumentParser(
        description="Rate cut -> Milliquas crossmatch + AGN veto -> Legacy Survey crossmatch, "
                     "each stage only run against the previous stage's survivors (see module docstring)."
    )
    parser.add_argument("--h5", type=pathlib.Path, default=DEFAULT_H5,
                         help="Master H5 file to read from and write crossmatches into.")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT,
                         help="Output text file of all final-survivor objids, one per line.")
    parser.add_argument("--crossmatch-output", type=pathlib.Path, default=DEFAULT_CROSSMATCH_OUTPUT,
                         help="Output text file of final survivors that also have a "
                              "ZTF/Rubin crossmatch, one per line.")
    parser.add_argument("--rise-threshold", type=float, default=DEFAULT_RISE_THRESHOLD,
                         help="Minimum brightening rate (mag/day) to qualify (default: 0.7).")
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
    parser.add_argument("--max-baseline-days", type=float, default=DEFAULT_MAX_BASELINE_DAYS,
                         help="Reject objects detected over a longer span than this (default: "
                              "200.0) -- looks like a long-lived variable, not a single fast "
                              "transient event. Use --no-max-baseline to disable entirely.")
    parser.add_argument("--no-max-baseline", action="store_true",
                         help="Disable the max-baseline-days cut entirely.")
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
    parser.add_argument("--milliquas-radius-arcsec", type=float, default=DEFAULT_RADIUS_ARCSEC,
                         help="Milliquas cross-match radius in arcseconds (default: 3.0).")
    parser.add_argument("--milliquas-catalog", type=str, default=DEFAULT_MILLIQUAS_CATALOG,
                         help=f"VizieR catalog id for X-Match (default: {DEFAULT_MILLIQUAS_CATALOG}).")
    parser.add_argument("--milliquas-chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                         help="Rate-cut survivors per X-Match request (default: 5000).")
    parser.add_argument("--legacy-layer", type=str, default=DEFAULT_LAYER,
                         help="Legacy Survey data release layer (default: ls-dr10).")
    parser.add_argument("--legacy-pixscale", type=float, default=DEFAULT_PIXSCALE,
                         help="Legacy Survey cutout pixel scale in arcsec/pixel (default: 0.262).")
    parser.add_argument("--legacy-size", type=int, default=DEFAULT_SIZE_PIX, dest="legacy_size_pix",
                         help="Legacy Survey cutout size in pixels (default: 150).")
    parser.add_argument("--legacy-sleep", type=float, default=DEFAULT_SLEEP,
                         help="Seconds to sleep between Legacy Survey requests (default: 0.2).")
    parser.add_argument("--enable-agn-veto", action="store_true",
                         help="Also run the Milliquas crossmatch + AGN veto (stages 2-3). Off by "
                              "default: rate-cut survivors go straight to Legacy Survey crossmatch.")
    parser.add_argument("--subset-h5", type=pathlib.Path, default=DEFAULT_SUBSET_H5,
                         help="Also write a smaller H5 file containing just the final survivors' "
                              "full data (lightcurve, cutouts, crossmatch, milliquas, legacy_survey) "
                              "-- for faster webapp browsing (default: data/fast_real_transients.h5). "
                              "Use --no-subset-h5 to skip this.")
    parser.add_argument("--no-subset-h5", action="store_true",
                         help="Skip writing the candidate subset H5 file.")
    args = parser.parse_args()

    final_survivors, crossmatched_objids, stage_counts, n_legacy_failed = run(
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
        max_baseline_days=None if args.no_max_baseline else args.max_baseline_days,
        max_reversal_frac=args.max_reversal_frac,
        max_persistent_frac=args.max_persistent_frac,
        event_window_days=args.event_window_days,
        min_outside_points=args.min_outside_points,
        milliquas_radius_arcsec=args.milliquas_radius_arcsec,
        milliquas_catalog=args.milliquas_catalog,
        milliquas_chunk_size=args.milliquas_chunk_size,
        legacy_layer=args.legacy_layer,
        legacy_pixscale=args.legacy_pixscale,
        legacy_size_pix=args.legacy_size_pix,
        legacy_sleep=args.legacy_sleep,
        enable_agn_veto=args.enable_agn_veto,
        subset_h5=None if args.no_subset_h5 else args.subset_h5,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for c in final_survivors:
            f.write(c[0] + "\n")

    args.crossmatch_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.crossmatch_output, "w") as f:
        for c in final_survivors:
            if c[0] in crossmatched_objids:
                f.write(c[0] + "\n")

    final_stage_label = "passed AGN veto (final)" if args.enable_agn_veto else "final (AGN veto skipped)"
    logger.info(
        f"Done. {stage_counts['total']} scanned -> {stage_counts['rate_cut_passed']} passed rate cut -> "
        f"{stage_counts['final_passed']} {final_stage_label} -> {args.output} "
        f"({len(crossmatched_objids)} with ZTF/Rubin crossmatch -> {args.crossmatch_output}). "
        f"Legacy Survey fetch failures: {n_legacy_failed}."
    )
    for c in final_survivors[:20]:
        objid, score, rate, direction, significance, band_rows, row_a, row_b, rows = c
        logger.info(
            f"  {objid}: {direction} {rate:.3f} mag/day, {significance:.1f}sigma "
            f"({row_a['band']}/{row_a['survey']} -> {row_b['band']}/{row_b['survey']}, "
            f"dt={row_b['jd'] - row_a['jd']:.2f}d)"
            f"{'  [XM]' if objid in crossmatched_objids else ''}"
        )


if __name__ == "__main__":
    main()
