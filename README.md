# tidez

Pipeline for building, cross-matching, filtering, and visually scanning
DECam transient candidates from the 3DTS/DESIRT search, combined with ZTF
and LSST (via BOOM), Milliquas, and Legacy Survey data.

## Pipeline overview

```
get_data.py               -> list candidate FITS files on disk
        |
        v
build_h5.py                -> tidez_master.h5   (one group per object:
        |                                         lightcurve + cutouts)
        |
        v
crossmatch_boom.py         -> adds ZTF/LSST crossmatch lightcurves (full dataset)
        |
        v
find_real_fast_transients.py
        -> Stage 1: rate cut (cheap, no crossmatch needed) on the FULL dataset
        -> Stage 2: Milliquas crossmatch, only for rate-cut survivors
        -> Stage 3: AGN veto (Milliquas match + stochastic-variability check)
        -> Stage 4: Legacy Survey crossmatch, only for final survivors
        -> fast_real_transients.txt / fast_real_transients_crossmatched.txt
        |
        v
webapp.py / lightcurve_viewer.py    -> interactive scanning UI
```

The crossmatching cost scales with object count, but the filter itself
doesn't need any crossmatch data to run (`find_best_rate_pair` in
`transient_filters.py` works off DECam+ZTF+LSST lightcurve data alone) --
so `find_real_fast_transients.py` runs the rate cut across the *entire*
dataset first, and only pays for the (much slower, network-bound) Milliquas
and Legacy Survey crossmatches on whatever small fraction of objects
actually survives it. At tens of thousands of objects this is the
difference between crossmatching everything vs. crossmatching a few
hundred candidates.

`find_fast_transients.py`, `transient_hunt.py`, and `afterglow_hunt.py`
(progressively stricter standalone filters) and the standalone
`crossmatch_milliquas.py` / `crossmatch_legacy_survey.py` scripts still
exist and work the same as before, for cases where you want crossmatch
data for the *whole* dataset rather than just candidates, or want to
iterate on filter thresholds without re-deriving the staged candidate set.
See `scripts/05_crossmatch_legacy_survey_full.sh`,
`scripts/06_crossmatch_milliquas_full.sh`, and
`scripts/07_afterglow_hunt_full.sh`.

Everything lives in one HDF5 file, `data/tidez_master.h5`. Every script that
touches it is resumable: re-running just processes objects that don't yet
have the relevant group, so it's safe to append new candidates or re-run a
crossmatch pass at any time without redoing existing work.

### `tidez_master.h5` layout

```
/catalog                          compound dataset: objid, ra, dec, n_epochs

/<objid>/                         attrs: objid, ra, dec
/<objid>/lightcurve                DECam forced-photometry epochs (all
                                    STATUS_FPHOT values kept: 'p' good,
                                    'q' marginal, 'm' non-detection)
/<objid>/cutouts/{template,science,diff}   (n_epochs, 121, 121) float32 stamps

/<objid>/crossmatch/               attrs: radius_arcsec           [crossmatch_boom.py]
/<objid>/crossmatch/ztf/           attrs: objectId, separation_arcsec, n_alerts
/<objid>/crossmatch/ztf/lightcurve
/<objid>/crossmatch/lsst/          (same shape as ztf, if matched)

/<objid>/milliquas/                attrs: matched (bool), and if matched,   [crossmatch_milliquas.py]
                                    separation_arcsec + whatever columns
                                    the Milliquas VizieR table returned
                                    (Name, Type, z, Qpct, ...)

/<objid>/legacy_survey/            attrs: layer, pixscale, size_pix         [crossmatch_legacy_survey.py]
/<objid>/legacy_survey/cutout_jpg   raw JPEG bytes (uint8 array)
```

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/); every
script below is run as `uv run python -m tidez.<script>` (or
`uv run tidez/<script>.py`), which resolves the environment automatically.

BOOM crossmatching needs credentials in the environment:

```
export BOOM_USERNAME=...
export BOOM_PASSWORD=...
```

## 1. Building the master H5

```
uv run python -m tidez.build_h5 --input-list data/candidate_fits_files.txt
```

- `--input-list` — a text file of candidate FITS paths, one per line (use
  this on the remote/cluster; see `get_data.py` for how these lists get
  generated from the raw DESIRT search output).
- `--input-dir` — alternatively, glob a local directory of `*.fits` files.
- `--output` — defaults to `data/tidez_master.h5`.

Re-running with a new/updated input list **appends** — it skips any objid
already present in the H5, so you can add new candidates over time without
rebuilding from scratch.

## 2. ZTF/LSST crossmatch

Run this against the master H5 after `build_h5.py`, on the full dataset —
`find_real_fast_transients.py` (step 3 below) relies on it already being
there for its crossmatched-output split.

```
uv run python -m tidez.crossmatch_boom --radius-arcsec 5
```
Batches objects into one cone-search request per `BATCH_SIZE` (500)
objects rather than one request per object.

Resumability is tracked via a small checkpoint text file
(`data/boom_progress.txt` — override with `--progress-file`), not by
re-scanning the whole H5 every run. At tens of thousands of objects,
checking "is this done?" against the H5 directly means one HDF5 metadata
lookup per object, which gets slow on networked/parallel filesystems (e.g.
Bridges2's Ocean) — the checkpoint file makes a re-run's resumability
check a single fast file read instead. If it's missing (first run, or
deleted), the script falls back to scanning the H5 once (with a progress
bar) and rebuilds it. Same pattern used by `crossmatch_milliquas.py` and
`crossmatch_legacy_survey.py` below.

## 3. Finding real fast transients

`find_real_fast_transients.py` is the recommended way to go from a master
H5 to a candidate list — see the pipeline diagram above for why it stages
the crossmatching after the rate cut instead of before it.

```
uv run python -m tidez.find_real_fast_transients --rise-threshold 0.5 --fade-threshold 0.2
```

Its four stages (implemented in `tidez/transient_filters.py`,
`crossmatch_milliquas.py`, and `crossmatch_legacy_survey.py`):

| Stage | What it does |
|---|---|
| 1. Rate cut | Quality-gated, **asymmetric** rise/fade rate cut on every object's DECam(`'p'`-only)+ZTF+LSST lightcurve: DECam `'p'`-only detections, SNR floor, minimum baseline, minimum amplitude, statistical significance, anti-blip check, minimum detection count, then rise ≥ 0.5 mag/day / fade ≥ 0.2 mag/day by default (real fast transients typically rise faster than they fade). No crossmatch data needed — this is what lets the rest of the pipeline skip crossmatching most of the dataset. |
| 2. Milliquas crossmatch | Only for stage-1 survivors lacking a `/milliquas` group already. |
| 3. AGN veto | Reject a Milliquas match, or stochastic-variability signals: reversal fraction (how often consecutive delta-mag flips sign) and persistent-detection fraction (what fraction of epochs *outside* the event window are still detections) — both skip rather than reject when there's too little data to judge. |
| 4. Legacy Survey crossmatch | Only for the *final* survivors (post-AGN-veto) lacking a `/legacy_survey` group already. |

Run `--help` for the full list of tunable cuts (baseline, significance,
blip window, stochasticity thresholds, Milliquas/Legacy Survey params,
etc.) — all documented in the module docstring along with the reasoning
behind each cut.

It logs a per-stage breakdown (passed rate cut / rejected by Milliquas /
rejected by stochasticity / final passed) so you can see where the sample
is actually shrinking, not just the final count. Outputs:
`data/fast_real_transients.txt` and
`data/fast_real_transients_crossmatched.txt` (the subset with a ZTF/Rubin
crossmatch).

### Standalone filters / full-dataset crossmatch (optional)

These pre-date `find_real_fast_transients.py` and still work the same way
— useful if you want crossmatch data for the *whole* dataset (not just
candidates), or want to iterate on filter thresholds without re-deriving
the staged candidate set:

| Script | What it does |
|---|---|
| `find_fast_transients.py` | Basic filter: steepest \|delta mag\|/delta t within any single band, same-visit duplicate guard. No quality/AGN cuts. |
| `transient_hunt.py` | Adds `find_real_fast_transients.py`'s stage-1 quality cuts (artifact rejection) but with a single symmetric rate threshold, no AGN vetoing. |
| `afterglow_hunt.py` | The same filter as `find_real_fast_transients.py`'s stages 1+2+3, but assumes Milliquas has already been crossmatched for the *whole* H5 (see `scripts/06_crossmatch_milliquas_full.sh`) rather than staging it — use this if you already have full-dataset Milliquas data and just want to re-filter with different thresholds. |
| `crossmatch_milliquas.py` / `crossmatch_legacy_survey.py` | Crossmatch every object in the H5, not just candidates (`--radius-arcsec 3` / `--layer ls-dr10`). |

```
uv run python -m tidez.find_fast_transients --threshold 0.1
uv run python -m tidez.transient_hunt --threshold 0.1
uv run python -m tidez.afterglow_hunt --rise-threshold 0.5 --fade-threshold 0.2
```

## 4. Scanning candidates

### Web scanner (recommended)

```
uv run python -m tidez.webapp --port 8000 [--plot-all] [--host 0.0.0.0]
```
Then open `http://<host>:8000`. Features:
- Prev/Next through the object list, or type a candidates `.txt` filename
  (e.g. `afterglow_candidates.txt`) into the filter box and click Apply to
  restrict browsing to just that list; typing a bare objid jumps straight
  to it.
- Interactive Plotly lightcurve (DECam + ZTF + LSST), with a
  zoom-adaptive top/bottom axis pair ("Days ago" / "Days since first
  detection") that recomputes tick spacing and precision as you zoom.
- Click a DECam point to load its template/science/diff cutout stamps.
- Legacy Survey cutout panel next to the plot, with an "Open in Legacy
  Survey ↗" button that opens the full viewer at that object's RA/Dec.
- Copyable DECam/ZTF/Rubin object IDs and RA/Dec (comma-separated) in the
  header bar.
- Favorite button, writing to `data/favorite_sources.txt`.
- By default only `STATUS_FPHOT == 'p'` (good subtraction quality) DECam
  points are plotted; pass `--plot-all` to include `'q'`/`'m'` epochs too.

**Running on a remote cluster (e.g. Bridges2):** start the webapp on the
compute/login node, then tunnel a local port to it from your laptop:
```
ssh -L 8000:localhost:8000 <user>@<cluster-login-node>
```
(if `webapp.py` is running on a compute node reached via the login node,
target the compute node's hostname instead of `localhost` in the `-L`
mapping, and make sure the webapp itself binds `--host 0.0.0.0` so it's
reachable from off-node.) Then open `http://localhost:8000` locally.

### Matplotlib slide-deck viewer

```
uv run python -m tidez.lightcurve_viewer --objids-file data/afterglow_candidates.txt
```
Prev/Next through objects one at a time; double-click a series/legend
entry to isolate it, double-click empty space to reset; same favoriting
behavior as the webapp.

### Single-object plot

```
uv run python -m tidez.plot_lightcurve <objid> [--save output.png]
```

## Other utilities

- `query_h5.py` — read-side helpers: look up a single object's lightcurve
  and cutouts by objid, or cone-search the catalog table by RA/Dec.
- `cutouts.py` / `legacy_cutout.py` — render DECam / Legacy Survey stamps
  as base64 PNGs (used internally by the webapp).
- `get_data.py` — discovers raw DESIRT candidate FITS files on the shared
  filesystem and writes the file-list inputs used by `build_h5.py`.
- `transient_filters.py` — shared rate-cut / stochastic-variability-veto
  logic used by both `find_real_fast_transients.py` and `afterglow_hunt.py`,
  so the two definitions of "passes the filter" can't drift apart.
- `progress.py` — shared checkpoint-file resumability logic used by
  `crossmatch_boom.py`, `crossmatch_milliquas.py`, and
  `crossmatch_legacy_survey.py` (see "ZTF/LSST crossmatch" above).
- `logger.py` — shared colored console + file logger (writes to `logs/`),
  used by every script above.

## Batch scripts (Bridges2 / SLURM)

`scripts/` has one `sbatch` script per pipeline stage, numbered in
run order:

| Script | Stage |
|---|---|
| `01_get_data.sh` | `get_data.py` |
| `02_build_h5.sh` | `build_h5.py` |
| `03_crossmatch_boom.sh` | `crossmatch_boom.py` (full dataset) |
| `04_find_real_fast_transients.sh` | `find_real_fast_transients.py` (primary candidate-finding step) |
| `05_crossmatch_legacy_survey_full.sh` | optional: `crossmatch_legacy_survey.py` on the full dataset |
| `06_crossmatch_milliquas_full.sh` | optional: `crossmatch_milliquas.py` on the full dataset |
| `07_afterglow_hunt_full.sh` | optional: `afterglow_hunt.py` re-filter, assumes 05/06 already ran |

All of them set `export HDF5_USE_FILE_LOCKING=FALSE` (except
`01_get_data.sh`, which never touches the H5) — needed on Bridges2's Ocean
filesystem, where HDF5's default file-locking check can hang or fail even
without a genuine concurrent writer. Every stage is resumable, so any of
these are safe to `scancel` and resubmit.
