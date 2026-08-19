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
        +--> crossmatch_boom.py             -> adds ZTF/LSST crossmatch lightcurves
        +--> crossmatch_milliquas.py        -> adds Milliquas AGN/quasar match
        +--> crossmatch_legacy_survey.py    -> adds a Legacy Survey cutout image
        |
        v
find_fast_transients.py / transient_hunt.py / afterglow_hunt.py
        -> candidate objid text files (progressively stricter filters)
        |
        v
webapp.py / lightcurve_viewer.py    -> interactive scanning UI
```

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

## 2. Cross-matching

Run these against the master H5 after `build_h5.py`. Each one only
processes objects that don't already have its group, so re-running after
appending new objects to the H5 is cheap — only the new ones get queried.

Resumability is tracked via a small checkpoint text file per script
(`data/boom_progress.txt`, `data/milliquas_progress.txt`,
`data/legacy_survey_progress.txt` — override with `--progress-file`), not
by re-scanning the whole H5 every run. At tens of thousands of objects,
checking "is this done?" against the H5 directly means one HDF5 metadata
lookup per object, which gets slow on networked/parallel filesystems (e.g.
Bridges2's Ocean) — the checkpoint file makes a re-run's resumability
check a single fast file read instead. If a checkpoint file is missing
(first run, or deleted), the script falls back to scanning the H5 once
(with a progress bar) and rebuilds it.

**ZTF / LSST (via BOOM alert broker)**
```
uv run python -m tidez.crossmatch_boom --radius-arcsec 5
```
Batches objects into one cone-search request per `BATCH_SIZE` (500)
objects rather than one request per object.

**Milliquas (AGN/quasar catalog, via CDS X-Match)**
```
uv run python -m tidez.crossmatch_milliquas --radius-arcsec 3
```
Uploads the whole object table in chunks and matches it against the
Milliquas VizieR catalog in one request per chunk (bulk, not per-object —
this is what keeps it fast at catalog scale).

**Legacy Survey cutout image**
```
uv run python -m tidez.crossmatch_legacy_survey --layer ls-dr10
```
Fetches one DR10 cutout JPEG per object and stores it directly (rendered
with a crosshair overlay on demand by the webapp, not at fetch time).

## 3. Finding candidates

Three filters of increasing strictness, each writing two output lists: all
qualifying objids, and the subset that also has a ZTF/LSST crossmatch.

| Script | What it does |
|---|---|
| `find_fast_transients.py` | Basic filter: steepest \|delta mag\|/delta t within any single band (DECam+ZTF+LSST pooled), same-visit duplicate guard. |
| `transient_hunt.py` | Adds quality cuts meant to reject subtraction/photometry artifacts: DECam `'p'`-only, SNR floor, minimum amplitude, statistical significance, anti-blip check, minimum detection count. |
| `afterglow_hunt.py` | Adds two AGN-rejection stages on top of `transient_hunt.py`'s cuts, aimed at orphan afterglows/TDEs/GRB afterglows/FBOTs: a Milliquas veto, a stochastic-variability veto (reversal fraction + persistent-detection fraction), and an **asymmetric** rise/fade rate cut (rise ≥ 0.5 mag/day, fade ≥ 0.2 mag/day by default — real fast transients typically rise faster than they fade). |

```
uv run python -m tidez.find_fast_transients --threshold 0.1
uv run python -m tidez.transient_hunt --threshold 0.1
uv run python -m tidez.afterglow_hunt --rise-threshold 0.5 --fade-threshold 0.2
```

Run `--help` on any of them for the full list of tunable cuts (baseline,
significance, blip window, stochasticity thresholds, etc.) — all documented
in each script's module docstring along with the reasoning behind each cut.

`afterglow_hunt.py` logs a per-stage breakdown (rejected by Milliquas /
rejected by stochasticity / no qualifying rate pair / passed) so you can see
where the sample is actually shrinking, not just the final count.

Outputs land in `data/`, e.g. `afterglow_candidates.txt` and
`afterglow_candidates_crossmatched.txt`.

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
- `logger.py` — shared colored console + file logger (writes to `logs/`),
  used by every script above.
