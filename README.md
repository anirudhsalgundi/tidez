# tidez - Transients in DECam and ZTF

An archival search for Fast-transient search on the DECam (DESIRT + DECam–ZTF experiment) candidates,
combined with ZTF and LSST light curves and catalog crossmatches from BOOM.
Everything up to step 6 runs on Bridges-2; scanning runs on a laptop.

## Pipeline

| Step | SLURM script (`scripts/`) | Python (`tidez/`) | Writes (`data/`) |
|---|---|---|---|
| 1 | `01_create_masterfile.sh` | `create_masterdb.py` | `all_objects.csv` — objid, program, file_path, ra, dec |
| 2 | `02_build_lcs.sh` | `build_lcs.py` | `lightcurves.parquet` (status `p` required), `lightcurves_any_status.parquet` |
| 3 | `03_get_boom_lc.sh` | `get_boom_lcs.py` | `boom_lc.parquet` — ZTF/LSST alerts, non-detections, forced photometry |
| 4a | `04a_enrich.sh` | `enrich.py` | `crossmatches/<catalog>.parquet` — every BOOM catalog match within 5″ |
| 4b | `04b_download_tns_objects.sh` | `download_tns_objects.py` | `tns_public_objects.csv` |
| 5 | `05_find_fast_transients.sh` | `find_fast_transients.py` | `fast_transients.csv`, `.txt`, `_stats.parquet` |
| 6 | `06_scanning_setup.sh` | `scanning_setup.py` | `fast_transients.h5` — the scan bundle |
| scan | — (laptop) | `webapp.py` + `utils/scanner.html` | `favorite_sources.txt`, `tns_to_report.txt` |

Libraries used by the steps: `basic_filtering.py` (rock / star / variable / AGN /
on-TNS flags from `crossmatches/` and the TNS dump), `multisurvey_photstats.py`
(per-band rise/fade rates and variability), `cutouts.py` and `legacy_cutout.py`
(reading stamps and Legacy Survey images from the bundle),
`crossmatch_legacy_survey.py` (fetching Legacy Survey images), `plot_lightcurve.py`
(light-curve rows and band colours).

Steps 2, 3, 4a and 4b only need step 1; step 5 needs 2–4; step 6 needs 1, 3 and 5.
Steps 3, 4a and 6 are resumable: re-submitting the job continues where it stopped.

### Detection rules

- DECam (step 2): SNR > 5, CNN score > 0.7, mag ≤ LIM_MAG5, and (in
  `lightcurves.parquet`) STATUS_FPHOT == `p`. Every other epoch is an upper limit at
  LIM_MAG5; epochs without a filter or LIM_MAG5 are dropped.
- ZTF/LSST (step 3): alerts are detections, non-detections are upper limits, forced
  photometry counts as a detection at SNR ≥ 5 with positive flux.

### Fast-transient filter (step 5)

The log prints a funnel of how many objects survive each cut:

1. not a rock — at least 2 detections, unless an extended Legacy Survey source is within 2″
2. not a star — no Gaia match within 1.5″ with ≥ 5σ parallax or proper motion
3. not an AGN/quasar — no Milliquas match within 1.5″
4. age (first → last detection, all surveys) < 50 d
5. not stochastic — the light curve doesn't keep changing direction
6. rise ≥ 1 and fade ≥ 0.25 mag/day in some band

Rates are measured per band (survey + filter) between each epoch and the first epoch
≥ 1 day later, requiring Δmag ≥ 0.5 and ≥ 4σ. Upper limit → detection gives a lower
bound on the rise, detection → upper limit a lower bound on the fade. Candidates are
ranked by the slower of the two. All thresholds are options:

```
sbatch 05_find_fast_transients.sh --rise-threshold 1.0 --require either
sbatch 05_find_fast_transients.sh --method peak --filters i z y
```

## Running on Bridges-2

The scripts call the env's Python directly (`/jet/home/salgundi/envs/tidez/bin/python`)
and run from `scripts/`, writing logs to `logs/`:

```
cd scripts
sbatch 01_create_masterfile.sh
sbatch 02_build_lcs.sh
export BOOM_USERNAME=... BOOM_PASSWORD=...
sbatch 03_get_boom_lc.sh
sbatch 04a_enrich.sh
export TNS_ANIBOT_API=... TNS_ANIBOT_USER_AGENT=...
sbatch 04b_download_tns_objects.sh
sbatch 05_find_fast_transients.sh
sbatch 06_scanning_setup.sh            # --retry-legacy re-fetches failed Legacy Survey images
```

Install into a fresh env with `pip install -e .` (Python ≥ 3.12).

## Scanning (laptop)

Copy the bundle (≈ 1 MB per candidate) and start the scanner:

```
scp <bridges2>:/ocean/projects/phy250012p/salgundi/tidez/data/fast_transients.h5 data/
python -m tidez.webapp                 # http://127.0.0.1:8000, reads data/fast_transients.h5
```

The page shows the candidates in the step-5 ranking:

- Prev/Next, or type an objids `.txt` file (relative to `data/`) or a bare objid and
  click Apply.
- Plotly light curve of DECam + ZTF + LSST; clicking a DECam detection shows its
  template / science / difference stamps (stamps are stored for detections only).
- Legacy Survey image, with a link to the full viewer.
- DECam, program, ZTF, Rubin and RA/Dec chips — click to copy.
- **TNS ↗** opens the matched TNS page; the badge says whether our DECam objid is in
  the TNS internal names ("DECam on TNS"), the object is on TNS without our data
  ("DECam not on TNS"), or there is no TNS match ("Not on TNS").
- **☆ Favorite** and **⚑ To report** toggle the object in `data/favorite_sources.txt`
  and `data/tns_to_report.txt` (one objid per line).

Options: `--h5`, `--favorites-file`, `--report-file`, `--port`, `--host`, and
`--plot-all` (also count STATUS_FPHOT `q` epochs as detections).

## Scan bundle layout (`fast_transients.h5`)

```
/<objid>/                         attrs: objid, ra, dec, program, decam_on_tns, TNS match,
                                  and every column of fast_transients.csv
/<objid>/lightcurve               DECam epochs: MJD_OBS, FILTER, MAG_FPHOT, MAGERR_FPHOT,
                                  SNR_FPHOT, STATUS_FPHOT, LIM_MAG3, LIM_MAG5, CNNSCORE, DETECTED
/<objid>/cutouts/{template,science,diff}
                                  float16 (n, 121, 121) stamps of detection epochs;
                                  pixel value = stamp * <key>_scale
/<objid>/cutouts/epoch_idx        lightcurve row of each stamp
/<objid>/crossmatch/{ztf,lsst}/   ZTF / LSST light curves from boom_lc.parquet
/<objid>/legacy_survey/           Legacy Survey DR10 JPEG
/scan_order                       candidate objids, fastest first
```
