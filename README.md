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
  LIM_MAG5. Epochs without a filter, or with LIM_MAG5 outside 19–26, are dropped: those
  are failed subtractions (5σ depths of mag 0–17 and "detections" at mag ~5).
- ZTF/LSST (step 3): alerts are detections, non-detections are upper limits, forced
  photometry counts as a detection at SNR ≥ 5 with positive flux.

### Fast-transient filter (step 5)

All objects are searched (`--prefix T` keeps only objids starting with T). The log
prints a funnel of how many objects survive each cut:

1. ≥ 2 detections in any survey (`--min-detections`) — a single detection never passes
2. not a star — no Gaia match within 1.5″ with ≥ 5σ parallax or proper motion
3. not an AGN/quasar — no Milliquas match within 1.5″
4. age (first → last detection, all surveys) < 50 d
5. not stochastic — the light curve doesn't keep changing direction
6. rise ≥ 0.7 and fade ≥ 0.15 mag/day in some band

Rates are measured per band (survey + filter) between each epoch and the first epoch
≥ 1 day later, requiring Δmag ≥ 0.5 and ≥ 4σ. Upper limit → detection gives a lower
bound on the rise, detection → upper limit a lower bound on the fade. Candidates are
ranked by the slower of the two. All thresholds are options:

```
sbatch 05_find_fast_transients.sh --rise-threshold 1.0 --require either
sbatch 05_find_fast_transients.sh --method peak --filters i z y
sbatch 05_find_fast_transients.sh --prefix T
```

The output keeps, for the rise and the fade, the stretch of light curve that gave the
rate (`rise_start_mjd` → `rise_end_mjd`, same for fade); the scanner shades it.

**Benchmark.** `T202502161025195m274154` is a known fast transient that every version of
the filter must keep. Step 5 ends its log with `BENCHMARK PASSED`, or
`BENCHMARK TEST FAILED` with the cuts it failed and its numbers — then the filter needs
changing before the candidate list is trusted:

```
grep -A1 BENCHMARK logs/tidez_find_fast_transients_<jobid>.err
```

## Running on Bridges-2

`scripts/run_pipeline.sh` submits every step at once, each as its own job chained with
`--dependency=afterok` on the steps it needs (2, 3, 4a wait for 1; 5 waits for 2, 3, 4a,
4b; 6 waits for 5). Independent steps run in parallel, and a failed step cancels the ones
after it. Step 6 is always run with `--fresh`, so the bundle is rebuilt from the new list.

```
git pull
export BOOM_USERNAME=... BOOM_PASSWORD=...                # steps 3, 4a
export TNS_ANIBOT_API=... TNS_ANIBOT_USER_AGENT=...       # step 4b
bash scripts/run_pipeline.sh                    # everything -> data/fast_transients.h5
bash scripts/run_pipeline.sh --from 5           # only steps 5 and 6
bash scripts/run_pipeline.sh --steps "2 5 6"    # an explicit list
bash scripts/run_pipeline.sh --from 5 -- --prefix T --rise-threshold 1.0   # after -- : step-5 options
squeue -u $USER
```

### What to re-run after a change

Only re-run a step when its inputs or its code changed; everything downstream of it must
follow. Steps you leave out reuse their files in `data/`.

| What changed | Run | Skip |
|---|---|---|
| New DECam data on disk | `--steps "1 2 3 4a 5 6"` (step 1 is incremental) | 4b, unless you also want a newer TNS dump |
| DECam detection / epoch rules (`build_lcs.py`) | `--steps "2 5 6"` | 1, 3, 4a, 4b |
| ZTF/LSST light-curve rules (`get_boom_lcs.py`) | `--steps "3 5 6"` | 1, 2, 4a, 4b |
| Catalog crossmatch radius / catalogs (`enrich.py`) | `--steps "4a 5 6"` | 1, 2, 3, 4b |
| Star / AGN / variable / TNS rules (`basic_filtering.py`) | `--from 5` | 1–4 |
| Refresh the TNS dump | `--steps "4b 5 6"` | 1, 2, 3, 4a |
| Filter cuts or thresholds (`find_fast_transients.py`, `multisurvey_photstats.py`) | `--from 5` | 1–4 |
| What goes into the bundle (`scanning_setup.py`) | `--steps 6` | 1–5 |
| Scanner page or webapp only (`webapp.py`, `utils/scanner.html`, `plot_lightcurve.py`) | nothing — restart the webapp | everything |

Steps 3 (BOOM light curves, ~12 h) and 4a (catalog crossmatches) are the slow,
network-bound ones and rarely need re-running. Each script can also be submitted on its
own from `scripts/` (`sbatch 05_find_fast_transients.sh --prefix T`);
`06_scanning_setup.sh --retry-legacy` re-fetches failed Legacy Survey images.

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
- Plotly light curve of DECam + ZTF + LSST, coloured by filter (u violet, g green,
  r red, i orange, z white, y teal; DECam ◆, ZTF ●, LSST ▲, upper limits ▼). The
  stretches that gave the filter's rise and fade are shaded, labelled with the rate,
  band and pair type (a `≥` rate is a bound from an upper limit). Clicking a DECam
  detection shows its template / science / difference stamps (stored for detections only).
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
