#!/bin/bash
#SBATCH -J tidez_afterglow_hunt
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH -A phy250012p
#SBATCH -t 04:00:00
#SBATCH -o ../logs/tidez_afterglow_hunt_%j.out
#SBATCH -e ../logs/tidez_afterglow_hunt_%j.err

# Runs the full afterglow/TDE/FBOT filter: Milliquas veto -> stochastic-
# variability veto -> asymmetric rise/fade rate cut. Requires
# 03_crossmatch_boom.sh (for the crossmatched-output split) and
# 05_crossmatch_milliquas.sh (for the Milliquas veto) to have already run;
# see tidez/afterglow_hunt.py's module docstring for the full cut list.
# CPU-only, no network calls -- doesn't need the compute allocation this
# job requests, but kept as a batch job for consistency with the rest of
# the pipeline and to avoid tying up the login node on a large H5.

source /ocean/projects/phy250012p/salgundi/tidez/.venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

time uv run python -m tidez.afterglow_hunt \
    --h5 /ocean/projects/phy250012p/salgundi/tidez/data/tidez_master.h5 \
    --rise-threshold 0.5 \
    --fade-threshold 0.2
