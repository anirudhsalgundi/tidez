#!/bin/bash
#SBATCH -J tidez_crossmatch_legacy_survey
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH -A phy250012p
#SBATCH -t 30:00:00
#SBATCH -o ../logs/tidez_crossmatch_legacy_survey_%j.out
#SBATCH -e ../logs/tidez_crossmatch_legacy_survey_%j.err

# Fetches and stores a Legacy Survey DR10 cutout image for every object in
# the master H5 (one HTTPS request per object). Resumable via
# data/legacy_survey_progress.txt.

source /ocean/projects/phy250012p/salgundi/tidez/.venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

time uv run python -m tidez.crossmatch_legacy_survey \
    --h5 /ocean/projects/phy250012p/salgundi/tidez/data/tidez_master.h5 \
    --layer ls-dr10
