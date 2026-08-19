#!/bin/bash
#SBATCH -J tidez_crossmatch_legacy_survey_full
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH -A phy250012p
#SBATCH -t 30:00:00
#SBATCH -o ../logs/tidez_crossmatch_legacy_survey_full_%j.out
#SBATCH -e ../logs/tidez_crossmatch_legacy_survey_full_%j.err

# OPTIONAL / standalone: fetches a Legacy Survey DR10 cutout for EVERY
# object in the master H5, not just candidates. Not needed for the normal
# pipeline -- 04_find_real_fast_transients.sh already fetches Legacy
# Survey cutouts, but only for its final candidate list, which is far
# cheaper at ~60k-object scale. Only run this if you specifically want
# cutouts for the full dataset (e.g. for some other analysis). Resumable
# via data/legacy_survey_progress.txt.

source /ocean/projects/phy250012p/salgundi/tidez/.venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

time uv run python -m tidez.crossmatch_legacy_survey \
    --h5 /ocean/projects/phy250012p/salgundi/tidez/data/tidez_master.h5 \
    --layer ls-dr10
