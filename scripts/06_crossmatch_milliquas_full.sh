#!/bin/bash
#SBATCH -J tidez_crossmatch_milliquas_full
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH -A phy250012p
#SBATCH -t 10:00:00
#SBATCH -o ../logs/tidez_crossmatch_milliquas_full_%j.out
#SBATCH -e ../logs/tidez_crossmatch_milliquas_full_%j.err

# OPTIONAL / standalone: cross-matches EVERY object in the master H5
# against Milliquas, not just candidates. Not needed for the normal
# pipeline -- 04_find_real_fast_transients.sh already crossmatches
# against Milliquas, but only for rate-cut survivors, which is far
# cheaper at ~60k-object scale. Only run this if you specifically want
# Milliquas data for the full dataset. Resumable via
# data/milliquas_progress.txt.

source /ocean/projects/phy250012p/salgundi/tidez/.venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

time uv run python -m tidez.crossmatch_milliquas \
    --h5 /ocean/projects/phy250012p/salgundi/tidez/data/tidez_master.h5 \
    --radius-arcsec 3
