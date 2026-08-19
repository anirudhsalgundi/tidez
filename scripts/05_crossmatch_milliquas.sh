#!/bin/bash
#SBATCH -J tidez_crossmatch_milliquas
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH -A phy250012p
#SBATCH -t 10:00:00
#SBATCH -o ../logs/tidez_crossmatch_milliquas_%j.out
#SBATCH -e ../logs/tidez_crossmatch_milliquas_%j.err

# Cross-matches every object in the master H5 against the Milliquas
# (AGN/quasar) catalog via a bulk CDS X-Match request per chunk.
# Resumable via data/milliquas_progress.txt.

source /ocean/projects/phy250012p/salgundi/tidez/.venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

time uv run python -m tidez.crossmatch_milliquas \
    --h5 /ocean/projects/phy250012p/salgundi/tidez/data/tidez_master.h5 \
    --radius-arcsec 3
