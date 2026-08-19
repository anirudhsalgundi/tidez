#!/bin/bash
#SBATCH -J tidez_crossmatch_boom
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH -A phy250012p
#SBATCH -t 30:00:00
#SBATCH -o ../logs/tidez_crossmatch_boom_%j.out
#SBATCH -e ../logs/tidez_crossmatch_boom_%j.err

# Cross-matches every object in the master H5 against ZTF/LSST alerts via
# the BOOM broker. Needs BOOM_USERNAME / BOOM_PASSWORD in the environment
# (e.g. exported here, or already in your shell profile on this cluster).
# Resumable via data/boom_progress.txt.

source /ocean/projects/phy250012p/salgundi/tidez/.venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

time uv run python -m tidez.crossmatch_boom \
    --h5 /ocean/projects/phy250012p/salgundi/tidez/data/tidez_master.h5 \
    --radius-arcsec 5
