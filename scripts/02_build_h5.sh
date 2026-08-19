#!/bin/bash
#SBATCH -J tidez_build_h5
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH -A phy250012p
#SBATCH -t 12:00:00
#SBATCH -o ../logs/tidez_build_h5_%j.out
#SBATCH -e ../logs/tidez_build_h5_%j.err

# Builds/appends data/tidez_master.h5 from the FITS file list written by
# 01_get_data.sh. Resumable: re-running (e.g. after 01_get_data.sh finds
# new candidates) only ingests objids not already present in the H5.

source /ocean/projects/phy250012p/salgundi/tidez/.venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

time uv run python -m tidez.build_h5 \
    --input-list /ocean/projects/phy250012p/salgundi/tidez/data/max_detections_candidate_fits_files.txt \
    --output /ocean/projects/phy250012p/salgundi/tidez/data/tidez_master.h5
