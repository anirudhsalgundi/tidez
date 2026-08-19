#!/bin/bash
#SBATCH -J tidez_get_data
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH -A phy250012p
#SBATCH -t 06:00:00
#SBATCH -o ../logs/tidez_get_data_%j.out
#SBATCH -e ../logs/tidez_get_data_%j.err

# Discovers raw DESIRT candidate FITS files on the shared filesystem and
# writes the file-list inputs used by build_h5.py:
#   data/all_candidate_fits_files.txt
#   data/max_detections_candidate_fits_files.txt  (one file per observation,
#                                                    keeping the max-detections FITS)

source /ocean/projects/phy250012p/salgundi/tidez/.venv/bin/activate

time uv run python -m tidez.get_data
