#!/bin/bash
#SBATCH -J tidez_find_real_fast_transients
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH -A phy250012p
#SBATCH -t 12:00:00
#SBATCH -o ../logs/tidez_find_real_fast_transients_%j.out
#SBATCH -e ../logs/tidez_find_real_fast_transients_%j.err

# Primary candidate-finding pipeline, recommended over running the
# standalone crossmatch_milliquas.py / crossmatch_legacy_survey.py /
# afterglow_hunt.py scripts on the full ~60k-object H5: this runs the rate
# cut FIRST (cheap, no network calls), then only crossmatches Milliquas
# and Legacy Survey for the (much smaller) surviving candidate set.
# Requires 03_crossmatch_boom.sh to have already run on the full H5 (for
# the ZTF/Rubin crossmatched-output split). See
# tidez/find_real_fast_transients.py's module docstring for the full
# 4-stage breakdown.
#
# Writes:
#   data/fast_real_transients.txt
#   data/fast_real_transients_crossmatched.txt

source /ocean/projects/phy250012p/salgundi/tidez/.venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

time uv run python -m tidez.find_real_fast_transients \
    --h5 /ocean/projects/phy250012p/salgundi/tidez/data/tidez_master.h5 \
    --rise-threshold 0.5 \
    --fade-threshold 0.2
