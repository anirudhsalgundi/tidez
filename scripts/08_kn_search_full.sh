#!/bin/bash
#SBATCH -J tidez_kn_search
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH -A phy250012p
#SBATCH -t 12:00:00
#SBATCH -o ../logs/tidez_kn_search_%j.out
#SBATCH -e ../logs/tidez_kn_search_%j.err

# Kilonova-oriented candidate-finding pipeline: same shape as
# 04_find_real_fast_transients.sh, but with a 14-day max detection span and
# a rate cut that also fires on upper-limit/detection pairs (catches events
# that crossed the detection threshold entirely between two epochs), not
# just detection/detection pairs. Requires 03_crossmatch_boom.sh to have
# already run on the full H5 (for the ZTF/Rubin crossmatched-output split).
# See tidez/kn_search.py's module docstring for the full 4-stage breakdown.
#
# Writes:
#   data/kn_candidates.txt
#   data/kn_candidates_crossmatched.txt

source /ocean/projects/phy250012p/salgundi/tidez/.venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

time uv run python -m tidez.kn_search \
    --h5 /ocean/projects/phy250012p/salgundi/tidez/data/tidez_master.h5 \
    --rise-threshold 0.5 \
    --fade-threshold 0.2 \
    --ul-rise-threshold 0.5 \
    --ul-fade-threshold 0.2 \
    --max-baseline-days 14
