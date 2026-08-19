#!/bin/bash
#SBATCH -J tidez_afterglow_hunt_full
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH -A phy250012p
#SBATCH -t 04:00:00
#SBATCH -o ../logs/tidez_afterglow_hunt_full_%j.out
#SBATCH -e ../logs/tidez_afterglow_hunt_full_%j.err

# OPTIONAL / standalone: re-applies the AGN-veto filter across the WHOLE
# H5 using whatever crossmatch data already exists (silently passing
# objects that haven't been crossmatched yet -- see module docstring).
# Not needed for the normal pipeline -- 04_find_real_fast_transients.sh
# already runs an equivalent filter as part of its staged approach. Useful
# if you've already crossmatched everything (e.g. via 05/06_*_full.sh) and
# just want to re-run the filter with different thresholds without
# re-fetching anything. Requires 03_crossmatch_boom.sh to have run (for
# the crossmatched-output split); see tidez/afterglow_hunt.py's module
# docstring for the full cut list.

source /ocean/projects/phy250012p/salgundi/tidez/.venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

time uv run python -m tidez.afterglow_hunt \
    --h5 /ocean/projects/phy250012p/salgundi/tidez/data/tidez_master.h5 \
    --rise-threshold 0.5 \
    --fade-threshold 0.2
