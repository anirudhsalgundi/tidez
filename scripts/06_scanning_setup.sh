#!/bin/bash
#SBATCH -J tidez_make_scan_bundle
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH -A phy250012p
#SBATCH -t 06:00:00
#SBATCH -o ../logs/tidez_make_scan_bundle_%j.out
#SBATCH -e ../logs/tidez_make_scan_bundle_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anirudhs@unc.edu

# Step 6: builds data/scan_bundle.h5 for the candidates in data/fast_transients.csv
# (step 5): their DECam light curves, detection-epoch cutouts (float16), ZTF/LSST light
# curves from data/boom_lc.parquet, a Legacy Survey cutout each (needs internet), and the
# filter's numbers + TNS match. Copy it to a laptop and scan it with
#   python -m tidez.webapp --h5 data/scan_bundle.h5
# Resumable: re-submitting skips objects already in the bundle. Extra options are passed
# through, e.g. sbatch 06_make_scan_bundle.sh --retry-legacy

# Call the conda env's interpreter directly: `conda activate` doesn't work in a
# non-interactive batch shell.
PYTHON=/jet/home/salgundi/envs/tidez/bin/python
PROJECT_DIR=/ocean/projects/phy250012p/salgundi/tidez

# make_scan_bundle.py does `from tidez... import ...`; PYTHONPATH lets that resolve
# when the script is run by path.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_DIR"
time "$PYTHON" "$PROJECT_DIR/tidez/scanning_setup.py" "$@"
