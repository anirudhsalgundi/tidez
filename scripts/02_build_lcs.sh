#!/bin/bash
#SBATCH -J tidez_build_lcs
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16000M
#SBATCH -A phy250012p
#SBATCH -t 08:00:00
#SBATCH -o ../logs/tidez_build_lcs_%j.out
#SBATCH -e ../logs/tidez_build_lcs_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anirudhs@unc.edu

# Step 2: reads every FITS file listed in data/all_objects.csv (from step 1) and
# writes two light-curve Parquet files in data/:
#   lightcurves.parquet             detections need STATUS_FPHOT == 'p'
#   lightcurves_any_status.parquet  same cuts, status ignored
# Detection = SNR > 5, CNN > 0.7, mag <= limmag5; every other epoch is an upper
# limit. Only light-curve columns are read (no thumbnails). Workers default to
# SLURM_CPUS_PER_TASK.

# Call the conda env's interpreter directly: `conda activate` doesn't work in a
# non-interactive batch shell.
PYTHON=/jet/home/salgundi/envs/tidez/bin/python
PROJECT_DIR=/ocean/projects/phy250012p/salgundi/tidez

# build_lcs.py does `from tidez.logger import ...`; PYTHONPATH lets that resolve
# when the script is run by path.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_DIR"
time "$PYTHON" "$PROJECT_DIR/tidez/build_lcs.py"