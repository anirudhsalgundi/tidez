#!/bin/bash
#SBATCH -J tidez_create_masterfile
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8000M
#SBATCH -A phy250012p
#SBATCH -t 06:00:00
#SBATCH -o ../logs/tidez_create_masterfile_%j.out
#SBATCH -e ../logs/tidez_create_masterfile_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anirudhs@unc.edu

# Step 1: builds data/all_objects.csv, the master index of candidate objects
# (objid, program, file_path, ra, dec; one row per program/objid, keeping the
# max-detections FITS file). Re-runs only re-read headers of objects whose file
# changed; pass --full to re-read every header. Workers default to SLURM_CPUS_PER_TASK.

# Call the conda env's interpreter directly: `conda activate` doesn't work in a
# non-interactive batch shell.
PYTHON=/jet/home/salgundi/envs/tidez/bin/python
PROJECT_DIR=/ocean/projects/phy250012p/salgundi/tidez

# create_masterdb.py does `from tidez.logger import ...`; PYTHONPATH lets that resolve
# when the script is run by path.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_DIR"
time "$PYTHON" "$PROJECT_DIR/tidez/create_masterdb.py"
