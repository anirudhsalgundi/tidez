#!/bin/bash
#SBATCH -J tidez_get_boom_lcs
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000M
#SBATCH -A phy250012p
#SBATCH -t 12:00:00
#SBATCH -o ../logs/tidez_get_boom_lcs_%j.out
#SBATCH -e ../logs/tidez_get_boom_lcs_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anirudhs@unc.edu

# Step 3: cone-searches BOOM's ZTF_alerts_aux / LSST_alerts_aux for every object in
# data/all_objects.csv and writes the matched objects' alerts, non-detections and
# forced photometry to data/boom_lc.parquet. Network-bound: the 16 CPUs are only there
# because RM-shared hands out memory per core (~2 GB each), and dense ZTF fields need it.
# Resumable: finished batches are kept in data/boom_lc_parts/ and skipped when the
# job is re-submitted; the final file is written once every object is done.
#
# Needs BOOM_USERNAME and BOOM_PASSWORD exported in the shell you run sbatch from
# (sbatch passes your environment on to the job).

if [ -z "$BOOM_USERNAME" ] || [ -z "$BOOM_PASSWORD" ]; then
    echo "BOOM_USERNAME / BOOM_PASSWORD are not set; export them before sbatch." >&2
    exit 1
fi

# Call the conda env's interpreter directly: `conda activate` doesn't work in a
# non-interactive batch shell.
PYTHON=/jet/home/salgundi/envs/tidez/bin/python
PROJECT_DIR=/ocean/projects/phy250012p/salgundi/tidez

# get_boom_lcs.py does `from tidez.logger import ...`; PYTHONPATH lets that resolve
# when the script is run by path.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_DIR"
time "$PYTHON" "$PROJECT_DIR/tidez/get_boom_lcs.py"