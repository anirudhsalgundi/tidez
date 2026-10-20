#!/bin/bash
#SBATCH -J tidez_enrich
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH -A phy250012p
#SBATCH -t 06:00:00
#SBATCH -o ../logs/tidez_enrich_%j.out
#SBATCH -e ../logs/tidez_enrich_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anirudhs@unc.edu

# Step 4a: cone-searches every BOOM reference catalog (Gaia_DR3, LSPSC, LSDR10, PS1_DR2,
# 2MASS_PSC, AllWISE, CatWISE2020, GALEX, DESI_DR1, NED, VSX, milliquas_v8, TNS) at every
# object in data/all_objects.csv and keeps every match within 5" with all its fields:
# data/crossmatches/<catalog>.parquet. Network-bound, so few CPUs are needed.
# Resumable: finished batches are kept in data/crossmatches_parts/ and skipped when the job
# is re-submitted; data/crossmatches/ is written once every object is done.
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

# enrich.py does `from tidez.logger import ...`; PYTHONPATH lets that resolve
# when the script is run by path.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_DIR"
time "$PYTHON" "$PROJECT_DIR/tidez/enrich.py"
