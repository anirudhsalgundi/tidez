#!/bin/bash
#SBATCH -J tidez_download_tns_objects
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH -A phy250012p
#SBATCH -t 00:30:00
#SBATCH -o ../logs/tidez_download_tns_objects_%j.out
#SBATCH -e ../logs/tidez_download_tns_objects_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anirudhs@unc.edu

# Step 4b: downloads the TNS public-objects dump to data/tns_public_objects.csv, which
# step 5 uses to flag objects already on TNS (and whether our DECam objid is listed).
#
# Needs TNS_ANIBOT_API and TNS_ANIBOT_USER_AGENT exported in the shell you run sbatch
# from (sbatch passes your environment on to the job).
if [ -z "$TNS_ANIBOT_API" ] || [ -z "$TNS_ANIBOT_USER_AGENT" ]; then
    echo "TNS_ANIBOT_API / TNS_ANIBOT_USER_AGENT are not set; export them before sbatch." >&2
    exit 1
fi

# Call the conda env's interpreter directly: `conda activate` doesn't work in a
# non-interactive batch shell.
PYTHON=/jet/home/salgundi/envs/tidez/bin/python
PROJECT_DIR=/ocean/projects/phy250012p/salgundi/tidez

# download_tns_objects.py is run by path; PYTHONPATH kept for consistency
# with the other steps.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_DIR"
time "$PYTHON" "$PROJECT_DIR/tidez/download_tns_objects.py"
