#!/bin/bash
#SBATCH -J tidez_build_h5
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH -A phy250012p
#SBATCH -t 30:00:00
#SBATCH -o ../logs/tidez_build_h5_%j.out
#SBATCH -e ../logs/tidez_build_h5_%j.err

source /ocean/projects/phy250012p/salgundi/tidez/.venv/bin/activate

cp /ocean/projects/phy250012p/salgundi/tidez/data/tidez_master.h5 \
   /ocean/projects/phy250012p/salgundi/tidez/data/desirt_master_bd.h5

time uv run python /ocean/projects/phy250012p/salgundi/tidez/tidez/crossmatch_boom.py \
    --h5 /ocean/projects/phy250012p/salgundi/tidez/data/tidez_master.h5