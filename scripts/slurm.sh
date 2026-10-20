#!/bin/bash
#SBATCH -J tidez_run
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH -A phy250012p
#SBATCH -t 06:00:00
#SBATCH -o ../logs/tidez_run_%j.out
#SBATCH -e ../logs/tidez_run_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anirudhs@unc.edu


bash /ocean/projects/phy250012p/salgundi/tidez/scripts/run_pipeline.sh --steps "2 5 6"  