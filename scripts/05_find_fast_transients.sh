#!/bin/bash
#SBATCH -J tidez_find_fast_transients
#SBATCH -p RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000M
#SBATCH -A phy250012p
#SBATCH -t 02:00:00
#SBATCH -o ../logs/tidez_find_fast_transients_%j.out
#SBATCH -e ../logs/tidez_find_fast_transients_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anirudhs@unc.edu

# Step 5: fast-transient filter on the combined DECam + ZTF + LSST light curves
# (data/lightcurves.parquet + data/boom_lc.parquet), with the basic vetoes from
# data/crossmatches/ and data/tns_public_objects.csv. The funnel (objects left after each
# cut, and removed by each cut alone) is printed to the .err log. Writes
# data/fast_transients.csv/.txt and data/fast_transients_stats.parquet. The log ends with
# BENCHMARK PASSED / BENCHMARK TEST FAILED for T202502161025195m274154.
# Single-threaded; the 32 CPUs are only there because RM-shared hands out memory
# per core (~2 GB each). Extra options are passed through, e.g.
#   sbatch 05_find_fast_transients.sh --rise-threshold 1.0 --require either

# Call the conda env's interpreter directly: `conda activate` doesn't work in a
# non-interactive batch shell.
PYTHON=/jet/home/salgundi/envs/tidez/bin/python
PROJECT_DIR=/ocean/projects/phy250012p/salgundi/tidez

# find_fast_transients.py does `from tidez... import ...`; PYTHONPATH lets that resolve
# when the script is run by path.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_DIR"
time "$PYTHON" "$PROJECT_DIR/tidez/find_fast_transients.py" "$@"
