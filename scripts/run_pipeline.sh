#!/bin/bash
# Submit the whole pipeline (or part of it) in one go, ending with data/fast_transients.h5.
# Each step is its own SLURM job, chained with --dependency=afterok on the steps it needs,
# so independent steps run in parallel and a failed step stops everything after it.
#
#   step 1   01_create_masterfile.sh       data/all_objects.csv
#   step 2   02_build_lcs.sh               data/lightcurves*.parquet          needs 1
#   step 3   03_get_boom_lc.sh             data/boom_lc.parquet               needs 1
#   step 4a  04a_enrich.sh                 data/crossmatches/                 needs 1
#   step 4b  04b_download_tns_objects.sh   data/tns_public_objects.csv
#   step 5   05_find_fast_transients.sh    data/fast_transients.csv/.txt     needs 2, 3, 4a, 4b
#   step 6   06_scanning_setup.sh --fresh  data/fast_transients.h5           needs 5
#
# Steps left out are assumed done: their output files in data/ are used as they are.
#
# Usage (from anywhere):
#   bash run_pipeline.sh                       # everything
#   bash run_pipeline.sh --from 5              # steps 5 and 6 (filter changed)
#   bash run_pipeline.sh --steps "2 5 6"       # DECam light-curve rules changed
#   bash run_pipeline.sh --from 5 -- --prefix T --rise-threshold 1.0
#                                              # anything after -- goes to step 5
#
# Steps 3 and 4a need BOOM_USERNAME / BOOM_PASSWORD, step 4b TNS_ANIBOT_API /
# TNS_ANIBOT_USER_AGENT, exported in this shell (sbatch passes them on to the jobs).

set -euo pipefail
cd "$(dirname "$0")"   # the job scripts write their logs to ../logs relative to here
mkdir -p ../logs

ALL_STEPS="1 2 3 4a 4b 5 6"
steps="$ALL_STEPS"
filter_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --steps) steps="$2"; shift 2 ;;
        --from)
            steps=""; on=0
            for s in $ALL_STEPS; do
                [ "${s%[ab]}" = "$2" ] || [ "$s" = "$2" ] && on=1
                [ $on = 1 ] && steps="$steps $s"
            done
            [ -n "$steps" ] || { echo "unknown step: $2" >&2; exit 1; }
            shift 2 ;;
        --) shift; filter_args=("$@"); break ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done
# "--steps 4" means both 4a and 4b
steps=$(for s in $steps; do [ "$s" = 4 ] && echo "4a 4b" || echo "$s"; done | xargs)

has() { [[ " $steps " == *" $1 "* ]]; }

if { has 3 || has 4a; } && { [ -z "${BOOM_USERNAME:-}" ] || [ -z "${BOOM_PASSWORD:-}" ]; }; then
    echo "steps 3/4a need BOOM_USERNAME and BOOM_PASSWORD exported" >&2; exit 1
fi
if has 4b && { [ -z "${TNS_ANIBOT_API:-}" ] || [ -z "${TNS_ANIBOT_USER_AGENT:-}" ]; }; then
    echo "step 4b needs TNS_ANIBOT_API and TNS_ANIBOT_USER_AGENT exported" >&2; exit 1
fi

script_of() {
    case "$1" in
        1) echo 01_create_masterfile.sh ;;   2) echo 02_build_lcs.sh ;;
        3) echo 03_get_boom_lc.sh ;;         4a) echo 04a_enrich.sh ;;
        4b) echo 04b_download_tns_objects.sh ;;
        5) echo 05_find_fast_transients.sh ;; 6) echo 06_scanning_setup.sh ;;
    esac
}
needs_of() {
    case "$1" in
        2|3|4a) echo 1 ;;  5) echo 2 3 4a 4b ;;  6) echo 5 ;;  *) echo "" ;;
    esac
}

for s in $ALL_STEPS; do
    has "$s" || continue
    deps=""
    for n in $(needs_of "$s"); do
        job_var="JOB_$n"
        [ -n "${!job_var:-}" ] && deps="$deps:${!job_var}"
    done
    opts=(--parsable)
    [ -n "$deps" ] && opts+=(--dependency="afterok$deps" --kill-on-invalid-dep=yes)
    args=()
    [ "$s" = 5 ] && args=(${filter_args[@]+"${filter_args[@]}"})
    [ "$s" = 6 ] && args=(--fresh)
    job=$(sbatch "${opts[@]}" "$(script_of "$s")" ${args[@]+"${args[@]}"})
    printf -v "JOB_$s" '%s' "$job"
    echo "step $s  job $job  $(script_of "$s") ${args[*]:-}${deps:+  (after ${deps#:})}"
done

echo
echo "Watch with: squeue -u $USER"
echo "Step 5 ends its log with BENCHMARK PASSED / BENCHMARK TEST FAILED:"
echo "  grep BENCHMARK ../logs/tidez_find_fast_transients_${JOB_5:-<jobid>}.err"
