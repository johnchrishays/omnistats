#!/bin/bash
#
#SBATCH --job-name=policylearning
#
#SBATCH --ntasks=1
#SBATCH -p sched_mit_sloan_batch_r8
#SBATCH -o "logs/slurm-%a.out"
#SBATCH -e "logs/slurm-%a.out"
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=ERROR,END
#SBATCH --mail-user=jhays@mit.edu
#SBATCH --array=0-3

set -euo pipefail

export JHAYS="/home/jhays"
PROJECT_DIR="$JHAYS/omnistats"
cd "$PROJECT_DIR"

RUN_ID="${SLURM_ARRAY_TASK_ID:-${RUN_ID:-0}}"
ARGS=("$@")

# If first arg is a file, treat it as an argument manifest.
# Each non-empty, non-comment line corresponds to one array index.
if [[ ${#ARGS[@]} -gt 0 && -f "${ARGS[0]}" ]]; then
  ARG_FILE="${ARGS[0]}"
  LINE_NUM=$((RUN_ID + 1))
  PARAM_LINE=$(awk 'NF && $1 !~ /^#/' "$ARG_FILE" | sed -n "${LINE_NUM}p")

  if [[ -z "${PARAM_LINE}" ]]; then
    echo "No experiment line found for run_id=${RUN_ID} in ${ARG_FILE}" >&2
    exit 1
  fi

  # shellcheck disable=SC2206
  eval "FILE_ARGS=(${PARAM_LINE})"
  ARGS=("${FILE_ARGS[@]}" "${ARGS[@]:1}")
fi

python3 "$PROJECT_DIR/policylearning.py" --run-id "${RUN_ID}" "${ARGS[@]}"
