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
MANIFEST_MODE=0
MANIFEST_STEM=""

append_csv_locked() {
  local src="$1"
  local dst="$2"
  local lock_file="${dst}.lock"

  if [[ ! -s "$src" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "$dst")"

  if command -v flock >/dev/null 2>&1; then
    (
      flock 9
      if [[ ! -f "$dst" ]]; then
        cat "$src" > "$dst"
      else
        tail -n +2 "$src" >> "$dst"
      fi
    ) 9>"$lock_file"
  else
    if [[ ! -f "$dst" ]]; then
      cat "$src" > "$dst"
    else
      tail -n +2 "$src" >> "$dst"
    fi
  fi
}

# If first arg is a file, treat it as a single shared argument manifest.
# Every array task runs the same argument set; results are concatenated.
if [[ ${#ARGS[@]} -gt 0 && -f "${ARGS[0]}" ]]; then
  MANIFEST_MODE=1
  ARG_FILE="${ARGS[0]}"
  MANIFEST_STEM="$(basename "$ARG_FILE")"
  MANIFEST_STEM="${MANIFEST_STEM%.*}"

  # Strip comments/blank lines and trailing continuation backslashes.
  PARAM_TEXT="$(
    awk '
      {
        line = $0
        sub(/[[:space:]]*#.*$/, "", line)
        sub(/[[:space:]]*\\[[:space:]]*$/, "", line)
        if (line ~ /^[[:space:]]*$/) next
        printf "%s ", line
      }
    ' "$ARG_FILE"
  )"

  if [[ -z "${PARAM_TEXT//[[:space:]]/}" ]]; then
    echo "No arguments found in manifest: ${ARG_FILE}" >&2
    exit 1
  fi

  # shellcheck disable=SC2206
  eval "FILE_ARGS=(${PARAM_TEXT})"
  ARGS=("${FILE_ARGS[@]}" "${ARGS[@]:1}")
fi

if [[ $MANIFEST_MODE -eq 1 ]]; then
  TMP_DIR="results/_array_tmp/${MANIFEST_STEM}"
  mkdir -p "$TMP_DIR"

  AGG_SUMMARY_PATH="${RESULTS_AGG_SUMMARY:-results/sweeps/${MANIFEST_STEM}_summary.csv}"
  AGG_DETAILED_PATH="${RESULTS_AGG_DETAILED:-results/sweeps/${MANIFEST_STEM}_detailed.csv}"
  TASK_SUMMARY_PATH="${TMP_DIR}/summary_${RUN_ID}.csv"
  TASK_DETAILED_PATH="${TMP_DIR}/detailed_${RUN_ID}.csv"
  TASK_CONFIG_PATH="${TMP_DIR}/config_${RUN_ID}.json"

  python3 "$PROJECT_DIR/policylearning.py" \
    "${ARGS[@]}" \
    --run-id "${RUN_ID}" \
    --summary-path "${TASK_SUMMARY_PATH}" \
    --detailed-path "${TASK_DETAILED_PATH}" \
    --config-path "${TASK_CONFIG_PATH}"

  append_csv_locked "${TASK_SUMMARY_PATH}" "${AGG_SUMMARY_PATH}"
  append_csv_locked "${TASK_DETAILED_PATH}" "${AGG_DETAILED_PATH}"

  echo "Task ${RUN_ID} appended to:"
  echo "  ${AGG_SUMMARY_PATH}"
  echo "  ${AGG_DETAILED_PATH}"
else
  python3 "$PROJECT_DIR/policylearning.py" "${ARGS[@]}" --run-id "${RUN_ID}"
fi
