#!/bin/bash
#
#SBATCH --job-name=policylearning_gerber
#
#SBATCH --ntasks=1
#SBATCH -p sched_mit_sloan_batch_r8
#SBATCH -o "logs/gerber-%a.out"
#SBATCH -e "logs/gerber-%a.out"
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=ERROR,END
#SBATCH --array=0-19

set -euo pipefail

PROJECT_DIR="/home/jhays/omnistats"
cd "$PROJECT_DIR"

RUN_ID="${SLURM_ARRAY_TASK_ID:-${RUN_ID:-0}}"
OUT_DIR="results/_array_tmp/gerber"
mkdir -p "$OUT_DIR"

python3 "$PROJECT_DIR/policylearning.py" \
  --dataset gerber \
  --n-reps 50 \
  --train-size 0.7 \
  --evaluation-modes holdout,full \
  --tree-train-cost 0.1 \
  --cost-grid-start 0.0 \
  --cost-grid-stop 0.2 \
  --cost-grid-num 21 \
  --run-id "${RUN_ID}" \
  --summary-path "${OUT_DIR}/summary_${RUN_ID}.csv" \
  --detailed-path "${OUT_DIR}/detailed_${RUN_ID}.csv" \
  --config-path "${OUT_DIR}/config_${RUN_ID}.json"
