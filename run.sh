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
#SBATCH --array=0-19

## # SBATCH --mail-user=jhays@mit.edu

set -euo pipefail

export JHAYS="/home/jhays"
PROJECT_DIR="$JHAYS/omnistats"
cd "$PROJECT_DIR"

RUN_ID="${SLURM_ARRAY_TASK_ID:-${RUN_ID:-0}}"
python3 "$PROJECT_DIR/policylearning.py" "$@" --run-id "${RUN_ID}"
