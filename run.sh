#!/bin/bash
#
#SBATCH --job-name=policylearning
#
#SBATCH --ntasks=1
#SBATCH -p sched_mit_sloan_batch_r8
#SBATCH -o "slurm-%a.out"
#SBATCH -e "slurm-%a.out"
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=ERROR,END
#SBATCH --mail-user=jhays@mit.edu
#SBATCH --array=0-99


export JHAYS="/home/jhays"
python3 $JHAYS/omnistats/policylearning.py -i $SLURM_ARRAY_TASK_ID
