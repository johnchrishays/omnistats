# Code for policy learning vs CATE estimation

## Setup

```
pip install -r requirements.txt
```

## Run simulations
```
sbatch --array=0-9 gerber.sh
sbatch --array=0-9 nsw.sh
```

Dataset-specific scripts:
- `gerber.sh`
- `nsw.sh`

Each run now uses one parameter setting per dataset. `experiment_name` is set to the dataset name automatically.
`full` evaluation mode means full-information evaluation (train + evaluate on the full sample, no train/test split).

Array tasks write per-task outputs here:
- `results/_array_tmp/gerber/summary_<run_id>.csv`
- `results/_array_tmp/nsw/summary_<run_id>.csv`

## Run analysis
```
python3 analyze_policy_results.py \
  --summary-glob 'results/_array_tmp/gerber/summary_*.csv,results/_array_tmp/nsw/summary_*.csv' \
  --output-dir results/analysis \
  --plot-y-mode absolute
```

## Clear analysis in preparation for a new training run
```
./clear.sh
```
