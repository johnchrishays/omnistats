# Code for policy learning vs CATE estimation

## Setup

```
pip install -r requirements.txt
```

## Run simulations
```
sbatch run.sh experiments/debug_full_info.txt
```

## Run analysis
```
python3 analyze_policy_results.py  --summary-paths results/sweeps/debug_full_info_summary.csv  --output-dir results/analysis  --output-prefix debug_full_info --plot-top 3
```

## Clear analysis in preparation for a new training run
```
./clear.sh
```