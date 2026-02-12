# Code for policy learning vs CATE estimation

## Setup

```
pip install -r requirements.txt
```

## Run simulations
```
sbatch run.sh experiments/debug_full_info.txt
```

Use `--dataset all` in a manifest (see `experiments/policy_sweep_example.txt`) to run both configured datasets:
- `gerber`: defaults to `treat` / `voted14`
- `nsw`: defaults to `treated` / `re78`, filtered to `sample == 1`, with all covariates as default features

## Run analysis
```
python3 analyze_policy_results.py --summary-paths results/sweeps/debug_full_info_summary.csv --output-dir results/analysis --output-prefix debug_full_info --plot-top-nontrivial 3
```

## Clear analysis in preparation for a new training run
```
./clear.sh
```
