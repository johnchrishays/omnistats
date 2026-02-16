# Code for policy learning vs CATE estimation

## Setup

Install necessary packages:
```
pip install -r requirements.txt
```

Download the datasets, and put each into its corresponding directory.
| Directory name | Data link | 
|----------|----------|
| `gerber`  | https://huber.research.yale.edu/writings.html  | 
| `nsw` | https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/23407/DYEWLO&version=1.0. |
| `jtpa` | https://www.econometricsociety.org/publications/econometrica/2018/03/01/who-should-be-treated-empirical-welfare-maximization-methods |

## Run simulations
```
sbatch gerber.sh
sbatch nsw.sh
sbatch jtpa.sh
```

Dataset-specific scripts:
- `gerber.sh`
- `nsw.sh`
- `jtpa.sh`

Each run now uses one parameter setting per dataset. `experiment_name` is set to the dataset name automatically.
`full` evaluation mode means full-information evaluation (train + evaluate on the full sample, no train/test split).

Array tasks write per-task outputs here:
- `results/_array_tmp/gerber/summary_<run_id>.csv`
- `results/_array_tmp/nsw/summary_<run_id>.csv`
- `results/_array_tmp/jtpa/summary_<run_id>.csv`

## Run analysis
```
python3 analyze_policy_results.py \
  --plot-y-mode absolute
```

## Clear analysis in preparation for a new training run
```
./clear.sh
```
