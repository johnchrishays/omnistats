import argparse
import json
from pathlib import Path

from econml.policy import DRPolicyTree
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DEFAULT_OUTPUT_DIR = "results"
ALL_DATASET_NAME = "all"
ALL_FEATURE_SET = "all"
EVAL_MODE_CHOICES = {"holdout", "full"}

DATASET_PRESETS = {
    "gerber": {
        "data_path": "gerber/gerber_generalizability_2/PublicReplicationData.dta",
        "state_var": "state",
        "state_value": "TX",
        "treat_var": "treat",
        "outcome_var": "voted14",
        "subset_query": None,
        "cate_feature_set": "cate_gerber_alt",
        "policy_feature_set": "policy_gerber",
        "inject_state_column": False,
    },
    "nsw": {
        "data_path": "nsw/ec675_nsw.tab",
        "state_var": "__dataset_state__",
        "state_value": "NSW",
        "treat_var": "treated",
        "outcome_var": "re78",
        "subset_query": "sample == 1",
        "cate_feature_set": "cate_nsw",
        "policy_feature_set": "cate_nsw",
        "inject_state_column": True,
    },
}

FEATURE_PRESETS = {
    "cate_gerber": ["i_age", "flag_hhid_mult_hhid", "flag_hhid_mult_z", "flag_drop_hhid",
        "vote_hist", "state_median", "vh_stratum", "vhblw", "d_married",
        "d_unmarried", "d_hhsize1", "d_hhsize2", "d_hhsize3", "d_hhsize4"],
    "cate_gerber_alt": ["vhblw", "vhabv", 'd_female', "d_hhsize1", "d_hhsize2", "d_hhsize3"],
    "policy_gerber": ["d_married", "d_hhsize1", "d_hhsize2", "d_hhsize3", "d_hhsize4","d_race_b", "d_race_h", "d_race_o", "d_race_w", "d_female", "d_notfem"],
    "cate_nsw": ['age', 'educ', 'black', 'married', 'hisp',  're74', 're75'],
    "policy_nsw": ['age', 'educ', 'black', 'married','hisp', 'nodegree',  're74', 're75']
}
FEATURE_SET_CHOICES = sorted(list(FEATURE_PRESETS.keys()) + [ALL_FEATURE_SET])


def parse_csv_arg(raw):
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values if values else None


def parse_bool_or_none(raw):
    if raw is None:
        return None
    lowered = raw.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Could not parse boolean value from '{raw}'.")


def safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return np.nan
    return float(np.nanmean(arr))


def safe_nanstd(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return np.nan
    return float(np.nanstd(arr, ddof=0))


def get_ate(df, treatment_var, outcome_var):
    treat_mask = df[treatment_var] == 1
    control_mask = df[treatment_var] == 0

    n_treat = int(treat_mask.sum())
    n_control = int(control_mask.sum())

    mean_treat = float(df.loc[treat_mask, outcome_var].mean()) if n_treat > 0 else np.nan
    mean_control = float(df.loc[control_mask, outcome_var].mean()) if n_control > 0 else np.nan
    ate = mean_treat - mean_control if n_treat > 0 and n_control > 0 else np.nan

    return {
        "mean_treatment": mean_treat,
        "mean_control": mean_control,
        "ate": ate,
        "n_treat": n_treat,
        "n_control": n_control,
    }


def compute_cate(df, group_vars, outcome_var, treat_var):
    grouped = df.groupby(group_vars, as_index=False, dropna=False)
    try:
        cate = grouped.apply(
            lambda x: pd.Series(get_ate(x, treatment_var=treat_var, outcome_var=outcome_var)),
            include_groups=False,
        )
    except TypeError:
        cate = grouped.apply(
            lambda x: pd.Series(get_ate(x, treatment_var=treat_var, outcome_var=outcome_var))
        )
    if any(col not in cate.columns for col in group_vars):
        cate = cate.reset_index()
    return cate


def policy_value_cate(train_cates, eval_cates, group_vars, cost):
    train_assignments = train_cates[group_vars + ["ate"]].copy()
    train_assignments["recommend_treat"] = (
        train_assignments["ate"].notna() & ((train_assignments["ate"] - cost) > 0)
    )

    aligned = eval_cates.merge(
        train_assignments[group_vars + ["recommend_treat"]],
        on=group_vars,
        how="left",
    )
    aligned["recommend_treat"] = aligned["recommend_treat"].eq(True)

    subgroup_sizes = aligned["n_treat"] + aligned["n_control"]
    total_size = float(subgroup_sizes.sum())
    if total_size <= 0:
        return np.nan

    net_benefit = aligned["ate"] - cost
    net_benefit = net_benefit.fillna(0.0)
    weighted_net_benefit = (
        net_benefit * aligned["recommend_treat"].astype(float) * subgroup_sizes
    ).sum()
    return float(weighted_net_benefit / total_size)


def policy_value_tree(leaf_cates, cost):
    if leaf_cates is None:
        return np.nan

    net_benefit = leaf_cates["ate"] - cost
    net_benefit = net_benefit.fillna(0.0)

    treat_mask = leaf_cates["recommended_treatment"] == 1
    subgroup_sizes = leaf_cates["n_treat"] + leaf_cates["n_control"]
    total_size = float(subgroup_sizes.sum())
    if total_size <= 0:
        return np.nan

    weighted_net_benefit = (net_benefit * treat_mask.astype(float) * subgroup_sizes).sum()
    return float(weighted_net_benefit / total_size)


def policy_value_ate(train_ate, eval_ate, cost):
    if not np.isfinite(train_ate):
        return np.nan
    if train_ate > cost:
        if np.isfinite(eval_ate):
            return float(eval_ate - cost)
        return np.nan
    return 0.0


def policy_value_ate_grouped(train_cates, eval_cates, cost):
    train_net_benefit = train_cates["ate"] - cost
    train_net_benefit = train_net_benefit.fillna(0.0)

    train_subgroup_sizes = train_cates["n_treat"] + train_cates["n_control"]
    train_total_size = float(train_subgroup_sizes.sum())
    if train_total_size <= 0:
        return np.nan
    train_avg_net_benefit = float((train_net_benefit * train_subgroup_sizes).sum() / train_total_size)

    if train_avg_net_benefit <= 0:
        return 0.0

    eval_net_benefit = eval_cates["ate"] - cost
    eval_net_benefit = eval_net_benefit.fillna(0.0)

    eval_subgroup_sizes = eval_cates["n_treat"] + eval_cates["n_control"]
    eval_total_size = float(eval_subgroup_sizes.sum())
    if eval_total_size <= 0:
        return np.nan

    return float((eval_net_benefit * eval_subgroup_sizes).sum() / eval_total_size)


def get_policy_tree(df, policy_features, cost, outcome_var, treat_var, tree_kwargs, use_forest_nuisance):
    X = df[policy_features]
    T = df[treat_var].astype("category")
    Y = df[outcome_var] - cost

    resolved_tree_kwargs = dict(tree_kwargs)
    # if use_forest_nuisance:
    #     nuisance_random_state = resolved_tree_kwargs.get("random_state")
    #     resolved_tree_kwargs.setdefault(
    #         "model_propensity",
    #         RandomForestClassifier(
    #             n_estimators=300,
    #             min_samples_leaf=5,
    #             random_state=nuisance_random_state,
    #             n_jobs=1,
    #         ),
    #     )
    #     resolved_tree_kwargs.setdefault(
    #         "model_regression",
    #         RandomForestRegressor(
    #             n_estimators=300,
    #             min_samples_leaf=5,
    #             random_state=nuisance_random_state,
    #             n_jobs=1,
    #         ),
    #     )

    model = DRPolicyTree(**resolved_tree_kwargs)
    model.fit(Y, T, X=X)
    return model


def get_leaf_cates(policy_tree, df, policy_features, outcome_var, treat_var):
    X = df[policy_features]
    recommended_treatments = policy_tree.predict(X)
    leaf_ids = policy_tree.policy_model_.tree_.apply(X.values.astype(np.float64))

    leaf_df = df.copy()
    leaf_df["leaf_id"] = leaf_ids
    leaf_df["recommended_treatment"] = recommended_treatments

    leaf_cates = compute_cate(
        leaf_df,
        ["leaf_id"],
        outcome_var=outcome_var,
        treat_var=treat_var,
    )
    leaf_recommendations = leaf_df.groupby("leaf_id", as_index=False)["recommended_treatment"].first()
    leaf_cates = leaf_cates.merge(leaf_recommendations, on="leaf_id", how="left")
    return leaf_cates


def combine_queries(*queries):
    cleaned = [q.strip() for q in queries if q is not None and q.strip()]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    return "(" + ") & (".join(cleaned) + ")"


def load_dataframe(data_path):
    path = Path(data_path)
    suffix = path.suffix.lower()
    if suffix == ".dta":
        return pd.read_stata(path)
    if suffix in {".tab", ".tsv"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path)


def infer_all_covariates(df_columns, protected_cols):
    features = [col for col in sorted(df_columns) if col not in protected_cols]
    if not features:
        raise ValueError("No covariates available after excluding treatment/outcome/state columns.")
    return features


def build_binary_feature_matrix(df, feature_cols, prefix="bin__"):
    out = df.copy()
    feature_map = {}
    metadata = []

    for col in dict.fromkeys(feature_cols):
        if col not in out.columns:
            raise ValueError(f"Feature column '{col}' is not present in data.")

        s_num = pd.to_numeric(out[col], errors="coerce")
        non_missing = s_num.dropna()
        uniq = set(non_missing.unique().tolist())
        out_col = f"{prefix}{col}"

        if len(non_missing) == 0:
            out[out_col] = 0
            transform = "all_missing_to_zero"
            median = np.nan
        elif uniq.issubset({0.0, 1.0}):
            out[out_col] = s_num.fillna(0.0).astype(np.int8)
            transform = "binary_passthrough"
            median = np.nan
        else:
            median = float(non_missing.median())
            out[out_col] = (s_num.notna() & (s_num > median)).astype(np.int8)
            transform = "median_split"

        feature_map[col] = out_col
        metadata.append(
            {
                "source_feature": col,
                "binary_feature": out_col,
                "transform": transform,
                "median": median,
            }
        )

    return out, feature_map, metadata


def resolve_dataset_runs(args):
    if args.dataset == ALL_DATASET_NAME:
        dataset_names = sorted(DATASET_PRESETS.keys())
        if args.data_path:
            raise ValueError("--data-path cannot be used with --dataset all.")
    else:
        if args.dataset not in DATASET_PRESETS:
            raise ValueError(f"Unknown dataset '{args.dataset}'.")
        dataset_names = [args.dataset]

    dataset_runs = []
    for dataset_name in dataset_names:
        preset = DATASET_PRESETS[dataset_name]
        dataset_runs.append(
            {
                "dataset": dataset_name,
                "data_path": args.data_path if args.data_path else preset["data_path"],
                "state_var": preset["state_var"],
                "state_value": preset["state_value"],
                "treat_var": args.treat_var if args.treat_var is not None else preset["treat_var"],
                "outcome_var": args.outcome_var if args.outcome_var is not None else preset["outcome_var"],
                "subset_query": combine_queries(preset.get("subset_query"), args.subset_query),
                "drop_columns": list(preset.get("drop_columns", [])),
                "default_cate_feature_set": preset.get("cate_feature_set"),
                "default_policy_feature_set": preset.get("policy_feature_set"),
                "inject_state_column": bool(preset.get("inject_state_column", False)),
            }
        )
    return dataset_runs


def resolve_features(
    args,
    df_columns,
    state_var,
    treat_var,
    outcome_var,
    default_cate_feature_set,
    default_policy_feature_set,
):
    if args.cate_features:
        cate_features = parse_csv_arg(args.cate_features)
    else:
        cate_feature_set = args.cate_feature_set or default_cate_feature_set
        if cate_feature_set is None:
            raise ValueError("No CATE feature configuration resolved.")
        if cate_feature_set == ALL_FEATURE_SET:
            cate_features = infer_all_covariates(
                df_columns,
                protected_cols={state_var, treat_var, outcome_var},
            )
        else:
            cate_features = FEATURE_PRESETS[cate_feature_set]

    if args.policy_features:
        policy_features = parse_csv_arg(args.policy_features)
    else:
        policy_feature_set = (
            args.policy_feature_set
            if args.policy_feature_set is not None
            else default_policy_feature_set
        )
        if policy_feature_set is None:
            policy_features = list(cate_features)
        elif policy_feature_set == ALL_FEATURE_SET:
            policy_features = infer_all_covariates(
                df_columns,
                protected_cols={state_var, treat_var, outcome_var},
            )
        else:
            policy_features = FEATURE_PRESETS[policy_feature_set]

    required = set(cate_features + policy_features + [state_var, treat_var, outcome_var])
    missing = sorted(col for col in required if col not in df_columns)
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    return cate_features, policy_features


def resolve_eval_modes(args):
    modes = parse_csv_arg(args.evaluation_modes)
    if not modes:
        raise ValueError("At least one evaluation mode must be provided.")

    unknown = sorted(set(modes) - EVAL_MODE_CHOICES)
    if unknown:
        raise ValueError(f"Unknown evaluation modes: {unknown}. Allowed: {sorted(EVAL_MODE_CHOICES)}")
    return modes


def resolve_cost_grid(args):
    if args.cost_grid:
        values = [float(v) for v in parse_csv_arg(args.cost_grid)]
        if not values:
            raise ValueError("--cost-grid was provided but no values were parsed.")
        return np.array(values, dtype=float)
    return np.linspace(args.cost_grid_start, args.cost_grid_stop, args.cost_grid_num)


def resolve_tree_kwargs(args):
    tree_kwargs = {}
    if args.tree_max_depth is not None:
        tree_kwargs["max_depth"] = args.tree_max_depth
    if args.tree_min_samples_leaf is not None:
        tree_kwargs["min_samples_leaf"] = args.tree_min_samples_leaf
    if args.tree_min_samples_split is not None:
        tree_kwargs["min_samples_split"] = args.tree_min_samples_split
    if args.tree_random_state is not None:
        tree_kwargs["random_state"] = args.tree_random_state

    honest_value = parse_bool_or_none(args.tree_honest)
    if honest_value is not None:
        tree_kwargs["honest"] = honest_value

    if args.tree_extra_kwargs:
        extras = json.loads(args.tree_extra_kwargs)
        if not isinstance(extras, dict):
            raise ValueError("--tree-extra-kwargs must parse to a JSON object.")
        tree_kwargs.update(extras)

    return tree_kwargs


def default_output_paths(args):
    run_tag = "None" if args.run_id is None else str(args.run_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else output_dir / f"policy_learning_results_{run_tag}.csv"
    )
    detailed_path = (
        Path(args.detailed_path)
        if args.detailed_path
        else output_dir / f"policy_learning_detailed_{run_tag}.csv"
    )
    config_path = (
        Path(args.config_path)
        if args.config_path
        else output_dir / f"policy_learning_config_{run_tag}.json"
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    return summary_path, detailed_path, config_path


def filter_to_state(df, state_var, state_value):
    if state_var not in df.columns:
        raise ValueError(f"State/group column '{state_var}' is not present in data.")

    mask = df[state_var].astype(str) == str(state_value)
    if not mask.any():
        state_num = pd.to_numeric(pd.Series([state_value]), errors="coerce").iloc[0]
        if pd.notna(state_num):
            numeric_col = pd.to_numeric(df[state_var], errors="coerce")
            mask = numeric_col == float(state_num)

    filtered = df[mask].copy()
    if filtered.empty:
        available = sorted(df[state_var].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"No rows matched hardcoded state/group '{state_value}' in column '{state_var}'. "
            f"Available values: {available[:20]}"
        )

    return filtered


def run_experiment(args):
    eval_modes = resolve_eval_modes(args)
    costs = resolve_cost_grid(args)
    tree_train_cost = float(args.tree_train_cost)
    tree_kwargs = resolve_tree_kwargs(args)
    dataset_runs = resolve_dataset_runs(args)

    if args.train_size <= 0 or args.train_size >= 1:
        raise ValueError("--train-size must be in (0, 1).")
    if args.n_reps <= 0:
        raise ValueError("--n-reps must be positive.")

    run_id = 0 if args.run_id is None else args.run_id
    base_seed = args.seed if args.seed is not None else 1729 + (run_id * 1009)
    rng = np.random.default_rng(base_seed)

    detailed_rows = []
    resolved_dataset_runs = []

    for dataset_cfg in dataset_runs:
        dataset_name = dataset_cfg["dataset"]
        experiment_name = dataset_name
        state_var = dataset_cfg["state_var"]
        state_value = dataset_cfg["state_value"]
        treat_var = dataset_cfg["treat_var"]
        outcome_var = dataset_cfg["outcome_var"]

        df = load_dataframe(dataset_cfg["data_path"])
        if dataset_cfg.get("drop_columns"):
            df = df.drop(columns=dataset_cfg["drop_columns"], errors="ignore")
        if dataset_cfg["inject_state_column"] and state_var not in df.columns:
            df[state_var] = state_value

        if dataset_cfg["subset_query"]:
            try:
                df = df.query(dataset_cfg["subset_query"]).copy()
            except Exception as exc:
                raise ValueError(
                    f"Subset query failed for dataset '{dataset_name}': {dataset_cfg['subset_query']}"
                ) from exc

        df = filter_to_state(df, state_var=state_var, state_value=state_value)

        raw_cate_features, raw_policy_features = resolve_features(
            args,
            set(df.columns),
            state_var=state_var,
            treat_var=treat_var,
            outcome_var=outcome_var,
            default_cate_feature_set=dataset_cfg["default_cate_feature_set"],
            default_policy_feature_set=dataset_cfg["default_policy_feature_set"],
        )
        selected_feature_union = list(dict.fromkeys(raw_cate_features + raw_policy_features))
        full_bin_df, feature_map, binarization_metadata = build_binary_feature_matrix(df, selected_feature_union)
        cate_features = [feature_map[col] for col in raw_cate_features]
        policy_features = raw_policy_features # [feature_map[col] for col in raw_policy_features]

        resolved_dataset_runs.append(
            {
                "dataset": dataset_name,
                "data_path": dataset_cfg["data_path"],
                "state_var": state_var,
                "state_value": str(state_value),
                "treat_var": treat_var,
                "outcome_var": outcome_var,
                "subset_query": dataset_cfg["subset_query"],
                "resolved_cate_features_raw": list(raw_cate_features),
                "resolved_policy_features_raw": list(raw_policy_features),
                "resolved_cate_features": list(cate_features),
                "resolved_policy_features": list(policy_features),
                "feature_binarization": binarization_metadata,
                "resolved_tree_nuisance_mode": (
                    "forest" if dataset_name == "nsw" else "default"
                ),
            }
        )

        rep_iterator = tqdm(
            range(args.n_reps),
            desc=f"dataset={dataset_name}, state={state_value}",
            disable=args.no_progress,
        )

        for rep in rep_iterator:
            split_seed = None
            holdout_df = None
            train_df = None
            if "holdout" in eval_modes:
                split_seed = int(rng.integers(0, 2**31 - 1))
                train_df, holdout_df = train_test_split(
                    df,
                    train_size=args.train_size,
                    shuffle=True,
                    random_state=split_seed,
                )
                bin_train_df, _, _ = build_binary_feature_matrix(train_df, selected_feature_union)
                bin_holdout_df, _, _ = build_binary_feature_matrix(holdout_df, selected_feature_union)
                train_df = train_df.copy()
                holdout_df = holdout_df.copy()
            else:
                # No holdout requested: use a deterministic split-seed marker.
                split_seed = -1

            scenarios = []
            if "holdout" in eval_modes:
                scenarios.append(
                    {
                        "evaluation_mode": "holdout",
                        "split_mode": "random_split",
                        "split_seed": split_seed,
                        "train_df": bin_train_df,
                        "eval_df": bin_holdout_df,
                        "nonbin_train_df": train_df,
                        "nonbin_eval_df": holdout_df,
                    }
                )
            if "full" in eval_modes:
                scenarios.append(
                    {
                        "evaluation_mode": "full",
                        "split_mode": "full_information",
                        "split_seed": -1,
                        "train_df": full_bin_df,
                        "eval_df": full_bin_df,
                        "nonbin_train_df": df,
                        "nonbin_eval_df": df,
                    }
                )

            for scenario in scenarios:
                train_ate = get_ate(
                    scenario["train_df"],
                    treatment_var=treat_var,
                    outcome_var=outcome_var,
                )["ate"]
                train_cates = compute_cate(
                    scenario["train_df"],
                    cate_features,
                    outcome_var=outcome_var,
                    treat_var=treat_var,
                )

                try:
                    trained_tree = get_policy_tree(
                        scenario["nonbin_train_df"],
                        policy_features=policy_features,
                        cost=tree_train_cost,
                        outcome_var=outcome_var,
                        treat_var=treat_var,
                        tree_kwargs=tree_kwargs,
                        use_forest_nuisance=(dataset_name == "nsw"),
                    )
                    tree_error = None
                except Exception as exc:
                    trained_tree = None
                    tree_error = str(exc)

                eval_df = scenario["eval_df"]
                eval_ate = get_ate(eval_df, treatment_var=treat_var, outcome_var=outcome_var)["ate"]
                eval_cates = compute_cate(
                    eval_df,
                    cate_features,
                    outcome_var=outcome_var,
                    treat_var=treat_var,
                )

                leaf_cates = None
                try:
                    if trained_tree is not None:
                        leaf_cates = get_leaf_cates(
                            trained_tree,
                            scenario["nonbin_eval_df"],
                            policy_features=policy_features,
                            outcome_var=outcome_var,
                            treat_var=treat_var,
                        )
                except Exception as exc:
                    if tree_error is None:
                        tree_error = str(exc)

                for c in costs:
                    pv_cate = policy_value_cate(
                        train_cates=train_cates,
                        eval_cates=eval_cates,
                        group_vars=cate_features,
                        cost=c,
                    )
                    pv_ate = policy_value_ate(train_ate=train_ate, eval_ate=eval_ate, cost=c)
                    pv_ate_grouped = policy_value_ate_grouped(
                        train_cates=train_cates,
                        eval_cates=eval_cates,
                        cost=c,
                    )
                    pv_tree = policy_value_tree(leaf_cates=leaf_cates, cost=c)

                    detailed_rows.append(
                        {
                            "dataset": dataset_name,
                            "run_id": run_id,
                            "experiment_name": experiment_name,
                            "rep": rep,
                            "split_seed": scenario["split_seed"],
                            "split_mode": scenario["split_mode"],
                            "state": str(state_value),
                            "evaluation_mode": scenario["evaluation_mode"],
                            "c": float(c),
                            "tree_train_cost": tree_train_cost,
                            "pv_cate": pv_cate,
                            "pv_tree": pv_tree,
                            "pv_ate": pv_ate,
                            "pv_ate_grouped": pv_ate_grouped,
                            "train_ate": train_ate,
                            "eval_ate": eval_ate,
                            "n_train": int(scenario["train_df"].shape[0]),
                            "n_eval": int(eval_df.shape[0]),
                            "tree_error": tree_error,
                        }
                    )

    detailed = pd.DataFrame(detailed_rows)
    if detailed.empty:
        raise RuntimeError("No results were produced; check filters and arguments.")

    summary = (
        detailed.groupby(
            ["dataset", "run_id", "experiment_name", "state", "evaluation_mode", "c", "tree_train_cost"],
            dropna=False,
            as_index=False,
        )
        .agg(
            pv_cates_means=("pv_cate", safe_nanmean),
            pv_cates_sds=("pv_cate", safe_nanstd),
            pv_trees_means=("pv_tree", safe_nanmean),
            pv_trees_sds=("pv_tree", safe_nanstd),
            pv_ates_means=("pv_ate", safe_nanmean),
            pv_ates_sds=("pv_ate", safe_nanstd),
            pv_ates_grouped_means=("pv_ate_grouped", safe_nanmean),
            pv_ates_grouped_sds=("pv_ate_grouped", safe_nanstd),
            n_rows=("pv_cate", "size"),
            n_tree_errors=("tree_error", lambda s: int(pd.Series(s).notna().sum())),
        )
    )

    summary_path, detailed_path, config_path = default_output_paths(args)
    summary.to_csv(summary_path, index=False)
    detailed.to_csv(detailed_path, index=False)

    config_payload = vars(args).copy()
    config_payload["resolved_dataset_runs"] = resolved_dataset_runs
    config_payload["resolved_eval_modes"] = eval_modes
    config_payload["resolved_cost_grid"] = [float(x) for x in costs.tolist()]
    config_payload["resolved_tree_train_cost"] = tree_train_cost
    config_payload["resolved_tree_kwargs"] = tree_kwargs
    config_payload["resolved_datasets"] = [d["dataset"] for d in resolved_dataset_runs]
    config_payload["seed_used"] = base_seed

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, sort_keys=True)

    return summary_path, detailed_path, config_path, summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Policy learning experiment runner (single parameter setting, holdout + full-information evaluation)."
    )

    parser.add_argument("-i", "--run-id", type=int, required=False, dest="run_id")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(list(DATASET_PRESETS.keys()) + [ALL_DATASET_NAME]),
        default="gerber",
        help="Dataset preset to run. Use 'all' to run all configured datasets.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional override for data path. Cannot be combined with --dataset all.",
    )

    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument("--detailed-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)

    parser.add_argument("--subset-query", type=str, default=None)
    parser.add_argument("--treat-var", type=str, default=None)
    parser.add_argument("--outcome-var", type=str, default=None)

    parser.add_argument("--cate-feature-set", type=str, choices=FEATURE_SET_CHOICES, default=None)
    parser.add_argument("--cate-features", type=str, default=None)
    parser.add_argument("--policy-feature-set", type=str, choices=FEATURE_SET_CHOICES, default=None)
    parser.add_argument("--policy-features", type=str, default=None)

    parser.add_argument("--n-reps", type=int, default=50)
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--evaluation-modes",
        type=str,
        default="holdout,full",
        help="Comma-separated subset of holdout,full. full = train+evaluate on full sample (no split).",
    )

    parser.add_argument("--cost-grid", type=str, default=None)
    parser.add_argument("--cost-grid-start", type=float, default=0.0)
    parser.add_argument("--cost-grid-stop", type=float, default=0.2)
    parser.add_argument("--cost-grid-num", type=int, default=25)

    parser.add_argument("--tree-train-cost", type=float, default=0.1)
    parser.add_argument("--tree-max-depth", type=int, default=3)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=None)
    parser.add_argument("--tree-min-samples-split", type=int, default=None)
    parser.add_argument("--tree-random-state", type=int, default=None)
    parser.add_argument("--tree-honest", type=str, default=None)
    parser.add_argument("--tree-extra-kwargs", type=str, default=None)

    parser.add_argument("--no-progress", action="store_true")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    summary_path, detailed_path, config_path, _summary_df = run_experiment(args)

    print(f"Summary written to: {summary_path}")
    print(f"Detailed results written to: {detailed_path}")
    print(f"Run config written to: {config_path}")
