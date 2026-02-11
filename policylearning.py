import argparse
import inspect
import json
from pathlib import Path

from econml.policy import DRPolicyTree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DEFAULT_DATA_PATH = "gerber/gerber_generalizability_2/PublicReplicationData.dta"
DEFAULT_OUTPUT_DIR = "results"
EVAL_MODE_CHOICES = {"holdout", "full", "train"}

FEATURE_PRESETS = {
    "vote4": ["voted06", "voted08", "voted10", "voted12"],
    "race_gender": ["d_race_b", "d_race_h", "d_race_o", "d_race_w", "d_female", "d_notfem"],
    "vote_history": ["voted06", "voted08", "voted09", "voted10", "voted11", "voted12", "voted13"],
    "extended": [
        "voted06", "voted08", "voted09", "voted10", "voted11", "voted12", "voted13", "i_age",
        "age_miss", "age2", "flag_hhid_mult_hhid", "flag_hhid_mult_z", "flag_drop_hhid",
        "vote_hist", "state_median", "vh_stratum", "vhblw", "vhavg", "vhabv", "d_married",
        "d_unmarried", "d_hhsize1", "d_hhsize2", "d_hhsize3", "d_hhsize4",
    ],
}


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


def get_ate(df, treatment_var="treat", outcome_var="voted14"):
    treat_mask = df[treatment_var] == 1
    control_mask = df[treatment_var] == 0

    n_treat = int(treat_mask.sum())
    n_control = int(control_mask.sum())

    mean_treat = float(df.loc[treat_mask, outcome_var].mean()) if n_treat > 0 else np.nan
    mean_control = float(df.loc[control_mask, outcome_var].mean()) if n_control > 0 else np.nan
    ate = mean_treat - mean_control if n_treat > 0 and n_control > 0 else np.nan

    var_treat = float(df.loc[treat_mask, outcome_var].var(ddof=1)) if n_treat > 1 else np.nan
    var_control = float(df.loc[control_mask, outcome_var].var(ddof=1)) if n_control > 1 else np.nan

    if n_treat > 1 and n_control > 1 and np.isfinite(var_treat) and np.isfinite(var_control):
        se = float(np.sqrt(var_treat / n_treat + var_control / n_control))
    else:
        se = np.nan

    return {
        "mean_treatment": mean_treat,
        "mean_control": mean_control,
        "ate": ate,
        "se": se,
        "n_treat": n_treat,
        "n_control": n_control,
    }


def compute_cate(df, group_vars, outcome_var="voted14", treat_var="treat"):
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
    train_assignments["recommend_treat"] = (train_assignments["ate"] - cost) > 0
    train_assignments["recommend_treat"] = train_assignments["recommend_treat"].fillna(False)

    aligned = eval_cates.merge(
        train_assignments[group_vars + ["recommend_treat"]],
        on=group_vars,
        how="left",
    )
    aligned["recommend_treat"] = aligned["recommend_treat"].fillna(False)
    aligned["net_benefit"] = aligned["ate"] - cost
    aligned["net_benefit"] = aligned["net_benefit"].fillna(0.0)

    subgroup_sizes = aligned["n_treat"] + aligned["n_control"]
    total_size = float(subgroup_sizes.sum())
    if total_size <= 0:
        return np.nan

    weighted_net_benefit = (
        aligned["net_benefit"] * aligned["recommend_treat"].astype(float) * subgroup_sizes
    ).sum()
    return float(weighted_net_benefit / total_size)


def get_policy_tree(df, policy_features, cost, outcome_var, treat_var, tree_kwargs):
    X = df[policy_features]
    T = df[treat_var].astype("category")
    Y = df[outcome_var] - cost

    policy_tree = DRPolicyTree(**tree_kwargs)
    policy_tree.fit(Y, T, X=X)
    return policy_tree


def get_leaf_cates(policy_tree, df, policy_features, outcome_var="voted14", treat_var="treat"):
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
    leaf_recommendations = (
        leaf_df.groupby("leaf_id", as_index=False)["recommended_treatment"]
        .first()
    )
    leaf_cates = leaf_cates.merge(leaf_recommendations, on="leaf_id", how="left")
    return leaf_cates


def policy_value_tree(leaf_cates, cost):
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


def resolve_features(args, df_columns):
    if args.cate_features:
        cate_features = parse_csv_arg(args.cate_features)
    else:
        cate_features = FEATURE_PRESETS[args.cate_feature_set]

    if args.policy_features:
        policy_features = parse_csv_arg(args.policy_features)
    elif args.policy_feature_set:
        policy_features = FEATURE_PRESETS[args.policy_feature_set]
    else:
        policy_features = list(cate_features)

    required = set(cate_features + policy_features + [args.state_var, args.treat_var, args.outcome_var])
    missing = sorted(col for col in required if col not in df_columns)
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    return cate_features, policy_features


def resolve_cost_grid(args):
    if args.cost_grid:
        values = [float(v) for v in parse_csv_arg(args.cost_grid)]
        if not values:
            raise ValueError("--cost-grid was provided but no values were parsed.")
        return np.array(values, dtype=float)
    return np.linspace(args.cost_grid_start, args.cost_grid_stop, args.cost_grid_num)


def resolve_tree_train_costs(args):
    values = parse_csv_arg(args.tree_train_costs)
    if not values:
        return [args.tree_train_cost]
    return [float(v) for v in values]


def resolve_eval_modes(args):
    modes = parse_csv_arg(args.evaluation_modes)
    if not modes:
        raise ValueError("At least one evaluation mode must be provided.")

    unknown = sorted(set(modes) - EVAL_MODE_CHOICES)
    if unknown:
        raise ValueError(f"Unknown evaluation modes: {unknown}. Allowed: {sorted(EVAL_MODE_CHOICES)}")
    return modes


def resolve_tree_kwargs(args):
    candidate_kwargs = {
        "max_depth": args.tree_max_depth,
        "min_samples_leaf": args.tree_min_samples_leaf,
        "min_samples_split": args.tree_min_samples_split,
        "random_state": args.tree_random_state,
    }

    honest_value = parse_bool_or_none(args.tree_honest)
    if honest_value is not None:
        candidate_kwargs["honest"] = honest_value

    if args.tree_extra_kwargs:
        extras = json.loads(args.tree_extra_kwargs)
        if not isinstance(extras, dict):
            raise ValueError("--tree-extra-kwargs must parse to a JSON object.")
        candidate_kwargs.update(extras)

    signature = inspect.signature(DRPolicyTree)
    params = signature.parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())

    allowed = set(params.keys())
    allowed.discard("self")

    if accepts_kwargs:
        filtered = {k: v for k, v in candidate_kwargs.items() if v is not None}
        ignored = []
    else:
        filtered = {k: v for k, v in candidate_kwargs.items() if v is not None and k in allowed}
        ignored = sorted(k for k, v in candidate_kwargs.items() if v is not None and k not in allowed)
    return filtered, ignored


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


def append_frame(df, path_str):
    if not path_str:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def select_states(df, args):
    requested_states = parse_csv_arg(args.states)
    use_all_states = (
        requested_states
        and len(requested_states) == 1
        and requested_states[0].strip().upper() == "ALL"
    )
    if requested_states and not use_all_states:
        filtered = df[df[args.state_var].isin(requested_states)].copy()
    else:
        filtered = df.copy()
    states = sorted(filtered[args.state_var].dropna().unique().tolist())
    return filtered, states


def run_experiment(args):
    df = pd.read_stata(args.data_path)
    if args.subset_query:
        df = df.query(args.subset_query).copy()

    df, states = select_states(df, args)
    if not states:
        raise ValueError("No states available after filtering.")

    cate_features, policy_features = resolve_features(args, set(df.columns))
    eval_modes = resolve_eval_modes(args)
    costs = resolve_cost_grid(args)
    tree_train_costs = resolve_tree_train_costs(args)
    tree_kwargs, ignored_tree_kwargs = resolve_tree_kwargs(args)

    if not 0 < args.train_size < 1:
        raise ValueError("--train-size must be strictly between 0 and 1.")
    if args.n_reps <= 0:
        raise ValueError("--n-reps must be positive.")

    run_id = 0 if args.run_id is None else args.run_id
    base_seed = args.seed if args.seed is not None else 1729 + (run_id * 1009)
    rng = np.random.default_rng(base_seed)

    detailed_rows = []
    progress_disable = args.no_progress

    for state in states:
        df_state = df[df[args.state_var] == state].copy()
        if df_state.shape[0] < 2:
            continue

        rep_iterator = tqdm(
            range(args.n_reps),
            desc=f"state={state}",
            disable=progress_disable,
        )

        for rep in rep_iterator:
            split_seed = int(rng.integers(0, 2**31 - 1))

            train_df, holdout_df = train_test_split(
                df_state,
                train_size=args.train_size,
                shuffle=True,
                random_state=split_seed,
            )

            train_df = train_df.copy()
            holdout_df = holdout_df.copy()

            train_ate = get_ate(train_df, treatment_var=args.treat_var, outcome_var=args.outcome_var)["ate"]
            train_cates = compute_cate(
                train_df,
                cate_features,
                outcome_var=args.outcome_var,
                treat_var=args.treat_var,
            )

            trained_trees = {}
            tree_errors = {}
            for train_cost in tree_train_costs:
                try:
                    trained_trees[train_cost] = get_policy_tree(
                        train_df,
                        policy_features=policy_features,
                        cost=train_cost,
                        outcome_var=args.outcome_var,
                        treat_var=args.treat_var,
                        tree_kwargs=tree_kwargs,
                    )
                    tree_errors[train_cost] = None
                except Exception as exc:
                    trained_trees[train_cost] = None
                    tree_errors[train_cost] = str(exc)

            eval_sets = {}
            if "holdout" in eval_modes:
                eval_sets["holdout"] = holdout_df
            if "full" in eval_modes:
                eval_sets["full"] = df_state
            if "train" in eval_modes:
                eval_sets["train"] = train_df

            for eval_mode, eval_df in eval_sets.items():
                eval_ate = get_ate(eval_df, treatment_var=args.treat_var, outcome_var=args.outcome_var)["ate"]
                eval_cates = compute_cate(
                    eval_df,
                    cate_features,
                    outcome_var=args.outcome_var,
                    treat_var=args.treat_var,
                )

                leaf_cates_by_cost = {}
                for train_cost in tree_train_costs:
                    model = trained_trees[train_cost]
                    if model is None:
                        leaf_cates_by_cost[train_cost] = None
                    else:
                        try:
                            leaf_cates_by_cost[train_cost] = get_leaf_cates(
                                model,
                                eval_df,
                                policy_features=policy_features,
                                outcome_var=args.outcome_var,
                                treat_var=args.treat_var,
                            )
                        except Exception as exc:
                            leaf_cates_by_cost[train_cost] = None
                            if tree_errors[train_cost] is None:
                                tree_errors[train_cost] = str(exc)

                for c in costs:
                    pv_cate = policy_value_cate(
                        train_cates=train_cates,
                        eval_cates=eval_cates,
                        group_vars=cate_features,
                        cost=c,
                    )
                    pv_ate = policy_value_ate(train_ate=train_ate, eval_ate=eval_ate, cost=c)

                    for train_cost in tree_train_costs:
                        leaf_cates = leaf_cates_by_cost[train_cost]
                        pv_tree = policy_value_tree(leaf_cates, cost=c) if leaf_cates is not None else np.nan

                        detailed_rows.append(
                            {
                                "run_id": args.run_id,
                                "experiment_name": args.experiment_name,
                                "rep": rep,
                                "split_seed": split_seed,
                                "state": state,
                                "evaluation_mode": eval_mode,
                                "c": float(c),
                                "eval_cost": float(c),
                                "tree_train_cost": float(train_cost),
                                "cost_misspec": float(c - train_cost),
                                "abs_cost_misspec": float(abs(c - train_cost)),
                                "pv_cate": pv_cate,
                                "pv_tree": pv_tree,
                                "pv_ate": pv_ate,
                                "train_ate": train_ate,
                                "eval_ate": eval_ate,
                                "n_train": int(train_df.shape[0]),
                                "n_eval": int(eval_df.shape[0]),
                                "tree_error": tree_errors[train_cost],
                            }
                        )

    detailed = pd.DataFrame(detailed_rows)
    if detailed.empty:
        raise RuntimeError("No results were produced; check filters and arguments.")

    summary = (
        detailed.groupby(
            ["run_id", "experiment_name", "state", "evaluation_mode", "c", "tree_train_cost"],
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
            n_rows=("pv_cate", "size"),
            n_tree_errors=("tree_error", lambda s: int(s.notna().sum())),
        )
    )

    summary["cate_minus_ate_means"] = summary["pv_cates_means"] - summary["pv_ates_means"]
    summary["cate_minus_tree_means"] = summary["pv_cates_means"] - summary["pv_trees_means"]
    summary["cate_minus_best_baseline_means"] = summary["pv_cates_means"] - np.maximum(
        summary["pv_ates_means"], summary["pv_trees_means"]
    )
    summary["eval_cost"] = summary["c"]
    summary["cost_misspec"] = summary["c"] - summary["tree_train_cost"]
    summary["abs_cost_misspec"] = np.abs(summary["cost_misspec"])

    summary_path, detailed_path, config_path = default_output_paths(args)
    summary.to_csv(summary_path, index=False)
    detailed.to_csv(detailed_path, index=False)
    append_frame(summary, args.append_summary_path)
    append_frame(detailed, args.append_detailed_path)

    config_payload = vars(args).copy()
    config_payload["resolved_cate_features"] = cate_features
    config_payload["resolved_policy_features"] = policy_features
    config_payload["resolved_eval_modes"] = eval_modes
    config_payload["resolved_cost_grid"] = [float(x) for x in costs.tolist()]
    config_payload["resolved_tree_train_costs"] = [float(x) for x in tree_train_costs]
    config_payload["resolved_tree_kwargs"] = tree_kwargs
    config_payload["ignored_tree_kwargs"] = ignored_tree_kwargs
    config_payload["resolved_states"] = states
    config_payload["seed_used"] = base_seed

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, sort_keys=True)

    return summary_path, detailed_path, config_path, summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Policy learning experiment runner with configurable sweeps and split/full evaluation."
    )

    parser.add_argument("-i", "--run-id", type=int, required=False, dest="run_id")
    parser.add_argument("--experiment-name", type=str, default="policy_learning")

    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument("--detailed-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--append-summary-path", type=str, default=None)
    parser.add_argument("--append-detailed-path", type=str, default=None)

    parser.add_argument("--state-var", type=str, default="state")
    parser.add_argument("--states", type=str, default="TX")
    parser.add_argument("--subset-query", type=str, default=None)

    parser.add_argument("--treat-var", type=str, default="treat")
    parser.add_argument("--outcome-var", type=str, default="voted14")

    parser.add_argument("--cate-feature-set", type=str, choices=sorted(FEATURE_PRESETS.keys()), default="vote4")
    parser.add_argument("--cate-features", type=str, default=None)
    parser.add_argument("--policy-feature-set", type=str, choices=sorted(FEATURE_PRESETS.keys()), default=None)
    parser.add_argument("--policy-features", type=str, default=None)

    parser.add_argument("--n-reps", type=int, default=500)
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--evaluation-modes", type=str, default="holdout,full")

    parser.add_argument("--cost-grid", type=str, default=None)
    parser.add_argument("--cost-grid-start", type=float, default=0.0)
    parser.add_argument("--cost-grid-stop", type=float, default=0.2)
    parser.add_argument("--cost-grid-num", type=int, default=25)

    parser.add_argument("--tree-train-cost", type=float, default=0.1)
    parser.add_argument("--tree-train-costs", type=str, default=None)
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
    summary_path, detailed_path, config_path, summary_df = run_experiment(args)

    print(f"Summary written to: {summary_path}")
    print(f"Detailed results written to: {detailed_path}")
    print(f"Run config written to: {config_path}")

    best_rows = summary_df.sort_values("cate_minus_best_baseline_means", ascending=False).head(10)
    print("\nTop rows by CATE advantage over best baseline (ATE or tree):")
    print(
        best_rows[
            [
                "state",
                "evaluation_mode",
                "c",
                "tree_train_cost",
                "abs_cost_misspec",
                "pv_cates_means",
                "pv_trees_means",
                "pv_ates_means",
                "cate_minus_best_baseline_means",
            ]
        ].to_string(index=False)
    )
