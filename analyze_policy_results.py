import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


BASE_REQUIRED_COLS = {
    "c",
    "pv_cates_means",
    "pv_trees_means",
    "pv_ates_means",
}

POLICY_GROUP_COLS = ["run_id", "experiment_name", "state", "evaluation_mode", "tree_train_cost"]

SD_SE_SPECS = [
    ("pv_cates_sds", "pv_cates_se"),
    ("pv_trees_sds", "pv_trees_se"),
    ("pv_ates_sds", "pv_ates_se"),
    ("pv_ates_grouped_sds", "pv_ates_grouped_se"),
    ("pv_ates_grouped_oracle_sds", "pv_ates_grouped_oracle_se"),
]


def parse_csv_arg(raw):
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts if parts else None


def load_summary_frames(paths_arg, glob_pattern):
    paths = []
    if paths_arg:
        paths.extend(parse_csv_arg(paths_arg))
    if glob_pattern:
        paths.extend(sorted(glob.glob(glob_pattern)))

    seen = set()
    unique_paths = []
    for path in paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)

    if not unique_paths:
        raise ValueError("No summary files found. Pass --summary-paths and/or --summary-glob.")

    frames = []
    for path in unique_paths:
        frame = pd.read_csv(path)
        frame["source_file"] = path
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def ensure_columns(df, args):
    missing = sorted(BASE_REQUIRED_COLS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required summary columns: {missing}")

    if "run_id" not in df.columns:
        df["run_id"] = -1
    if "experiment_name" not in df.columns:
        df["experiment_name"] = "unknown_experiment"
    if "state" not in df.columns:
        df["state"] = "unknown_state"
    if "evaluation_mode" not in df.columns:
        df["evaluation_mode"] = args.default_evaluation_mode
    if "tree_train_cost" not in df.columns:
        df["tree_train_cost"] = args.default_tree_train_cost

    if "eval_cost" not in df.columns:
        df["eval_cost"] = df["c"]

    if "pv_ates_grouped_means" not in df.columns:
        df["pv_ates_grouped_means"] = df["pv_ates_means"]
    if "pv_ates_grouped_sds" not in df.columns:
        if "pv_ates_sds" in df.columns:
            df["pv_ates_grouped_sds"] = df["pv_ates_sds"]
        else:
            df["pv_ates_grouped_sds"] = np.nan

    if "pv_ates_grouped_oracle_means" not in df.columns:
        if "pv_ates_oracle_means" in df.columns:
            df["pv_ates_grouped_oracle_means"] = df["pv_ates_oracle_means"]
        else:
            df["pv_ates_grouped_oracle_means"] = df["pv_ates_means"]
    if "pv_ates_grouped_oracle_sds" not in df.columns:
        if "pv_ates_sds" in df.columns:
            df["pv_ates_grouped_oracle_sds"] = df["pv_ates_sds"]
        else:
            df["pv_ates_grouped_oracle_sds"] = np.nan

    numeric_cols = [
        "run_id",
        "tree_train_cost",
        "c",
        "eval_cost",
        "pv_cates_means",
        "pv_trees_means",
        "pv_ates_means",
        "pv_ates_grouped_means",
        "pv_ates_grouped_sds",
        "pv_ates_grouped_oracle_means",
        "pv_ates_grouped_oracle_sds",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "cate_minus_tree_means" not in df.columns:
        df["cate_minus_tree_means"] = df["pv_cates_means"] - df["pv_trees_means"]
    # Use grouped train/test ATE as the default ATE baseline in analysis.
    df["cate_minus_ate_means"] = df["pv_cates_means"] - df["pv_ates_grouped_means"]
    df["cate_minus_best_baseline_means"] = df["pv_cates_means"] - np.maximum(
        df["pv_ates_grouped_means"], df["pv_trees_means"]
    )
    if "n_rows" not in df.columns:
        df["n_rows"] = np.nan
    for sd_col, _ in SD_SE_SPECS:
        if sd_col not in df.columns:
            df[sd_col] = np.nan

    if "cost_misspec" not in df.columns:
        df["cost_misspec"] = df["eval_cost"] - df["tree_train_cost"]
    if "abs_cost_misspec" not in df.columns:
        df["abs_cost_misspec"] = np.abs(df["cost_misspec"])

    for sd_col, _ in SD_SE_SPECS:
        df[sd_col] = pd.to_numeric(df[sd_col], errors="coerce")

    df["best_baseline"] = np.where(df["pv_ates_grouped_means"] >= df["pv_trees_means"], "ate", "tree")
    return df


def apply_row_filters(df, args):
    out = df.copy()

    states = parse_csv_arg(args.states)
    if states:
        out = out[out["state"].isin(states)]

    eval_modes = parse_csv_arg(args.evaluation_modes)
    if eval_modes:
        out = out[out["evaluation_mode"].isin(eval_modes)]

    experiment_names = parse_csv_arg(args.experiment_names)
    if experiment_names:
        out = out[out["experiment_name"].isin(experiment_names)]

    run_ids = parse_csv_arg(args.run_ids)
    if run_ids:
        run_id_values = {int(x) for x in run_ids}
        out = out[out["run_id"].isin(run_id_values)]

    if args.min_n_rows is not None:
        out = out[(out["n_rows"].isna()) | (out["n_rows"] >= args.min_n_rows)]

    return out


def collapse_duplicates(df):
    # If the same policy/cost appears multiple times (e.g., overlapping input files),
    # average metrics so ranking is not biased by duplicate rows.
    group_cols = POLICY_GROUP_COLS + ["eval_cost"]
    collapsed = (
        df.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            pv_cates_means=("pv_cates_means", "mean"),
            pv_trees_means=("pv_trees_means", "mean"),
            pv_ates_means=("pv_ates_means", "mean"),
            pv_ates_grouped_means=("pv_ates_grouped_means", "mean"),
            pv_ates_grouped_oracle_means=("pv_ates_grouped_oracle_means", "mean"),
            pv_cates_sds=("pv_cates_sds", "mean"),
            pv_trees_sds=("pv_trees_sds", "mean"),
            pv_ates_sds=("pv_ates_sds", "mean"),
            pv_ates_grouped_sds=("pv_ates_grouped_sds", "mean"),
            pv_ates_grouped_oracle_sds=("pv_ates_grouped_oracle_sds", "mean"),
            cate_minus_tree_means=("cate_minus_tree_means", "mean"),
            cate_minus_ate_means=("cate_minus_ate_means", "mean"),
            cate_minus_best_baseline_means=("cate_minus_best_baseline_means", "mean"),
            n_rows=("n_rows", "mean"),
            source_file=("source_file", "first"),
            duplicate_count=("source_file", "size"),
        )
    )
    collapsed["cost_misspec"] = collapsed["eval_cost"] - collapsed["tree_train_cost"]
    collapsed["abs_cost_misspec"] = np.abs(collapsed["cost_misspec"])
    collapsed["best_baseline"] = np.where(
        collapsed["pv_ates_grouped_means"] >= collapsed["pv_trees_means"], "ate", "tree"
    )
    n_rows_eff = pd.to_numeric(collapsed["n_rows"], errors="coerce") * pd.to_numeric(
        collapsed["duplicate_count"], errors="coerce"
    ).fillna(1.0)
    collapsed["n_rows_eff"] = np.where(n_rows_eff > 0, n_rows_eff, np.nan)
    for sd_col, se_col in SD_SE_SPECS:
        collapsed[se_col] = collapsed[sd_col] / np.sqrt(collapsed["n_rows_eff"])
    return collapsed


def compute_policy_rankings(cost_rows, args):
    df = cost_rows.copy()

    df["cate_positive"] = (df["pv_cates_means"] >= args.cate_positive_threshold).astype(float)
    df["ate_positive"] = (df["pv_ates_grouped_means"] >= args.ate_positive_threshold).astype(float)
    df["joint_positive"] = (
        (df["pv_cates_means"] >= args.cate_positive_threshold)
        & (df["pv_ates_grouped_means"] >= args.ate_positive_threshold)
    ).astype(float)

    sweep = (
        df.groupby(POLICY_GROUP_COLS, as_index=False, dropna=False)
        .agg(
            n_cost_points=("eval_cost", "size"),
            eval_cost_min=("eval_cost", "min"),
            eval_cost_max=("eval_cost", "max"),
            sum_adv_vs_tree=("cate_minus_tree_means", "sum"),
            sum_adv_vs_ate=("cate_minus_ate_means", "sum"),
            sum_adv_vs_best=("cate_minus_best_baseline_means", "sum"),
            mean_adv_vs_tree=("cate_minus_tree_means", "mean"),
            mean_adv_vs_ate=("cate_minus_ate_means", "mean"),
            mean_adv_vs_best=("cate_minus_best_baseline_means", "mean"),
            cate_positive_share=("cate_positive", "mean"),
            ate_positive_share=("ate_positive", "mean"),
            joint_positive_share=("joint_positive", "mean"),
            duplicate_count_sum=("duplicate_count", "sum"),
        )
    )

    target_rows = df.copy()
    target_rows["target_cost_distance"] = np.abs(target_rows["eval_cost"] - target_rows["tree_train_cost"])
    target_rows = (
        target_rows.sort_values(POLICY_GROUP_COLS + ["target_cost_distance", "eval_cost"])
        .groupby(POLICY_GROUP_COLS, as_index=False, dropna=False)
        .head(1)
    )

    target_rows = target_rows[
        POLICY_GROUP_COLS
        + [
            "eval_cost",
            "target_cost_distance",
            "pv_trees_means",
            "pv_cates_means",
            "pv_ates_grouped_means",
            "cate_minus_tree_means",
            "cate_minus_ate_means",
            "cate_minus_best_baseline_means",
        ]
    ].rename(
        columns={
            "eval_cost": "tree_target_eval_cost",
            "pv_trees_means": "tree_value_at_target_cost",
            "pv_cates_means": "cate_value_at_target_cost",
            "pv_ates_grouped_means": "ate_value_at_target_cost",
            "cate_minus_tree_means": "adv_vs_tree_at_target_cost",
            "cate_minus_ate_means": "adv_vs_ate_at_target_cost",
            "cate_minus_best_baseline_means": "adv_vs_best_at_target_cost",
        }
    )

    ranking = sweep.merge(target_rows, on=POLICY_GROUP_COLS, how="left")
    ranking["combined_sweep_advantage"] = ranking["sum_adv_vs_tree"] + ranking["sum_adv_vs_ate"]

    ranking["passes_nontriviality"] = (
        (ranking["cate_positive_share"] >= args.min_cate_positive_share)
        & (ranking["ate_positive_share"] >= args.min_ate_positive_share)
        & (ranking["joint_positive_share"] >= args.min_joint_positive_share)
        & (ranking["tree_value_at_target_cost"] >= args.tree_opt_positive_threshold)
        & (ranking["n_cost_points"] >= args.min_cost_points)
    )

    ranking["nontriviality_score"] = (
        ranking["cate_positive_share"]
        + ranking["ate_positive_share"]
        + ranking["joint_positive_share"]
    ) / 3.0

    ranking["ranking_score"] = ranking["combined_sweep_advantage"]
    if args.nontriviality_penalty > 0:
        ranking.loc[~ranking["passes_nontriviality"], "ranking_score"] = (
            ranking.loc[~ranking["passes_nontriviality"], "ranking_score"] - args.nontriviality_penalty
        )

    ranking["policy_id"] = (
        "run="
        + ranking["run_id"].fillna(-1).astype(int).astype(str)
        + "|exp="
        + ranking["experiment_name"].astype(str)
        + "|state="
        + ranking["state"].astype(str)
        + "|mode="
        + ranking["evaluation_mode"].astype(str)
        + "|tree_cost="
        + ranking["tree_train_cost"].map(lambda x: f"{x:.6f}")
    )

    ranking = ranking.sort_values(
        [
            "passes_nontriviality",
            "ranking_score",
            "combined_sweep_advantage",
            "sum_adv_vs_tree",
            "sum_adv_vs_ate",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)

    return ranking


def select_top_cost_paths(cost_rows, ranking, top_k):
    top_policy_ids = ranking.head(top_k)["policy_id"].tolist()
    if not top_policy_ids:
        return cost_rows.iloc[0:0].copy()

    keyed = cost_rows.copy()
    keyed["policy_id"] = (
        "run="
        + keyed["run_id"].fillna(-1).astype(int).astype(str)
        + "|exp="
        + keyed["experiment_name"].astype(str)
        + "|state="
        + keyed["state"].astype(str)
        + "|mode="
        + keyed["evaluation_mode"].astype(str)
        + "|tree_cost="
        + keyed["tree_train_cost"].map(lambda x: f"{x:.6f}")
    )
    out = keyed[keyed["policy_id"].isin(top_policy_ids)].copy()
    out = out.merge(
        ranking[["policy_id", "rank", "passes_nontriviality", "ranking_score"]],
        on="policy_id",
        how="left",
    )
    return out.sort_values(["rank", "eval_cost"])


def plot_policy_curves(cost_paths, ranking, top_n, output_path, title_prefix):
    if top_n <= 0 or ranking.empty or cost_paths.empty:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or run without --plot-top-nontrivial."
        ) from exc

    top_ranked = ranking.head(top_n).copy()
    policy_ids = top_ranked["policy_id"].tolist()
    plot_df = cost_paths[cost_paths["policy_id"].isin(policy_ids)].copy()
    if plot_df.empty:
        return None

    n_panels = len(policy_ids)
    fig, axes = plt.subplots(n_panels, 1, figsize=(5, 3.8 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, policy_id in zip(axes, policy_ids):
        series = plot_df[plot_df["policy_id"] == policy_id].sort_values("eval_cost")
        if series.empty:
            continue

        rank_row = top_ranked[top_ranked["policy_id"] == policy_id].iloc[0]
        x = pd.to_numeric(series["eval_cost"], errors="coerce").to_numpy(dtype=float)
        baseline_mean = pd.to_numeric(series["pv_ates_grouped_means"], errors="coerce").to_numpy(dtype=float)
        baseline_se = (
            pd.to_numeric(series["pv_ates_grouped_se"], errors="coerce").to_numpy(dtype=float)
            if "pv_ates_grouped_se" in series.columns
            else np.full_like(baseline_mean, np.nan)
        )
        plot_specs = [
            ("pv_cates_means", "pv_cates_se", "CATE+postprocess", "#1f77b4"),
            ("pv_trees_means", "pv_trees_se", "Policy tree", "#ff7f0e"),
        ]
        for mean_col, se_col, label, color in plot_specs:
            y_raw = pd.to_numeric(series[mean_col], errors="coerce").to_numpy(dtype=float)
            y = y_raw - baseline_mean
            ax.plot(
                x,
                y,
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=color,
                label=label,
            )
            if se_col in series.columns:
                current_se = pd.to_numeric(series[se_col], errors="coerce").to_numpy(dtype=float)
                # Approximate SE for differences assuming independence.
                se = np.where(
                    np.isfinite(baseline_se),
                    np.sqrt(np.square(current_se) + np.square(baseline_se)),
                    current_se,
                )

                valid = np.isfinite(y) & np.isfinite(se)
                if valid.any():
                    lower = y - 1.96 * se
                    upper = y + 1.96 * se
                    ax.fill_between(x[valid], lower[valid], upper[valid], color=color, alpha=0.15)

        ax.axvline(rank_row["tree_train_cost"], color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
        ax.set_ylabel("Delta policy value vs grouped ATE")
        ax.grid(alpha=0.25)
        run_id_val = int(rank_row["run_id"]) if pd.notna(rank_row["run_id"]) else -1
        exp_val = str(rank_row["experiment_name"])
        state_val = str(rank_row["state"])
        mode_val = str(rank_row["evaluation_mode"])
        nontrivial_val = bool(rank_row["passes_nontriviality"])
        joint_pos_val = float(rank_row["joint_positive_share"])
        tree_target_val = float(rank_row["tree_value_at_target_cost"])
        ax.set_title(
            f"Rank {int(rank_row['rank'])} | mode={mode_val} | state={state_val} | exp={exp_val}\n"
            f"run={run_id_val} | nontrivial={nontrivial_val} | joint_pos={joint_pos_val:.2f} | "
            f"tree@target={tree_target_val:.4f}"
        )

    axes[-1].set_xlabel("Treatment cost")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, min(len(labels), 3)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(title_prefix, y=1.04, fontsize=12)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Rank fixed-tree-cost policies by integrated sweep advantage and non-triviality criteria."
    )
    parser.add_argument(
        "--summary-glob",
        type=str,
        default="results/policy_learning_results_*.csv",
        help="Glob for summary CSV files.",
    )
    parser.add_argument(
        "--summary-paths",
        type=str,
        default=None,
        help="Comma-separated explicit summary CSV paths.",
    )
    parser.add_argument("--output-dir", type=str, default="results/analysis")
    parser.add_argument("--output-prefix", type=str, default="policy_sweep")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--top-k-cost-paths", type=int, default=10)

    parser.add_argument("--states", type=str, default=None)
    parser.add_argument("--evaluation-modes", type=str, default=None)
    parser.add_argument("--experiment-names", type=str, default=None)
    parser.add_argument("--run-ids", type=str, default=None)
    parser.add_argument("--min-n-rows", type=int, default=1)

    parser.add_argument("--default-tree-train-cost", type=float, default=0.1)
    parser.add_argument("--default-evaluation-mode", type=str, default="holdout")

    parser.add_argument("--cate-positive-threshold", type=float, default=0.0)
    parser.add_argument("--ate-positive-threshold", type=float, default=0.0)
    parser.add_argument("--tree-opt-positive-threshold", type=float, default=0.0)
    parser.add_argument("--min-cate-positive-share", type=float, default=0.8)
    parser.add_argument("--min-ate-positive-share", type=float, default=0.8)
    parser.add_argument("--min-joint-positive-share", type=float, default=0.8)
    parser.add_argument("--min-cost-points", type=int, default=5)
    parser.add_argument("--nontriviality-penalty", type=float, default=0.0)
    parser.add_argument("--nontrivial-only", action="store_true")
    parser.add_argument("--plot-top-nontrivial", type=int, default=3)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    summary = load_summary_frames(args.summary_paths, args.summary_glob)
    summary = ensure_columns(summary, args)
    summary = apply_row_filters(summary, args)
    if summary.empty:
        raise RuntimeError("No summary rows remain after filtering.")

    cost_rows = collapse_duplicates(summary)
    rankings = compute_policy_rankings(cost_rows, args)

    if args.nontrivial_only:
        rankings = rankings[rankings["passes_nontriviality"]].copy()
        rankings = rankings.reset_index(drop=True)
        rankings["rank"] = np.arange(1, len(rankings) + 1)

    if rankings.empty:
        raise RuntimeError("No ranked policies remain after applying constraints.")

    top_rankings = rankings.head(args.top_k).copy()
    nontrivial_rankings = rankings[rankings["passes_nontriviality"]].copy()
    top_nontrivial = nontrivial_rankings.head(args.top_k).copy()
    top_cost_paths = select_top_cost_paths(cost_rows, top_rankings, args.top_k_cost_paths)
    top_nontrivial_cost_paths = select_top_cost_paths(cost_rows, top_nontrivial, args.top_k_cost_paths)

    out_dir = Path(args.output_dir)
    prefix = args.output_prefix

    top_overall_plot_path = out_dir / f"{prefix}_top_overall_policies.png"
    top_nontrivial_plot_path = out_dir / f"{prefix}_top_nontrivial_policies.png"

    if args.nontrivial_only:
        plot_ranking = top_nontrivial
        plot_cost_paths = top_nontrivial_cost_paths
        plot_output_path = top_nontrivial_plot_path
        plot_title = f"Top {args.plot_top_nontrivial} non-trivial policies"
    else:
        plot_ranking = top_rankings
        plot_cost_paths = top_cost_paths
        plot_output_path = top_overall_plot_path
        plot_title = f"Top {args.plot_top_nontrivial} overall policies"

    plot_policy_curves(
        cost_paths=plot_cost_paths,
        ranking=plot_ranking,
        top_n=args.plot_top_nontrivial,
        output_path=plot_output_path,
        title_prefix=plot_title,
    )

    print(f"Filtered input rows: {len(summary)}")
    print(f"Unique policy-cost rows: {len(cost_rows)}")
    print(f"Ranked policies: {len(rankings)}")

if __name__ == "__main__":
    main()
