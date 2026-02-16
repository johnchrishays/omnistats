import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REQUIRED_SUMMARY_COLUMNS = {
    "dataset",
    "run_id",
    "experiment_name",
    "state",
    "evaluation_mode",
    "c",
    "tree_train_cost",
    "pv_cates_means",
    "pv_cates_sds",
    "pv_trees_means",
    "pv_trees_sds",
    "pv_ates_grouped_means",
    "pv_ates_grouped_sds",
    "n_rows",
}

STUDY_TITLES = {
    "gerber": "Social Pressure",
    "nsw": "National Supported Work Demonstration",
    "jtpa": "Job Training Partnership Act",
}

# Keep seaborn's default color cycle accessible for reuse in future plots.
DEFAULT_PALETTE = sns.color_palette("Set2")


def parse_csv_arg(raw):
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values if values else None


def configure_plot_style():
    sns.set_theme(style="whitegrid", palette=DEFAULT_PALETTE)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        }
    )


def resolve_summary_paths(path_arg, glob_arg):
    explicit_paths = parse_csv_arg(path_arg) or []
    glob_patterns = parse_csv_arg(glob_arg) or []

    resolved = []
    for path_str in explicit_paths:
        resolved.append(Path(path_str))
    for pattern in glob_patterns:
        matches = [Path(p) for p in sorted(glob.glob(pattern))]
        if not matches:
            raise FileNotFoundError(f"--summary-glob matched no files: {pattern}")
        resolved.extend(matches)

    if not resolved:
        raise ValueError("Provide --summary-paths and/or --summary-glob with at least one file.")

    unique = []
    seen = set()
    for path in resolved:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def load_summaries(path_arg, glob_arg):
    paths = resolve_summary_paths(path_arg, glob_arg)

    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Summary CSV not found: {path}")
        frame = pd.read_csv(path)
        frame["_source_path"] = str(path)
        frames.append(frame)

    summary = pd.concat(frames, ignore_index=True)
    missing = sorted(REQUIRED_SUMMARY_COLUMNS - set(summary.columns))
    if missing:
        raise ValueError(
            "Summary CSV is missing required columns: "
            f"{missing}. Please regenerate summaries with policylearning.py."
        )
    return summary, paths


def apply_filters(df, args):
    out = df.copy()

    datasets = parse_csv_arg(args.datasets)
    if datasets:
        out = out[out["dataset"].astype(str).isin(datasets)].copy()

    states = parse_csv_arg(args.states)
    if states:
        out = out[out["state"].astype(str).isin(states)].copy()

    evaluation_modes = parse_csv_arg(args.evaluation_modes)
    if evaluation_modes:
        out = out[out["evaluation_mode"].astype(str).isin(evaluation_modes)].copy()

    run_ids = parse_csv_arg(args.run_ids)
    if run_ids:
        run_ids_int = []
        for value in run_ids:
            try:
                run_ids_int.append(int(value))
            except ValueError as exc:
                raise ValueError(f"Could not parse run id '{value}' as int.") from exc
        out = out[out["run_id"].isin(run_ids_int)].copy()

    return out


def _combine_mean_sd(group, mean_col, sd_col):
    n = pd.to_numeric(group["n_rows"], errors="coerce").to_numpy(dtype=float)
    means = pd.to_numeric(group[mean_col], errors="coerce").to_numpy(dtype=float)
    sds = pd.to_numeric(group[sd_col], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(n) & (n > 0) & np.isfinite(means)
    if not np.any(valid):
        return np.nan, np.nan

    n = n[valid]
    means = means[valid]
    sds = sds[valid]
    sds = np.where(np.isfinite(sds), sds, 0.0)

    total_n = float(n.sum())
    pooled_mean = float(np.sum(n * means) / total_n)
    pooled_second_moment = float(np.sum(n * (sds**2 + means**2)) / total_n)
    pooled_var = max(pooled_second_moment - pooled_mean**2, 0.0)
    pooled_sd = float(np.sqrt(pooled_var))
    return pooled_mean, pooled_sd


def collapse_summary_rows(df):
    key_cols = ["dataset", "experiment_name", "state", "evaluation_mode", "c", "tree_train_cost"]
    metric_cols = [
        ("pv_cates_means", "pv_cates_sds"),
        ("pv_trees_means", "pv_trees_sds"),
        ("pv_ates_grouped_means", "pv_ates_grouped_sds"),
    ]

    rows = []
    grouped = df.groupby(key_cols, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(key_cols, keys)}

        n_vals = pd.to_numeric(group["n_rows"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        row["n_rows"] = int(np.round(np.sum(np.where(np.isfinite(n_vals), n_vals, 0.0))))

        if "n_tree_errors" in group.columns:
            tree_err = pd.to_numeric(group["n_tree_errors"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            row["n_tree_errors"] = int(np.round(np.sum(np.where(np.isfinite(tree_err), tree_err, 0.0))))

        for mean_col, sd_col in metric_cols:
            pooled_mean, pooled_sd = _combine_mean_sd(group, mean_col=mean_col, sd_col=sd_col)
            row[mean_col] = pooled_mean
            row[sd_col] = pooled_sd

        rows.append(row)

    collapsed = pd.DataFrame(rows)
    if collapsed.empty:
        return collapsed
    collapsed = collapsed.sort_values(
        ["dataset", "experiment_name", "state", "evaluation_mode", "tree_train_cost", "c"],
        ignore_index=True,
    )
    return collapsed


def add_standard_errors(df):
    out = df.copy()
    n = out["n_rows"].astype(float)
    sqrt_n = np.sqrt(np.where(n > 0, n, np.nan))
    out["pv_cates_se"] = out["pv_cates_sds"] / sqrt_n
    out["pv_trees_se"] = out["pv_trees_sds"] / sqrt_n
    out["pv_ates_grouped_se"] = out["pv_ates_grouped_sds"] / sqrt_n
    return out


def add_plot_columns(df, y_mode):
    out = df.copy()

    if y_mode == "relative":
        out["plot_cate"] = out["pv_cates_means"] - out["pv_ates_grouped_means"]
        out["plot_tree"] = out["pv_trees_means"] - out["pv_ates_grouped_means"]
        out["plot_ate"] = 0.0

        out["plot_cate_se"] = np.sqrt(out["pv_cates_se"] ** 2 + out["pv_ates_grouped_se"] ** 2)
        out["plot_tree_se"] = np.sqrt(out["pv_trees_se"] ** 2 + out["pv_ates_grouped_se"] ** 2)
        out["plot_ate_se"] = 0.0
    else:
        out["plot_cate"] = out["pv_cates_means"]
        out["plot_tree"] = out["pv_trees_means"]
        out["plot_ate"] = out["pv_ates_grouped_means"]

        out["plot_cate_se"] = out["pv_cates_se"]
        out["plot_tree_se"] = out["pv_trees_se"]
        out["plot_ate_se"] = out["pv_ates_grouped_se"]

    return out


def _plot_band(ax, x, y, se, color, alpha=0.18):
    lower = y - se
    upper = y + se
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0)


def _slugify(name):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    slug = slug.strip("_")
    return slug or "dataset"


def _study_title(dataset_name):
    key = str(dataset_name).strip().lower()
    return STUDY_TITLES.get(key, str(dataset_name))


def make_dataset_plot(rows, dataset_name, y_mode, output_path, dpi):
    modes = [("holdout", "Holdout"), ("full", "Full Information")]
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.2), sharey=True)

    y_label = "Policy Value" if y_mode == "absolute" else "Policy Value Difference"

    for ax, (mode, panel_label) in zip(axes, modes):
        sub = rows[rows["evaluation_mode"] == mode].sort_values("c")
        ax.set_title(panel_label)
        ax.set_xlabel(f"Treatment cost")
        ax.grid(alpha=0.22)

        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.axhline(0.0, color="#adb5bd", linewidth=1.0)
            continue

        train_costs = sorted(pd.Series(sub["tree_train_cost"]).dropna().unique().tolist())
        if len(train_costs) > 1:
            raise ValueError(
                f"Dataset '{dataset_name}' mode '{mode}' has multiple tree_train_cost values. "
                "Expected exactly one parameter setting."
            )

        x = sub["c"].to_numpy(dtype=float)
        cate_y = sub["plot_cate"].to_numpy(dtype=float)
        tree_y = sub["plot_tree"].to_numpy(dtype=float)
        ate_y = sub["plot_ate"].to_numpy(dtype=float)

        cate_se = sub["plot_cate_se"].to_numpy(dtype=float)
        tree_se = sub["plot_tree_se"].to_numpy(dtype=float)
        ate_se = sub["plot_ate_se"].to_numpy(dtype=float)

        if train_costs:
            ax.axvline(
                float(train_costs[0]),
                color="gray",
                linestyle="--",
                linewidth=1.3,
                label="Target cost",
            )
        ax.axhline(0.0, color="black", linewidth=1.0)

        ax.plot(x, cate_y, color=DEFAULT_PALETTE[0], linewidth=2.2, label="CATE + postprocess")
        _plot_band(ax, x, cate_y, cate_se, color=DEFAULT_PALETTE[0])

        ax.plot(x, tree_y, color=DEFAULT_PALETTE[1], linewidth=2.2, label="Policy tree")
        _plot_band(ax, x, tree_y, tree_se, color=DEFAULT_PALETTE[1])

        if y_mode == "absolute":
            ax.plot(x, ate_y, color=DEFAULT_PALETTE[2], linewidth=2.0, linestyle="--", label="Grouped ATE")
            _plot_band(ax, x, ate_y, ate_se, color=DEFAULT_PALETTE[2], alpha=0.12)


    axes[0].set_ylabel(y_label)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    dedup = {}
    for h, l in zip(handles, labels):
        dedup[l] = h
    if dedup:
        fig.legend(dedup.values(), dedup.keys(), loc="lower center", ncol=4, fontsize=9)

    fig.suptitle(_study_title(dataset_name), fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def make_plots(plot_rows, y_mode, output_dir, output_prefix, dpi):
    output_paths = []
    datasets = sorted(plot_rows["dataset"].astype(str).unique().tolist())
    for dataset_name in datasets:
        rows = plot_rows[plot_rows["dataset"].astype(str) == dataset_name].copy()
        if output_prefix == "":
            output_path = output_dir / f"{_slugify(dataset_name)}.png"
        else:
            output_path = output_dir / f"{output_prefix}_{_slugify(dataset_name)}.png"
        saved = make_dataset_plot(rows, dataset_name, y_mode=y_mode, output_path=output_path, dpi=dpi)
        output_paths.append(saved)
    return output_paths


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Plot holdout vs full-information policy-value curves side by side for each dataset."
    )
    parser.add_argument("--summary-paths", type=str, default=None, help="Comma-separated summary CSV paths.")
    parser.add_argument(
        "--summary-glob",
        type=str,
        default="results/_array_tmp/gerber/summary_*.csv,results/_array_tmp/nsw/summary_*.csv,results/_array_tmp/jtpa/summary_*.csv",
        help="Comma-separated glob patterns for summary CSVs (e.g. results/_array_tmp/gerber/summary_*.csv).",
    )
    parser.add_argument("--output-dir", type=str, default="results/analysis")
    parser.add_argument("--output-prefix", type=str, default="")

    parser.add_argument("--datasets", type=str, default=None, help="Optional dataset filter.")
    parser.add_argument("--states", type=str, default=None, help="Optional state/group filter.")
    parser.add_argument(
        "--evaluation-modes",
        type=str,
        default="holdout,full",
        help="Optional eval-mode filter. Defaults to holdout,full.",
    )
    parser.add_argument("--run-ids", type=str, default=None, help="Optional run-id filter.")

    parser.add_argument(
        "--plot-y-mode",
        type=str,
        choices=["relative", "absolute"],
        default="relative",
        help="Plot delta vs grouped ATE or absolute policy values.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    summary, input_paths = load_summaries(args.summary_paths, args.summary_glob)
    summary = apply_filters(summary, args)
    if summary.empty:
        raise RuntimeError("No rows remaining after filters.")

    combined_summary_path = output_dir / f"{args.output_prefix}_summary_combined.csv"
    summary = collapse_summary_rows(summary)
    summary.to_csv(combined_summary_path, index=False)

    summary = add_standard_errors(summary)
    plot_rows = add_plot_columns(summary, y_mode=args.plot_y_mode)

    cost_rows_path = output_dir / f"{args.output_prefix}_cost_rows_collapsed.csv"
    plot_rows.to_csv(cost_rows_path, index=False)

    saved_paths = make_plots(
        plot_rows,
        y_mode=args.plot_y_mode,
        output_dir=output_dir,
        output_prefix=args.output_prefix,
        dpi=args.dpi,
    )

    print(f"Loaded {len(input_paths)} summary file(s).")
    print(f"Combined summary written to: {combined_summary_path}")
    print(f"Collapsed rows written to: {cost_rows_path}")
    if saved_paths:
        print("Plots written to:")
        for path in saved_paths:
            print(f"  {path}")
    else:
        print("No plots generated.")


if __name__ == "__main__":
    parser = build_arg_parser()
    parsed_args = parser.parse_args()
    main(parsed_args)
