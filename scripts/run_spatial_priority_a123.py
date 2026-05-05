#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from analyze_eval_slices import (  # noqa: E402
    build_tile_level_summary,
    extract_dataset_metadata,
    infer_pred_bucket,
)


PREVALENCE_TASK_SPECS = {
    "B1_seasonality_present": {
        "gold_col": "B1_gold_seasonal_frac",
        "pred_col": "B1_pred_seasonal_frac",
        "display": "B1 seasonal prevalence",
        "short": "B1",
    },
    "C1_step_time": {
        "gold_col": "C1_gold_step_frac",
        "pred_col": "C1_pred_step_frac",
        "display": "C1 step prevalence",
        "short": "C1",
    },
    "C2_step_mag": {
        "gold_col": "C2_gold_step_frac",
        "pred_col": "C2_pred_step_frac",
        "display": "C2 step prevalence",
        "short": "C2",
    },
}


SYSTEM_STYLE = {
    "baseline_v3": {"color": "#222222", "marker": "s"},
    "gpt52_all_base": {"color": "#0B5FFF", "marker": "o"},
    "gpt52_full": {"color": "#00A88F", "marker": "D"},
    "llama4_all_base": {"color": "#A05A2C", "marker": "^"},
    "llama4_full": {"color": "#D1495B", "marker": "P"},
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_system_arg(items: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --system entry: {item!r}. Expected label=per_example.csv path.")
        label, path = item.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Bad --system entry: {item!r}.")
        out.append((label, path))
    return out


def load_metadata(dataset_jsonl: str, metadata_csv: str) -> pd.DataFrame:
    if not os.path.exists(metadata_csv):
        extract_dataset_metadata(dataset_jsonl, metadata_csv)
    return pd.read_csv(metadata_csv)


def load_system_df(meta: pd.DataFrame, per_example_csv: str, label: str) -> pd.DataFrame:
    per_example = pd.read_csv(per_example_csv)
    keep_cols = [c for c in ["id", "task_name", "pred_answer", "ok", "parse_error", "reason"] if c in per_example.columns]
    per_example = per_example[keep_cols].copy()
    if "task_name" not in per_example.columns:
        raise ValueError(f"{per_example_csv} missing task_name column")
    if "pred_answer" not in per_example.columns:
        raise ValueError(f"{per_example_csv} missing pred_answer column")
    if "ok" not in per_example.columns:
        raise ValueError(f"{per_example_csv} missing ok column")
    if "parse_error" not in per_example.columns:
        per_example["parse_error"] = None
    if "reason" not in per_example.columns:
        per_example["reason"] = None

    df = meta.merge(per_example, on=["id", "task_name"], how="inner")
    df["pred_bucket"] = [infer_pred_bucket(t, a) for t, a in zip(df["task_name"], df["pred_answer"])]
    df["system"] = label
    return df


def safe_corr(a: pd.Series, b: pd.Series, method: str) -> Optional[float]:
    if len(a) < 2:
        return None
    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return None
    v = a.corr(b, method=method)
    if pd.isna(v):
        return None
    return float(v)


def compute_prevalence_tables(system_dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_tile_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for system_label, df in system_dfs.items():
        tile_summary = build_tile_level_summary(df)
        tile_summary["system"] = system_label
        for task_name, spec in PREVALENCE_TASK_SPECS.items():
            sub = tile_summary[["tile_id", "n_examples", "n_pid", spec["gold_col"], spec["pred_col"]]].copy()
            sub = sub.rename(columns={spec["gold_col"]: "gold_prevalence", spec["pred_col"]: "pred_prevalence"})
            sub["task_name"] = task_name
            sub["task_display"] = spec["display"]
            sub["system"] = system_label
            sub["abs_error"] = (sub["pred_prevalence"] - sub["gold_prevalence"]).abs()
            by_tile_rows.extend(sub.to_dict(orient="records"))

            summary_rows.append(
                {
                    "system": system_label,
                    "task_name": task_name,
                    "task_display": spec["display"],
                    "n_tiles": int(len(sub)),
                    "mean_abs_error": float(sub["abs_error"].mean()),
                    "weighted_abs_error_by_examples": float(np.average(sub["abs_error"], weights=sub["n_examples"])),
                    "pearson_r": safe_corr(sub["gold_prevalence"], sub["pred_prevalence"], "pearson"),
                    "spearman_rho": safe_corr(sub["gold_prevalence"], sub["pred_prevalence"], "spearman"),
                }
            )

    by_tile_df = pd.DataFrame(by_tile_rows)
    summary_df = pd.DataFrame(summary_rows)
    return by_tile_df, summary_df


def plot_prevalence_scatter(by_tile_df: pd.DataFrame, systems: List[str], out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    for ax, task_name in zip(axes, PREVALENCE_TASK_SPECS.keys()):
        sub = by_tile_df[by_tile_df["task_name"] == task_name].copy()
        for system_label in systems:
            sys_sub = sub[sub["system"] == system_label]
            if sys_sub.empty:
                continue
            style = SYSTEM_STYLE.get(system_label, {"color": None, "marker": "o"})
            ax.scatter(
                sys_sub["gold_prevalence"],
                sys_sub["pred_prevalence"],
                label=system_label,
                s=60,
                alpha=0.85,
                color=style["color"],
                marker=style["marker"],
                edgecolors="white",
                linewidths=0.6,
            )
            for _, row in sys_sub.iterrows():
                ax.annotate(
                    row["tile_id"],
                    (row["gold_prevalence"], row["pred_prevalence"]),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                    alpha=0.8,
                )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Gold prevalence")
        ax.set_ylabel("Predicted prevalence")
        ax.set_title(PREVALENCE_TASK_SPECS[task_name]["display"])
        ax.grid(alpha=0.2, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Tile-level prevalence calibration", y=1.13, fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_prevalence_abs_error(by_tile_df: pd.DataFrame, systems: List[str], out_path: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    tile_order = sorted(by_tile_df["tile_id"].unique().tolist())

    for ax, task_name in zip(axes, PREVALENCE_TASK_SPECS.keys()):
        sub = by_tile_df[by_tile_df["task_name"] == task_name].copy()
        width = 0.8 / max(1, len(systems))
        xs = np.arange(len(tile_order))
        for idx, system_label in enumerate(systems):
            sys_sub = sub[sub["system"] == system_label].set_index("tile_id").reindex(tile_order)
            style = SYSTEM_STYLE.get(system_label, {"color": None})
            ax.bar(
                xs + (idx - (len(systems) - 1) / 2) * width,
                sys_sub["abs_error"].to_numpy(dtype=float),
                width=width,
                label=system_label,
                color=style["color"],
                alpha=0.9,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(tile_order)
        ax.set_ylabel("Absolute error")
        ax.set_title(PREVALENCE_TASK_SPECS[task_name]["display"])
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_ylim(0, max(0.05, float(sub["abs_error"].max()) * 1.18))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Per-tile prevalence absolute error", y=1.05, fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def mode_label(labels: List[str]) -> Optional[str]:
    if not labels:
        return None
    counts: Dict[str, int] = defaultdict(int)
    for label in labels:
        counts[str(label)] += 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def compute_pointwise_spatial_metrics(df: pd.DataFrame, tasks: List[str], knn: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for task_name in tasks:
        sub_task = df[df["task_name"] == task_name].copy()
        sub_task = sub_task.dropna(subset=["easting", "northing"])
        if sub_task.empty:
            continue

        for tile_id, sub in sub_task.groupby("tile_id", sort=True):
            if len(sub) <= knn:
                continue

            coords = sub[["easting", "northing"]].to_numpy(dtype=np.float64)
            gold = sub["bucket_label"].astype(str).tolist()
            pred = sub["pred_bucket"].astype(str).tolist()
            ids = sub["id"].astype(str).tolist()
            easting = sub["easting"].to_numpy(dtype=np.float64)
            northing = sub["northing"].to_numpy(dtype=np.float64)

            d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
            np.fill_diagonal(d2, np.inf)
            nn_idx = np.argpartition(d2, kth=knn - 1, axis=1)[:, :knn]

            for i in range(len(sub)):
                neigh = nn_idx[i].tolist()
                neigh_gold = [gold[j] for j in neigh]
                neigh_pred = [pred[j] for j in neigh]
                gold_agree = float(np.mean([gold[j] == gold[i] for j in neigh]))
                pred_agree = float(np.mean([pred[j] == pred[i] for j in neigh]))
                majority_gold = mode_label(neigh_gold)
                majority_pred = mode_label(neigh_pred)
                anomaly_flag = int(majority_gold is not None and gold[i] == majority_gold and pred[i] != majority_gold)
                oversmooth_flag = int(majority_gold is not None and gold[i] != majority_gold and pred[i] == majority_gold)
                pred_majority_disagree = int(majority_pred is not None and pred[i] != majority_pred)
                rows.append(
                    {
                        "id": ids[i],
                        "tile_id": tile_id,
                        "task_name": task_name,
                        "bucket_label": gold[i],
                        "pred_bucket": pred[i],
                        "easting": easting[i],
                        "northing": northing[i],
                        "gold_neighbor_agreement": gold_agree,
                        "pred_neighbor_agreement": pred_agree,
                        "agreement_gap": pred_agree - gold_agree,
                        "anomaly_flag": anomaly_flag,
                        "oversmooth_flag": oversmooth_flag,
                        "pred_majority_disagree": pred_majority_disagree,
                        "local_inconsistency_proxy": 1.0 - pred_agree,
                    }
                )
    return pd.DataFrame(rows)


def summarise_spatial_metrics(points_df: pd.DataFrame, system_label: str, knn: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_tile = (
        points_df.groupby(["task_name", "tile_id"], sort=True)
        .agg(
            n_points=("id", "size"),
            gold_neighbor_agreement=("gold_neighbor_agreement", "mean"),
            pred_neighbor_agreement=("pred_neighbor_agreement", "mean"),
            agreement_gap=("agreement_gap", "mean"),
            anomaly_rate=("anomaly_flag", "mean"),
            oversmooth_rate=("oversmooth_flag", "mean"),
            pred_majority_disagree_rate=("pred_majority_disagree", "mean"),
            local_inconsistency_proxy=("local_inconsistency_proxy", "mean"),
        )
        .reset_index()
    )
    by_tile["system"] = system_label
    by_tile["knn"] = int(knn)

    summary = (
        by_tile.groupby(["system", "task_name"], sort=True)
        .agg(
            n_tiles=("tile_id", "size"),
            mean_pred_neighbor_agreement=("pred_neighbor_agreement", "mean"),
            std_pred_neighbor_agreement=("pred_neighbor_agreement", "std"),
            mean_anomaly_rate=("anomaly_rate", "mean"),
            std_anomaly_rate=("anomaly_rate", "std"),
            mean_local_inconsistency_proxy=("local_inconsistency_proxy", "mean"),
            std_local_inconsistency_proxy=("local_inconsistency_proxy", "std"),
            mean_oversmooth_rate=("oversmooth_rate", "mean"),
            std_oversmooth_rate=("oversmooth_rate", "std"),
            mean_pred_majority_disagree_rate=("pred_majority_disagree_rate", "mean"),
            std_pred_majority_disagree_rate=("pred_majority_disagree_rate", "std"),
        )
        .reset_index()
    )
    return by_tile, summary


def compute_delta_vs_baseline(summary_df: pd.DataFrame, baseline_label: str) -> pd.DataFrame:
    base = summary_df[summary_df["system"] == baseline_label].set_index("task_name")
    rows: List[Dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        system = str(row["system"])
        task_name = str(row["task_name"])
        if system == baseline_label or task_name not in base.index:
            continue
        b = base.loc[task_name]
        rows.append(
            {
                "system": system,
                "task_name": task_name,
                "delta_neighbor_agreement": float(row["mean_pred_neighbor_agreement"] - b["mean_pred_neighbor_agreement"]),
                "delta_anomaly_rate": float(row["mean_anomaly_rate"] - b["mean_anomaly_rate"]),
                "delta_local_inconsistency_proxy": float(row["mean_local_inconsistency_proxy"] - b["mean_local_inconsistency_proxy"]),
                "delta_oversmooth_rate": float(row["mean_oversmooth_rate"] - b["mean_oversmooth_rate"]),
                "delta_pred_majority_disagree_rate": float(row["mean_pred_majority_disagree_rate"] - b["mean_pred_majority_disagree_rate"]),
            }
        )
    return pd.DataFrame(rows)


def save_spatial_summary_latex(summary_df: pd.DataFrame, out_path: str) -> None:
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"System & Task & Neighbor agr. & Anomaly rate & Local inconsistency \\",
        r"\midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "{} & {} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} \\\\".format(
                row["system"],
                PREVALENCE_TASK_SPECS.get(row["task_name"], {"short": row["task_name"]})["short"],
                row["mean_pred_neighbor_agreement"],
                0.0 if pd.isna(row["std_pred_neighbor_agreement"]) else row["std_pred_neighbor_agreement"],
                row["mean_anomaly_rate"],
                0.0 if pd.isna(row["std_anomaly_rate"]) else row["std_anomaly_rate"],
                row["mean_local_inconsistency_proxy"],
                0.0 if pd.isna(row["std_local_inconsistency_proxy"]) else row["std_local_inconsistency_proxy"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_hotspot_figure(
    baseline_points: pd.DataFrame,
    hybrid_points: pd.DataFrame,
    task_name: str,
    tile_id: str,
    baseline_label: str,
    hybrid_label: str,
    out_path: str,
) -> pd.DataFrame:
    base = baseline_points[(baseline_points["task_name"] == task_name) & (baseline_points["tile_id"] == tile_id)].copy()
    hyb = hybrid_points[(hybrid_points["task_name"] == task_name) & (hybrid_points["tile_id"] == tile_id)].copy()
    joined = base.merge(
        hyb[["id", "anomaly_flag", "agreement_gap", "pred_bucket"]],
        on="id",
        suffixes=("_baseline", "_hybrid"),
        how="inner",
    )
    if joined.empty:
        raise ValueError(f"No hotspot data for task={task_name} tile={tile_id}")

    joined["status"] = "stable_non_anomaly"
    joined.loc[(joined["anomaly_flag_baseline"] == 1) & (joined["anomaly_flag_hybrid"] == 0), "status"] = "fixed_by_hybrid"
    joined.loc[(joined["anomaly_flag_baseline"] == 0) & (joined["anomaly_flag_hybrid"] == 1), "status"] = "introduced_by_hybrid"
    joined.loc[(joined["anomaly_flag_baseline"] == 1) & (joined["anomaly_flag_hybrid"] == 1), "status"] = "persistent_anomaly"

    colors = {
        "stable_non_anomaly": "#C8C8C8",
        "fixed_by_hybrid": "#2E8B57",
        "introduced_by_hybrid": "#8E5EA2",
        "persistent_anomaly": "#D1495B",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)
    task_display = PREVALENCE_TASK_SPECS.get(task_name, {"display": task_name})["display"]

    for ax, col, title in [
        (axes[0], "anomaly_flag_baseline", f"{baseline_label} anomaly map"),
        (axes[1], "anomaly_flag_hybrid", f"{hybrid_label} anomaly map"),
    ]:
        ax.scatter(joined["easting"], joined["northing"], s=12, color="#D9D9D9", alpha=0.55, linewidths=0)
        anom = joined[joined[col] == 1]
        ax.scatter(anom["easting"], anom["northing"], s=20, color="#D1495B", alpha=0.95, linewidths=0)
        ax.set_title(f"{title}\nrate={anom.shape[0] / max(1, joined.shape[0]):.3f}")
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18, linewidth=0.5)

    ax = axes[2]
    for status, color in colors.items():
        sub = joined[joined["status"] == status]
        if sub.empty:
            continue
        ax.scatter(sub["easting"], sub["northing"], s=18, color=color, alpha=0.92, linewidths=0, label=status)
    ax.set_title("Difference map")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="best", fontsize=8, frameon=True)

    fig.suptitle(f"{task_display} hotspot comparison on {tile_id}", fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "task_name": task_name,
                "tile_id": tile_id,
                "baseline_label": baseline_label,
                "hybrid_label": hybrid_label,
                "n_points": int(joined.shape[0]),
                "baseline_anomaly_rate": float(joined["anomaly_flag_baseline"].mean()),
                "hybrid_anomaly_rate": float(joined["anomaly_flag_hybrid"].mean()),
                "fixed_by_hybrid": int((joined["status"] == "fixed_by_hybrid").sum()),
                "introduced_by_hybrid": int((joined["status"] == "introduced_by_hybrid").sum()),
                "persistent_anomaly": int((joined["status"] == "persistent_anomaly").sum()),
            }
        ]
    )
    return summary


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--system", action="append", required=True, help="label=per_example.csv path")
    ap.add_argument("--baseline_label", default="baseline_v3")
    ap.add_argument("--hybrid_label", default="gpt52_full")
    ap.add_argument("--hotspot_tile", default="E44N43")
    ap.add_argument("--hotspot_tasks", default="B1_seasonality_present,C1_step_time,C2_step_mag")
    ap.add_argument("--knn", type=int, default=8)
    args = ap.parse_args()

    systems = parse_system_arg(args.system)
    ensure_dir(args.out_dir)
    fig_dir = os.path.join(args.out_dir, "figures")
    ensure_dir(fig_dir)
    metadata_csv = os.path.join(args.out_dir, "dataset_metadata.csv")
    meta = load_metadata(args.dataset, metadata_csv)

    system_dfs: Dict[str, pd.DataFrame] = {}
    for label, per_example_csv in systems:
        system_dfs[label] = load_system_df(meta, per_example_csv, label)

    ordered_systems = [label for label, _ in systems]

    a1_dir = os.path.join(args.out_dir, "exp_a1_prevalence")
    ensure_dir(a1_dir)
    by_tile_df, summary_df = compute_prevalence_tables(system_dfs)
    by_tile_df.to_csv(os.path.join(a1_dir, "tile_prevalence_by_system_tile.csv"), index=False)
    summary_df.to_csv(os.path.join(a1_dir, "tile_prevalence_summary.csv"), index=False)
    plot_prevalence_scatter(by_tile_df, ordered_systems, os.path.join(fig_dir, "exp_a1_prevalence_scatter.png"))
    plot_prevalence_abs_error(by_tile_df, ordered_systems, os.path.join(fig_dir, "exp_a1_prevalence_abs_error.png"))

    a2_dir = os.path.join(args.out_dir, "exp_a2_hotspots")
    a3_dir = os.path.join(args.out_dir, "exp_a3_spatial_consistency")
    ensure_dir(a2_dir)
    ensure_dir(a3_dir)

    hotspot_tasks = [t.strip() for t in args.hotspot_tasks.split(",") if t.strip()]
    pointwise_by_system: Dict[str, pd.DataFrame] = {}
    by_tile_rows: List[pd.DataFrame] = []
    summary_rows: List[pd.DataFrame] = []

    for label in ordered_systems:
        pts = compute_pointwise_spatial_metrics(system_dfs[label], hotspot_tasks, int(args.knn))
        pts["system"] = label
        pts.to_csv(os.path.join(a3_dir, f"pointwise_{label}.csv"), index=False)
        pointwise_by_system[label] = pts
        by_tile_df_sys, summary_df_sys = summarise_spatial_metrics(pts, label, int(args.knn))
        by_tile_rows.append(by_tile_df_sys)
        summary_rows.append(summary_df_sys)

    by_tile_all = pd.concat(by_tile_rows, ignore_index=True)
    summary_all = pd.concat(summary_rows, ignore_index=True)
    delta_df = compute_delta_vs_baseline(summary_all, args.baseline_label)

    by_tile_all.to_csv(os.path.join(a3_dir, "spatial_consistency_by_tile.csv"), index=False)
    summary_all.to_csv(os.path.join(a3_dir, "spatial_consistency_summary.csv"), index=False)
    delta_df.to_csv(os.path.join(a3_dir, "spatial_consistency_delta_vs_baseline.csv"), index=False)
    save_spatial_summary_latex(summary_all, os.path.join(a3_dir, "table_spatial_consistency_summary.tex"))

    hotspot_summaries: List[pd.DataFrame] = []
    if args.baseline_label not in pointwise_by_system:
        raise ValueError(f"baseline label {args.baseline_label!r} not among systems")
    if args.hybrid_label not in pointwise_by_system:
        raise ValueError(f"hybrid label {args.hybrid_label!r} not among systems")
    for task_name in hotspot_tasks:
        hotspot_summary = plot_hotspot_figure(
            baseline_points=pointwise_by_system[args.baseline_label],
            hybrid_points=pointwise_by_system[args.hybrid_label],
            task_name=task_name,
            tile_id=args.hotspot_tile,
            baseline_label=args.baseline_label,
            hybrid_label=args.hybrid_label,
            out_path=os.path.join(fig_dir, f"exp_a2_hotspot_{task_name}_{args.hotspot_tile}.png"),
        )
        hotspot_summaries.append(hotspot_summary)
    pd.concat(hotspot_summaries, ignore_index=True).to_csv(os.path.join(a2_dir, "hotspot_summary.csv"), index=False)

    summary = {
        "dataset": args.dataset,
        "systems": ordered_systems,
        "baseline_label": args.baseline_label,
        "hybrid_label": args.hybrid_label,
        "hotspot_tile": args.hotspot_tile,
        "hotspot_tasks": hotspot_tasks,
        "knn": int(args.knn),
        "outputs": {
            "exp_a1_dir": a1_dir,
            "exp_a2_dir": a2_dir,
            "exp_a3_dir": a3_dir,
            "fig_dir": fig_dir,
        },
    }
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    print("Saved spatial priority package to:", args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
