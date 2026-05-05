#!/usr/bin/env python3
import argparse
import math
import os
import sys
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
    summarise_micro_macro,
    summarise_task_accuracy,
)


TASK_GAIN_COLS = {
    "B1_seasonality_present": "B1_gain",
    "B2_seasonality_amp": "B2_gain",
    "C1_step_time": "C1_gain",
    "C2_step_mag": "C2_gain",
}

REGIME_COLOR = {
    "stable_dominated": "#6A994E",
    "subsiding_dominated": "#386FA4",
    "uplifting_dominated": "#BC4749",
    "unknown": "#777777",
}

SYSTEM_COLOR = {
    "baseline_v3": "#222222",
    "full_hybrid": "#00A88F",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_system_df(meta: pd.DataFrame, per_example_csv: str, label: str) -> pd.DataFrame:
    per_example = pd.read_csv(per_example_csv)
    keep_cols = [c for c in ["id", "task_name", "pred_answer", "ok", "parse_error", "reason"] if c in per_example.columns]
    per_example = per_example[keep_cols].copy()
    df = meta.merge(per_example, on=["id", "task_name"], how="inner")
    df["pred_bucket"] = [infer_pred_bucket(t, a) for t, a in zip(df["task_name"], df["pred_answer"])]
    df["system"] = label
    return df


def dominant_trend_regime(tile_row: pd.Series) -> str:
    pairs = [
        ("subsiding_dominated", float(tile_row.get("A1_gold_frac_subsiding", float("nan")))),
        ("stable_dominated", float(tile_row.get("A1_gold_frac_stable", float("nan")))),
        ("uplifting_dominated", float(tile_row.get("A1_gold_frac_uplifting", float("nan")))),
    ]
    pairs = [(label, val) for label, val in pairs if math.isfinite(val)]
    if not pairs:
        return "unknown"
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    return pairs[0][0]


def build_tile_regimes(ref_df: pd.DataFrame) -> pd.DataFrame:
    tile_summary = build_tile_level_summary(ref_df)
    tile_summary["trend_regime"] = tile_summary.apply(dominant_trend_regime, axis=1)
    tile_summary["step_prevalence_gold"] = 0.5 * (
        tile_summary["C1_gold_step_frac"].fillna(0.0) + tile_summary["C2_gold_step_frac"].fillna(0.0)
    )
    seasonal_median = float(tile_summary["B1_gold_seasonal_frac"].median())
    step_median = float(tile_summary["step_prevalence_gold"].median())
    tile_summary["seasonality_regime"] = np.where(
        tile_summary["B1_gold_seasonal_frac"] >= seasonal_median,
        "high_seasonality",
        "lower_seasonality",
    )
    tile_summary["step_regime"] = np.where(
        tile_summary["step_prevalence_gold"] >= step_median,
        "high_step_prevalence",
        "lower_step_prevalence",
    )
    keep_cols = [
        "tile_id",
        "n_pid",
        "n_examples",
        "trend_regime",
        "seasonality_regime",
        "step_regime",
        "B1_gold_seasonal_frac",
        "step_prevalence_gold",
        "C1_gold_step_frac",
        "C2_gold_step_frac",
        "A1_gold_frac_subsiding",
        "A1_gold_frac_stable",
        "A1_gold_frac_uplifting",
    ]
    return tile_summary[keep_cols].copy()


def build_tile_perf(df: pd.DataFrame) -> pd.DataFrame:
    micro_macro = summarise_micro_macro(df, ["tile_id"]).rename(
        columns={
            "micro_acc": "micro_acc",
            "macro_acc": "macro_acc",
            "n": "n_examples",
            "n_pid": "n_pid",
        }
    )
    task_acc = summarise_task_accuracy(df, ["tile_id"])
    task_pivot = (
        task_acc[task_acc["task_name"].isin(TASK_GAIN_COLS.keys())]
        .pivot(index="tile_id", columns="task_name", values="acc")
        .rename(columns={task: f"{TASK_GAIN_COLS[task].replace('_gain', '')}_acc" for task in TASK_GAIN_COLS})
        .reset_index()
    )
    return micro_macro.merge(task_pivot, on="tile_id", how="left")


def build_tile_gain(baseline_df: pd.DataFrame, hybrid_df: pd.DataFrame) -> pd.DataFrame:
    base = build_tile_perf(baseline_df).add_prefix("baseline_").rename(columns={"baseline_tile_id": "tile_id"})
    hyb = build_tile_perf(hybrid_df).add_prefix("hybrid_").rename(columns={"hybrid_tile_id": "tile_id"})
    merged = base.merge(hyb, on="tile_id", how="inner")
    merged["n_pid"] = merged["hybrid_n_pid"]
    merged["n_examples"] = merged["hybrid_n_examples"]
    merged["micro_gain"] = merged["hybrid_micro_acc"] - merged["baseline_micro_acc"]
    merged["macro_gain"] = merged["hybrid_macro_acc"] - merged["baseline_macro_acc"]
    for short in ["B1", "B2", "C1", "C2"]:
        merged[f"{short}_gain"] = merged[f"hybrid_{short}_acc"] - merged[f"baseline_{short}_acc"]
    return merged


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int = 5000) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[sample_idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return mean, float(lo), float(hi)


def build_regime_gain_summary(tile_gain: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metrics = ["micro_gain", "macro_gain", "B1_gain", "B2_gain", "C1_gain", "C2_gain"]
    families = {
        "trend_regime": sorted(tile_gain["trend_regime"].dropna().unique().tolist()),
        "seasonality_regime": ["high_seasonality", "lower_seasonality"],
        "step_regime": ["high_step_prevalence", "lower_step_prevalence"],
    }
    for family, labels in families.items():
        for label in labels:
            sub = tile_gain[tile_gain[family] == label].copy()
            if sub.empty:
                continue
            tile_ids = ",".join(sorted(sub["tile_id"].astype(str).tolist()))
            for metric in metrics:
                mean, lo, hi = bootstrap_mean_ci(sub[metric].to_numpy(dtype=float), seed=seed + len(rows))
                rows.append(
                    {
                        "regime_family": family,
                        "regime_label": label,
                        "metric": metric,
                        "n_tiles": int(len(sub)),
                        "mean_gain": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "tile_ids": tile_ids,
                    }
                )
    return pd.DataFrame(rows)


def kth_distance_threshold(coords: np.ndarray, k: int) -> float:
    if len(coords) <= 1:
        return 0.0
    kk = max(1, min(k, len(coords) - 1))
    d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    kth = np.partition(d2, kk - 1, axis=1)[:, kk - 1]
    kth = kth[np.isfinite(kth)]
    if len(kth) == 0:
        return 0.0
    return float(np.sqrt(np.median(kth)))


def connected_component_sizes(coords: np.ndarray, distance_threshold: float) -> List[int]:
    n = len(coords)
    if n == 0:
        return []
    if n == 1 or distance_threshold <= 0:
        return [1] * n
    d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    adj = d2 <= (distance_threshold ** 2)
    np.fill_diagonal(adj, True)
    seen = np.zeros(n, dtype=bool)
    sizes: List[int] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            nbrs = np.where(adj[cur])[0]
            for nb in nbrs:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(int(nb))
        sizes.append(int(size))
    return sorted(sizes, reverse=True)


def cluster_profile_rows(
    points_df: pd.DataFrame,
    system_label: str,
    tile_id: str,
    k_for_threshold: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cluster_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []
    for task_name in ["B1_seasonality_present", "C1_step_time", "C2_step_mag"]:
        sub = points_df[(points_df["tile_id"] == tile_id) & (points_df["task_name"] == task_name)].copy()
        if sub.empty:
            continue
        coords_all = sub[["easting", "northing"]].to_numpy(dtype=np.float64)
        d_thr = kth_distance_threshold(coords_all, k=k_for_threshold)
        masks = {
            f"{task_name}_anomaly": sub["anomaly_flag"].astype(int) == 1,
        }
        if task_name == "B1_seasonality_present":
            masks["B1_false_seasonal"] = (sub["bucket_label"].astype(str) != "seasonal") & (sub["pred_bucket"].astype(str) == "seasonal")
        else:
            masks[f"{task_name}_false_step"] = (sub["bucket_label"].astype(str) != "step") & (sub["pred_bucket"].astype(str) == "step")

        for cluster_type, mask in masks.items():
            err = sub[mask].copy()
            sizes = connected_component_sizes(err[["easting", "northing"]].to_numpy(dtype=np.float64), d_thr)
            n_points = int(len(err))
            n_components = int(len(sizes))
            isolated_points = int(sum(1 for s in sizes if s == 1))
            cluster_rows.append(
                {
                    "system": system_label,
                    "tile_id": tile_id,
                    "task_name": task_name,
                    "cluster_type": cluster_type,
                    "distance_threshold": d_thr,
                    "n_error_points": n_points,
                    "n_components": n_components,
                    "isolated_points": isolated_points,
                    "isolated_point_frac": float(isolated_points / n_points) if n_points else float("nan"),
                    "max_component_size": int(max(sizes)) if sizes else 0,
                    "mean_component_size": float(np.mean(sizes)) if sizes else float("nan"),
                }
            )
            for rank, size in enumerate(sizes, start=1):
                component_rows.append(
                    {
                        "system": system_label,
                        "tile_id": tile_id,
                        "task_name": task_name,
                        "cluster_type": cluster_type,
                        "component_rank": int(rank),
                        "component_size": int(size),
                    }
                )
    return cluster_rows, component_rows


def plot_tile_gain_bubble(tile_gain: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    sizes = 40 + 2.5 * tile_gain["n_pid"].to_numpy(dtype=float)
    colors = [REGIME_COLOR.get(x, "#777777") for x in tile_gain["trend_regime"].astype(str)]
    ax.scatter(
        tile_gain["baseline_micro_acc"],
        tile_gain["micro_gain"],
        s=sizes,
        c=colors,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.8,
    )
    for _, row in tile_gain.iterrows():
        ax.annotate(
            row["tile_id"],
            (row["baseline_micro_acc"], row["micro_gain"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axhline(0.0, linestyle="--", color="#666666", linewidth=1)
    ax.set_xlabel("Baseline micro accuracy")
    ax.set_ylabel("Hybrid minus baseline micro gain")
    ax.set_title("Cross-tile gain heterogeneity")
    ax.grid(alpha=0.22, linewidth=0.5)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_regime_micro_gain(regime_df: pd.DataFrame, out_path: str) -> None:
    sub = regime_df[regime_df["metric"] == "micro_gain"].copy()
    if sub.empty:
        return
    sub["label"] = sub["regime_family"] + ":" + sub["regime_label"]
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    xs = np.arange(len(sub))
    means = sub["mean_gain"].to_numpy(dtype=float)
    yerr = np.vstack([
        means - sub["ci_low"].to_numpy(dtype=float),
        sub["ci_high"].to_numpy(dtype=float) - means,
    ])
    ax.bar(xs, means, color="#0B5FFF", alpha=0.85)
    ax.errorbar(xs, means, yerr=yerr, fmt="none", ecolor="#1F1F1F", capsize=3, linewidth=1)
    ax.axhline(0.0, linestyle="--", color="#666666", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(sub["label"], rotation=25, ha="right")
    ax.set_ylabel("Mean micro gain")
    ax.set_title("Regime-conditioned hybrid gain (tile bootstrap 95% CI)")
    ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_component_hist(components_df: pd.DataFrame, tile_id: str, out_path: str) -> None:
    plot_specs = [
        ("B1_false_seasonal", "B1 false seasonal clusters"),
        ("C1_step_time_false_step", "C1 false step clusters"),
        ("C2_step_mag_false_step", "C2 false step clusters"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6), constrained_layout=True)
    for ax, (cluster_type, title) in zip(axes, plot_specs):
        sub = components_df[components_df["cluster_type"] == cluster_type].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        for system_label, color in SYSTEM_COLOR.items():
            sys_sub = sub[sub["system"] == system_label]
            if sys_sub.empty:
                continue
            counts = sys_sub["component_size"].value_counts().sort_index()
            ax.plot(
                counts.index.to_numpy(dtype=float),
                counts.values.astype(float),
                marker="o",
                linewidth=1.6,
                color=color,
                label=system_label,
            )
        ax.set_xlabel("Component size")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(alpha=0.22, linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"Representative tile cluster profile: {tile_id}", y=1.09, fontsize=13)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--metadata_csv", required=True)
    ap.add_argument("--baseline_per_example", required=True)
    ap.add_argument("--hybrid_per_example", required=True)
    ap.add_argument("--baseline_pointwise_csv", required=True)
    ap.add_argument("--hybrid_pointwise_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=20260504)
    ap.add_argument("--cluster_k", type=int, default=8)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    figures_dir = os.path.join(args.out_dir, "figures")
    ensure_dir(figures_dir)

    if not os.path.exists(args.metadata_csv):
        extract_dataset_metadata(args.dataset_jsonl, args.metadata_csv)

    meta = pd.read_csv(args.metadata_csv)
    baseline_df = load_system_df(meta, args.baseline_per_example, "baseline_v3")
    hybrid_df = load_system_df(meta, args.hybrid_per_example, "full_hybrid")

    tile_regimes = build_tile_regimes(baseline_df)
    tile_gain = build_tile_gain(baseline_df, hybrid_df).merge(tile_regimes, on=["tile_id", "n_pid", "n_examples"], how="left")
    tile_gain = tile_gain.sort_values(["micro_gain", "macro_gain", "tile_id"], ascending=[False, False, True]).reset_index(drop=True)
    tile_gain.to_csv(os.path.join(args.out_dir, "tile_gain_summary.csv"), index=False)
    tile_gain.to_csv(os.path.join(args.out_dir, "tile_gain_ranking.csv"), index=False)

    regime_gain = build_regime_gain_summary(tile_gain, seed=int(args.seed))
    regime_gain.to_csv(os.path.join(args.out_dir, "regime_gain_summary.csv"), index=False)

    representative_tile = str(tile_gain.iloc[0]["tile_id"])
    baseline_points = pd.read_csv(args.baseline_pointwise_csv)
    hybrid_points = pd.read_csv(args.hybrid_pointwise_csv)
    cluster_rows, component_rows = [], []
    for system_label, points_df in [("baseline_v3", baseline_points), ("full_hybrid", hybrid_points)]:
        rows_a, rows_b = cluster_profile_rows(
            points_df=points_df,
            system_label=system_label,
            tile_id=representative_tile,
            k_for_threshold=int(args.cluster_k),
        )
        cluster_rows.extend(rows_a)
        component_rows.extend(rows_b)
    cluster_df = pd.DataFrame(cluster_rows)
    components_df = pd.DataFrame(component_rows)
    cluster_df.to_csv(os.path.join(args.out_dir, "representative_tile_cluster_profile.csv"), index=False)
    components_df.to_csv(os.path.join(args.out_dir, "representative_tile_component_sizes.csv"), index=False)

    plot_tile_gain_bubble(tile_gain, os.path.join(figures_dir, "tile_gain_bubble.png"))
    plot_regime_micro_gain(regime_gain, os.path.join(figures_dir, "regime_micro_gain.png"))
    if not components_df.empty:
        plot_component_hist(components_df, representative_tile, os.path.join(figures_dir, "representative_tile_component_hist.png"))

    summary = {
        "representative_tile": representative_tile,
        "n_tiles": int(tile_gain["tile_id"].nunique()),
        "best_micro_gain_tile": representative_tile,
        "best_micro_gain": float(tile_gain.iloc[0]["micro_gain"]),
        "mean_micro_gain": float(tile_gain["micro_gain"].mean()),
        "mean_macro_gain": float(tile_gain["macro_gain"].mean()),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
