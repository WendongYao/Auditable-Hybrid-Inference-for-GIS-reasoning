#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_slope(x: pd.Series, y: pd.Series) -> Optional[float]:
    if len(x) < 2:
        return None
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if np.allclose(xv, xv[0]):
        return None
    coef = np.polyfit(xv, yv, deg=1)
    return float(coef[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition_manifest_csv", required=True)
    ap.add_argument("--score_root", required=True)
    ap.add_argument("--score_subdir", default="score")
    ap.add_argument("--system_label", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    manifest = pd.read_csv(args.condition_manifest_csv)

    overall_rows: List[Dict[str, Any]] = []
    per_task_rows: List[Dict[str, Any]] = []

    for _, row in manifest.iterrows():
        cond = row["condition_id"]
        score_dir = os.path.join(args.score_root, cond, args.score_subdir)
        summary_path = os.path.join(score_dir, "summary.json")
        by_task_path = os.path.join(score_dir, "by_task.csv")
        if not os.path.exists(summary_path) or not os.path.exists(by_task_path):
            continue

        summary = load_json(summary_path)
        overall_rows.append(
            {
                "system": args.system_label,
                **row.to_dict(),
                "micro_acc": summary.get("micro_acc"),
                "macro_acc": summary.get("macro_acc_over_tasks"),
                "n_scored": summary.get("n_scored"),
            }
        )

        by_task = pd.read_csv(by_task_path)
        by_task["system"] = args.system_label
        by_task["condition_id"] = cond
        by_task["family"] = row["family"]
        by_task["severity"] = row["severity"]
        by_task["seed_idx"] = row["seed_idx"]
        by_task["mean_missing_rate"] = row["mean_missing_rate"]
        by_task["mean_observed_points"] = row["mean_observed_points"]
        per_task_rows.append(by_task)

    overall_df = pd.DataFrame(overall_rows)
    per_task_df = pd.concat(per_task_rows, ignore_index=True) if per_task_rows else pd.DataFrame()
    if overall_df.empty or per_task_df.empty:
        raise RuntimeError("No score outputs found to summarize.")

    clean_overall = overall_df[overall_df["condition_id"] == "clean"].iloc[0]
    overall_df["delta_micro_vs_clean"] = overall_df["micro_acc"] - float(clean_overall["micro_acc"])
    overall_df["delta_macro_vs_clean"] = overall_df["macro_acc"] - float(clean_overall["macro_acc"])

    clean_task = per_task_df[per_task_df["condition_id"] == "clean"][["task_name", "acc"]].rename(columns={"acc": "clean_acc"})
    per_task_df = per_task_df.merge(clean_task, on="task_name", how="left")
    per_task_df["delta_vs_clean"] = per_task_df["acc"] - per_task_df["clean_acc"]

    overall_df.to_csv(os.path.join(args.out_dir, "overall_by_condition.csv"), index=False)
    per_task_df.to_csv(os.path.join(args.out_dir, "per_task_by_condition.csv"), index=False)

    agg_overall = (
        overall_df[overall_df["family"] != "clean"]
        .groupby(["system", "family", "severity"], sort=True)
        .agg(
            n_runs=("condition_id", "size"),
            mean_missing_rate=("mean_missing_rate", "mean"),
            mean_observed_points=("mean_observed_points", "mean"),
            mean_micro_acc=("micro_acc", "mean"),
            std_micro_acc=("micro_acc", "std"),
            mean_macro_acc=("macro_acc", "mean"),
            std_macro_acc=("macro_acc", "std"),
            mean_delta_micro_vs_clean=("delta_micro_vs_clean", "mean"),
            std_delta_micro_vs_clean=("delta_micro_vs_clean", "std"),
            mean_delta_macro_vs_clean=("delta_macro_vs_clean", "mean"),
            std_delta_macro_vs_clean=("delta_macro_vs_clean", "std"),
        )
        .reset_index()
    )
    agg_overall.to_csv(os.path.join(args.out_dir, "family_overall_agg.csv"), index=False)

    agg_task = (
        per_task_df[per_task_df["family"] != "clean"]
        .groupby(["system", "family", "severity", "task_name"], sort=True)
        .agg(
            n_runs=("condition_id", "size"),
            mean_missing_rate=("mean_missing_rate", "mean"),
            mean_observed_points=("mean_observed_points", "mean"),
            mean_acc=("acc", "mean"),
            std_acc=("acc", "std"),
            mean_delta_vs_clean=("delta_vs_clean", "mean"),
            std_delta_vs_clean=("delta_vs_clean", "std"),
        )
        .reset_index()
    )
    agg_task.to_csv(os.path.join(args.out_dir, "family_task_agg.csv"), index=False)

    extreme_rows: List[Dict[str, Any]] = []
    for family, sub in agg_task.groupby("family", sort=True):
        max_sev = float(sub["severity"].max())
        sub_ext = sub[sub["severity"] == max_sev]
        for _, row in sub_ext.iterrows():
            extreme_rows.append(
                {
                    "system": row["system"],
                    "family": family,
                    "extreme_severity": max_sev,
                    "task_name": row["task_name"],
                    "mean_acc_at_extreme": row["mean_acc"],
                    "mean_delta_vs_clean_at_extreme": row["mean_delta_vs_clean"],
                }
            )
    pd.DataFrame(extreme_rows).to_csv(os.path.join(args.out_dir, "clean_to_extreme_task_drop.csv"), index=False)

    slope_rows: List[Dict[str, Any]] = []
    for family, sub in agg_overall.groupby("family", sort=True):
        slope_rows.append(
            {
                "system": args.system_label,
                "family": family,
                "task_name": "__overall__",
                "metric": "macro_acc",
                "slope_per_severity_unit": safe_slope(sub["severity"], sub["mean_macro_acc"]),
            }
        )
        slope_rows.append(
            {
                "system": args.system_label,
                "family": family,
                "task_name": "__overall__",
                "metric": "micro_acc",
                "slope_per_severity_unit": safe_slope(sub["severity"], sub["mean_micro_acc"]),
            }
        )
    for (family, task_name), sub in agg_task.groupby(["family", "task_name"], sort=True):
        slope_rows.append(
            {
                "system": args.system_label,
                "family": family,
                "task_name": task_name,
                "metric": "acc",
                "slope_per_severity_unit": safe_slope(sub["severity"], sub["mean_acc"]),
            }
        )
    pd.DataFrame(slope_rows).to_csv(os.path.join(args.out_dir, "robustness_slopes.csv"), index=False)

    summary = {
        "system": args.system_label,
        "n_conditions": int(overall_df["condition_id"].nunique()),
        "clean_micro_acc": float(clean_overall["micro_acc"]),
        "clean_macro_acc": float(clean_overall["macro_acc"]),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved robustness summary to:", args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
