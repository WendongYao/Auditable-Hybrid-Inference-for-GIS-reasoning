#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DATE_FMT_CANDIDATES = ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d")
SPATIAL_TASKS_DEFAULT = "A1_trend_dir,B1_seasonality_present,C1_step_time,C2_step_mag"


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_date(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    for fmt in DATE_FMT_CANDIDATES:
        try:
            return datetime.strptime(ss, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(ss)
    except Exception:
        return None


def try_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        xf = float(x)
        return xf if math.isfinite(xf) else None
    s = str(x).strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return None
    try:
        xf = float(s)
        return xf if math.isfinite(xf) else None
    except Exception:
        return None


def median_cadence_days(t_iso: List[Any]) -> Optional[float]:
    dts = [parse_date(x) for x in t_iso]
    dts = [dt for dt in dts if dt is not None]
    if len(dts) < 2:
        return None
    diffs = []
    for a, b in zip(dts[:-1], dts[1:]):
        diffs.append((b - a).days)
    if not diffs:
        return None
    return float(np.median(np.asarray(diffs, dtype=np.float64)))


def count_observed(mask: List[Any], d_mm: List[Any]) -> int:
    if mask:
        out = 0
        for x in mask:
            try:
                out += int(float(x) > 0)
            except Exception:
                pass
        return int(out)
    out = 0
    for x in d_mm:
        if try_float(x) is not None:
            out += 1
    return int(out)


def infer_pred_bucket(task_name: str, pred_answer: Any) -> str:
    if task_name == "A1_trend_dir":
        s = str(pred_answer).strip().lower() if pred_answer is not None else ""
        return s if s in {"subsiding", "uplifting", "stable"} else "unknown"

    if task_name == "A2_trend_v":
        v = try_float(pred_answer)
        if v is None:
            return "unknown"
        if v >= 1.0:
            return "uplifting"
        if v <= -1.0:
            return "subsiding"
        return "stable"

    if task_name == "B1_seasonality_present":
        s = str(pred_answer).strip().lower() if pred_answer is not None else ""
        if s in {"seasonal", "nonseasonal"}:
            return s
        return "unknown"

    if task_name == "B2_seasonality_amp":
        if pred_answer is None:
            return "nonseasonal"
        s = str(pred_answer).strip().lower()
        if s == "none":
            return "nonseasonal"
        if try_float(pred_answer) is not None:
            return "seasonal"
        return "unknown"

    if task_name == "C1_step_time":
        if pred_answer is None:
            return "no_step"
        s = str(pred_answer).strip().lower()
        if s in {"none", "no_step", "nostep", ""}:
            return "no_step"
        return "step"

    if task_name == "C2_step_mag":
        if pred_answer is None:
            return "no_step"
        s = str(pred_answer).strip().lower()
        if s in {"none", "no_step", "nostep", ""}:
            return "no_step"
        if try_float(pred_answer) is not None:
            return "step"
        return "unknown"

    return "unknown"


def rmse_bin(x: Any) -> str:
    v = try_float(x)
    if v is None:
        return "unknown"
    if v < 0.5:
        return "[0,0.5)"
    if v < 1.0:
        return "[0.5,1.0)"
    if v < 2.0:
        return "[1.0,2.0)"
    return "[2.0,+inf)"


def missing_bin(x: Any) -> str:
    v = try_float(x)
    if v is None:
        return "unknown"
    if v < 0.10:
        return "[0.00,0.10)"
    if v < 0.25:
        return "[0.10,0.25)"
    if v < 0.40:
        return "[0.25,0.40)"
    return "[0.40,1.00]"


def obs_bin(x: Any) -> str:
    v = try_float(x)
    if v is None:
        return "unknown"
    if v < 40:
        return "[0,40)"
    if v < 52:
        return "[40,52)"
    if v < 60:
        return "[52,60)"
    return "[60,64]"


def cadence_bin(x: Any) -> str:
    v = try_float(x)
    if v is None:
        return "unknown"
    if v < 7:
        return "[0,7)"
    if v < 10:
        return "[7,10)"
    if v < 14:
        return "[10,14)"
    return "[14,+inf)"


def extract_dataset_metadata(dataset_jsonl: str, out_csv: str) -> None:
    ensure_dir(os.path.dirname(out_csv))
    fields = [
        "id",
        "tile_id",
        "pid",
        "task_name",
        "bucket_label",
        "gold_answer",
        "easting",
        "northing",
        "rmse_mm",
        "observed_points",
        "total_points",
        "missing_rate",
        "median_cadence_days",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()
        for ex in read_jsonl(dataset_jsonl):
            ctx = ex.get("context") or {}
            series = ctx.get("series") or {}
            static = ctx.get("static") or {}
            coord = ex.get("coord") or {}

            t_iso = series.get("t_iso") or []
            d_mm = series.get("d_mm") or []
            mask = series.get("mask") or []
            total_points = int(len(t_iso))
            observed_points = count_observed(mask, d_mm)
            missing_rate = None
            if total_points > 0:
                missing_rate = 1.0 - (float(observed_points) / float(total_points))

            row = {
                "id": ex.get("id"),
                "tile_id": ex.get("tile_id"),
                "pid": ex.get("pid"),
                "task_name": ex.get("task_name") or ex.get("task"),
                "bucket_label": ex.get("bucket_label"),
                "gold_answer": ex.get("answer"),
                "easting": coord.get("easting"),
                "northing": coord.get("northing"),
                "rmse_mm": static.get("rmse_mm"),
                "observed_points": observed_points,
                "total_points": total_points,
                "missing_rate": missing_rate,
                "median_cadence_days": median_cadence_days(t_iso),
            }
            writer.writerow(row)


def attach_regime(df: pd.DataFrame) -> pd.DataFrame:
    a1 = df[df["task_name"] == "A1_trend_dir"]
    tile_major = (
        a1.groupby(["tile_id", "bucket_label"])
        .size()
        .reset_index(name="n")
        .sort_values(["tile_id", "n", "bucket_label"], ascending=[True, False, True])
        .drop_duplicates(subset=["tile_id"])
    )
    regime_map = {
        row["tile_id"]: f"{row['bucket_label']}_dominated"
        for _, row in tile_major.iterrows()
    }
    df["regime"] = df["tile_id"].map(regime_map).fillna("unknown")
    return df


def summarise_micro_macro(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if group_cols:
        grouped = df.groupby(group_cols, sort=True)
    else:
        grouped = [((), df)]

    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict[str, Any] = {}
        for col, val in zip(group_cols, key):
            row[col] = val
        by_task = sub.groupby("task_name")["ok"].mean()
        row.update(
            {
                "n": int(len(sub)),
                "n_pid": int(sub["pid"].nunique()),
                "micro_acc": float(sub["ok"].mean()),
                "macro_acc": float(by_task.mean()) if len(by_task) else float("nan"),
                "n_tasks": int(len(by_task)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_task_accuracy(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols + ["task_name"], sort=True)
        .agg(
            n=("ok", "size"),
            n_pid=("pid", "nunique"),
            acc=("ok", "mean"),
        )
        .reset_index()
    )
    return grouped


def build_tile_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for tile_id, sub in df.groupby("tile_id", sort=True):
        row: Dict[str, Any] = {
            "tile_id": tile_id,
            "n_examples": int(len(sub)),
            "n_pid": int(sub["pid"].nunique()),
        }

        a1 = sub[sub["task_name"] == "A1_trend_dir"]
        for label in ["subsiding", "stable", "uplifting"]:
            row[f"A1_gold_frac_{label}"] = float((a1["bucket_label"] == label).mean()) if len(a1) else float("nan")
            row[f"A1_pred_frac_{label}"] = float((a1["pred_bucket"] == label).mean()) if len(a1) else float("nan")

        a2 = sub[sub["task_name"] == "A2_trend_v"].copy()
        a2["gold_num"] = a2["gold_answer"].map(try_float)
        a2["pred_num"] = a2["pred_answer"].map(try_float)
        a2_ok = a2.dropna(subset=["gold_num", "pred_num"])
        row["A2_gold_mean"] = float(a2["gold_num"].mean()) if len(a2) else float("nan")
        row["A2_pred_mean"] = float(a2["pred_num"].mean()) if len(a2_ok) else float("nan")
        row["A2_mae"] = float((a2_ok["pred_num"] - a2_ok["gold_num"]).abs().mean()) if len(a2_ok) else float("nan")

        b1 = sub[sub["task_name"] == "B1_seasonality_present"]
        row["B1_gold_seasonal_frac"] = float((b1["bucket_label"] == "seasonal").mean()) if len(b1) else float("nan")
        row["B1_pred_seasonal_frac"] = float((b1["pred_bucket"] == "seasonal").mean()) if len(b1) else float("nan")

        b2 = sub[sub["task_name"] == "B2_seasonality_amp"].copy()
        b2["gold_num"] = b2["gold_answer"].map(try_float)
        b2["pred_num"] = b2["pred_answer"].map(try_float)
        b2_seasonal = b2[b2["bucket_label"] == "seasonal"]
        b2_overlap = b2_seasonal.dropna(subset=["gold_num", "pred_num"])
        row["B2_gold_mean_seasonal_amp"] = float(b2_seasonal["gold_num"].mean()) if len(b2_seasonal) else float("nan")
        row["B2_pred_mean_seasonal_amp"] = float(b2_overlap["pred_num"].mean()) if len(b2_overlap) else float("nan")
        row["B2_mae_seasonal_amp"] = float((b2_overlap["pred_num"] - b2_overlap["gold_num"]).abs().mean()) if len(b2_overlap) else float("nan")

        c1 = sub[sub["task_name"] == "C1_step_time"]
        row["C1_gold_step_frac"] = float((c1["bucket_label"] == "step").mean()) if len(c1) else float("nan")
        row["C1_pred_step_frac"] = float((c1["pred_bucket"] == "step").mean()) if len(c1) else float("nan")

        c2 = sub[sub["task_name"] == "C2_step_mag"].copy()
        c2["gold_num"] = c2["gold_answer"].map(try_float)
        c2["pred_num"] = c2["pred_answer"].map(try_float)
        c2_step = c2[c2["bucket_label"] == "step"]
        c2_overlap = c2_step.dropna(subset=["gold_num", "pred_num"])
        row["C2_gold_step_frac"] = float((c2["bucket_label"] == "step").mean()) if len(c2) else float("nan")
        row["C2_pred_step_frac"] = float((c2["pred_bucket"] == "step").mean()) if len(c2) else float("nan")
        row["C2_gold_mean_step_mag"] = float(c2_step["gold_num"].mean()) if len(c2_step) else float("nan")
        row["C2_pred_mean_step_mag"] = float(c2_overlap["pred_num"].mean()) if len(c2_overlap) else float("nan")
        row["C2_mae_step_mag"] = float((c2_overlap["pred_num"] - c2_overlap["gold_num"]).abs().mean()) if len(c2_overlap) else float("nan")

        rows.append(row)
    return pd.DataFrame(rows)


def mode_label(labels: List[str]) -> Optional[str]:
    if not labels:
        return None
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def compute_spatial_consistency(
    df: pd.DataFrame,
    tasks: List[str],
    knn: int,
    max_points: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for task_name in tasks:
        sub_task = df[df["task_name"] == task_name].copy()
        if sub_task.empty:
            continue
        sub_task = sub_task.dropna(subset=["easting", "northing"])
        if sub_task.empty:
            continue

        for tile_id, sub in sub_task.groupby("tile_id", sort=True):
            if len(sub) <= knn or len(sub) > max_points:
                continue

            coords = sub[["easting", "northing"]].to_numpy(dtype=np.float64)
            gold = sub["bucket_label"].astype(str).tolist()
            pred = sub["pred_bucket"].astype(str).tolist()
            n = int(len(sub))
            if n <= knn:
                continue

            d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
            np.fill_diagonal(d2, np.inf)
            nn_idx = np.argpartition(d2, kth=knn - 1, axis=1)[:, :knn]

            gold_agree = []
            pred_agree = []
            anomaly = 0
            oversmooth = 0
            for i in range(n):
                neigh = nn_idx[i].tolist()
                neigh_gold = [gold[j] for j in neigh]
                neigh_pred = [pred[j] for j in neigh]
                gold_agree.append(float(np.mean([gold[j] == gold[i] for j in neigh])))
                pred_agree.append(float(np.mean([pred[j] == pred[i] for j in neigh])))

                majority_gold = mode_label(neigh_gold)
                if majority_gold is not None:
                    if gold[i] == majority_gold and pred[i] != majority_gold:
                        anomaly += 1
                    if gold[i] != majority_gold and pred[i] == majority_gold:
                        oversmooth += 1

            rows.append(
                {
                    "task_name": task_name,
                    "tile_id": tile_id,
                    "n_points": n,
                    "knn": int(knn),
                    "gold_neighbor_agreement": float(np.mean(gold_agree)),
                    "pred_neighbor_agreement": float(np.mean(pred_agree)),
                    "agreement_gap_pred_minus_gold": float(np.mean(pred_agree) - np.mean(gold_agree)),
                    "anomaly_rate": float(anomaly / n),
                    "oversmooth_rate": float(oversmooth / n),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--per_example_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--metadata_csv", default="")
    ap.add_argument("--run_spatial", action="store_true")
    ap.add_argument("--spatial_tasks", default=SPATIAL_TASKS_DEFAULT)
    ap.add_argument("--knn", type=int, default=4)
    ap.add_argument("--max_spatial_points", type=int, default=12000)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    metadata_csv = args.metadata_csv.strip() or os.path.join(args.out_dir, "dataset_metadata.csv")
    if not os.path.exists(metadata_csv):
        extract_dataset_metadata(args.dataset, metadata_csv)

    meta = pd.read_csv(metadata_csv)
    per_example = pd.read_csv(args.per_example_csv)
    if "parse_error" not in per_example.columns:
        per_example["parse_error"] = None
    if "reason" not in per_example.columns:
        per_example["reason"] = per_example["note"] if "note" in per_example.columns else None
    keep_cols = ["id", "task_name", "pred_answer", "parse_error", "ok", "reason"]
    per_example = per_example[keep_cols]

    df = meta.merge(per_example, on=["id", "task_name"], how="inner")
    df["pred_bucket"] = [infer_pred_bucket(t, a) for t, a in zip(df["task_name"], df["pred_answer"])]
    df["rmse_bin"] = df["rmse_mm"].map(rmse_bin)
    df["missing_bin"] = df["missing_rate"].map(missing_bin)
    df["obs_bin"] = df["observed_points"].map(obs_bin)
    df["cadence_bin"] = df["median_cadence_days"].map(cadence_bin)
    df = attach_regime(df)

    summary = {
        "label": args.label,
        "n_examples": int(len(df)),
        "n_pid": int(df["pid"].nunique()),
        "micro_acc": float(df["ok"].mean()),
        "macro_acc": float(df.groupby("task_name")["ok"].mean().mean()),
        "tiles": sorted(df["tile_id"].dropna().unique().tolist()),
        "regimes": sorted(df["regime"].dropna().unique().tolist()),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summarise_micro_macro(df, []).to_csv(os.path.join(args.out_dir, "overall.csv"), index=False)
    summarise_micro_macro(df, ["tile_id"]).to_csv(os.path.join(args.out_dir, "by_tile.csv"), index=False)
    summarise_micro_macro(df, ["regime"]).to_csv(os.path.join(args.out_dir, "by_regime.csv"), index=False)
    summarise_task_accuracy(df, ["tile_id"]).to_csv(os.path.join(args.out_dir, "by_tile_task.csv"), index=False)
    summarise_task_accuracy(df, ["regime"]).to_csv(os.path.join(args.out_dir, "by_regime_task.csv"), index=False)

    summarise_micro_macro(df, ["rmse_bin"]).to_csv(os.path.join(args.out_dir, "by_rmse_bin.csv"), index=False)
    summarise_micro_macro(df, ["missing_bin"]).to_csv(os.path.join(args.out_dir, "by_missing_bin.csv"), index=False)
    summarise_micro_macro(df, ["obs_bin"]).to_csv(os.path.join(args.out_dir, "by_obs_bin.csv"), index=False)
    summarise_micro_macro(df, ["cadence_bin"]).to_csv(os.path.join(args.out_dir, "by_cadence_bin.csv"), index=False)

    summarise_task_accuracy(df, ["rmse_bin"]).to_csv(os.path.join(args.out_dir, "by_rmse_bin_task.csv"), index=False)
    summarise_task_accuracy(df, ["missing_bin"]).to_csv(os.path.join(args.out_dir, "by_missing_bin_task.csv"), index=False)
    summarise_task_accuracy(df, ["obs_bin"]).to_csv(os.path.join(args.out_dir, "by_obs_bin_task.csv"), index=False)
    summarise_task_accuracy(df, ["cadence_bin"]).to_csv(os.path.join(args.out_dir, "by_cadence_bin_task.csv"), index=False)

    build_tile_level_summary(df).to_csv(os.path.join(args.out_dir, "tile_level_summary.csv"), index=False)

    if args.run_spatial:
        tasks = [t.strip() for t in args.spatial_tasks.split(",") if t.strip()]
        spatial = compute_spatial_consistency(
            df=df,
            tasks=tasks,
            knn=int(args.knn),
            max_points=int(args.max_spatial_points),
        )
        spatial.to_csv(os.path.join(args.out_dir, "spatial_consistency.csv"), index=False)

    print("Saved analysis to:", args.out_dir)
    print("Summary:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
