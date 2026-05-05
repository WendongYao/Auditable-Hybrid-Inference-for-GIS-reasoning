# scripts/egms_autolabel_trend_qa_v5_8.py
# v5_8: fix step-label out-of-window by detecting steps on TRAIN window and (optionally) slicing prompts around the step.
#
# Usage:
#   python scripts/egms_autolabel_trend_qa_v5_8.py `
#     --input_dir data/raw `
#     --pattern "EGMS_L3_*_100km_U_2019_2023_*.csv" `
#     --dev_ratio 0.1 --test_ratio 0.1 `
#     --balanced_eval_per_class 2000 `
#     --chunksize 20000 `
#     --out_dir egmsqa/out_qa_v5_8 `
#     --seed 0 --max_train_per_tile 5000 --min_train_obs 30 `
#     --prompt_points 64 --series_mode last_n `
#     --step_series_mode around_step
#
# Output:
#   out_dir/jsonl/train.jsonl
#   out_dir/jsonl/dev.jsonl
#   out_dir/jsonl/test.jsonl
#   out_dir/jsonl/dev_balanced.jsonl
#   out_dir/jsonl/test_balanced.jsonl
#   out_dir/stats/*.csv

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


DATE_RE = re.compile(r"^\d{8}$")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_dir(os.path.dirname(path))
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    try:
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        return float(s)
    except Exception:
        return None


def year_fracs(date_cols: List[str]) -> np.ndarray:
    dts = [datetime.strptime(d, "%Y%m%d") for d in date_cols]
    t0 = dts[0]
    return np.array([(dt - t0).days / 365.25 for dt in dts], dtype=np.float64)


def fit_linear(t: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def fit_harmonic(t: np.ndarray, y: np.ndarray, period_years: float = 1.0) -> Tuple[float, float, float]:
    w = 2.0 * math.pi / period_years
    A = np.vstack([np.sin(w * t), np.cos(w * t), np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b, c = float(coef[0]), float(coef[1]), float(coef[2])
    amp = math.sqrt(a * a + b * b)
    return amp, c, w


def trend_dir_label(slope_mm_per_year: float, tol: float = 0.2) -> str:
    if slope_mm_per_year >= tol:
        return "uplifting"
    if slope_mm_per_year <= -tol:
        return "subsiding"
    return "stable"


def seasonality_label(amp_mm: float, tol: float = 0.5) -> str:
    return "seasonal" if amp_mm >= tol else "nonseasonal"


def detect_step(y: np.ndarray, date_cols: List[str], rmse_mm: float, min_jump_mm: float = 0.5) -> Tuple[str, str, float]:
    """
    Detect step on *this* window:
      - bucket_label: 'step' or 'no_step'
      - step_date: yyyymmdd of the RIGHT point of the jump
      - step_mag: signed jump (mm)
    """
    if y is None or len(y) < 2:
        return "no_step", "none", 0.0
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return "no_step", "none", 0.0

    dy = np.diff(y)
    ok = finite[:-1] & finite[1:]
    if ok.sum() == 0:
        return "no_step", "none", 0.0

    thr = max(float(min_jump_mm), 3.0 * float(rmse_mm))
    idxs = np.where(ok)[0]
    best = int(idxs[np.argmax(np.abs(dy[idxs]))])
    mag = float(dy[best])

    if abs(mag) < thr:
        return "no_step", "none", 0.0

    if best + 1 >= len(date_cols):
        return "no_step", "none", 0.0
    return "step", str(date_cols[best + 1]), mag


def build_prompt_series(date_cols: List[str], y: np.ndarray, prompt_points: int, series_mode: str) -> Dict[str, Any]:
    assert series_mode in {"last_n", "first_n"}
    T = len(date_cols)
    n = int(prompt_points)
    if T <= n:
        start, end = 0, T
    else:
        if series_mode == "last_n":
            start, end = T - n, T
        else:
            start, end = 0, n

    dates = date_cols[start:end]
    yy = np.asarray(y[start:end], dtype=np.float64)

    d_mm: List[Optional[float]] = []
    mask: List[int] = []
    for v in yy.tolist():
        if v is None or (isinstance(v, float) and (not math.isfinite(v))):
            d_mm.append(None)
            mask.append(0)
        else:
            d_mm.append(float(v))
            mask.append(1)

    return {
        "t_iso": [datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d") for d in dates],
        "d_mm": d_mm,
        "mask": mask,
    }


def build_prompt_series_around_step(date_cols: List[str], y: np.ndarray, step_date: str, prompt_points: int) -> Dict[str, Any]:
    T = len(date_cols)
    n = int(prompt_points)
    if T <= n:
        return build_prompt_series(date_cols, y, prompt_points, "last_n")
    try:
        idx_right = date_cols.index(step_date)
    except ValueError:
        return build_prompt_series(date_cols, y, prompt_points, "last_n")

    left = n // 2
    start = idx_right - left
    start = max(0, min(start, T - n))
    end = start + n

    dates = date_cols[start:end]
    yy = np.asarray(y[start:end], dtype=np.float64)

    d_mm: List[Optional[float]] = []
    mask: List[int] = []
    for v in yy.tolist():
        if v is None or (isinstance(v, float) and (not math.isfinite(v))):
            d_mm.append(None)
            mask.append(0)
        else:
            d_mm.append(float(v))
            mask.append(1)

    return {
        "t_iso": [datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d") for d in dates],
        "d_mm": d_mm,
        "mask": mask,
    }


@dataclass
class Reservoir:
    limit: int
    rng: random.Random
    items: List[Dict[str, Any]]
    seen: int = 0

    def add(self, x: Dict[str, Any]) -> None:
        self.seen += 1
        if self.limit <= 0:
            return
        if len(self.items) < self.limit:
            self.items.append(x)
            return
        j = self.rng.randint(0, self.seen - 1)
        if j < self.limit:
            self.items[j] = x


class BalancedReservoirMap:
    def __init__(self, per_class: int, seed: int):
        self.per_class = int(per_class)
        self.seed = int(seed)
        self._map: Dict[Tuple[str, str], Reservoir] = {}

    def add(self, ex: Dict[str, Any]) -> None:
        k = (ex.get("task_name", ""), ex.get("bucket_label", ""))
        if k not in self._map:
            rs = random.Random(self.seed ^ hash(k))
            self._map[k] = Reservoir(limit=self.per_class, rng=rs, items=[])
        self._map[k].add(ex)

    def collect_items(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _, r in sorted(self._map.items(), key=lambda kv: kv[0]):
            out.extend(r.items)
        return out

    def shortfall_rows(self, split_name: str) -> List[Dict[str, Any]]:
        rows = []
        for (task, bucket), r in sorted(self._map.items(), key=lambda kv: kv[0]):
            if len(r.items) < self.per_class:
                rows.append({
                    "split": split_name,
                    "task_name": task,
                    "bucket_label": bucket,
                    "kept": len(r.items),
                    "target": self.per_class,
                    "seen": r.seen,
                })
        return rows


def attach_context(ex: Dict[str, Any], series_ctx: Dict[str, Any], static_ctx: Dict[str, Any], meta_ctx: Dict[str, Any]) -> None:
    ex["context"] = {
        "series": series_ctx,
        "static": static_ctx,
        "meta": meta_ctx,
    }


def build_examples_for_pid(
    tile_id: str,
    pid: str,
    coord: Dict[str, Any],
    static_ctx: Dict[str, Any],
    meta_ctx_base: Dict[str, Any],
    date_cols: List[str],
    y_full: np.ndarray,
    train_epochs: int,
    prompt_points: int,
    series_mode: str,
    step_series_mode: str,
    tol_v: float = 0.2,
    tol_amp: float = 0.5,
    tol_step_mag: float = 0.5,
) -> List[Dict[str, Any]]:
    T = len(date_cols)
    t_train = int(train_epochs)
    if t_train < 2:
        return []

    date_cols_train = date_cols[:t_train]
    y_train = np.asarray(y_full[:t_train], dtype=np.float64)

    t_years = year_fracs(date_cols_train)
    finite = np.isfinite(y_train)
    t_fit = t_years[finite]
    y_fit = y_train[finite]
    if len(y_fit) < 3:
        return []

    slope, _ = fit_linear(t_fit, y_fit)
    trend_bucket = trend_dir_label(slope, tol=tol_v)

    amp, _, _ = fit_harmonic(t_fit, y_fit, period_years=1.0)
    seas_bucket = seasonality_label(amp, tol=tol_amp)

    rmse_mm = float(static_ctx.get("rmse_mm", 1.0))
    step_bucket, step_date, step_mag = detect_step(y_train, date_cols_train, rmse_mm, min_jump_mm=tol_step_mag)

    series_ctx_default = build_prompt_series(date_cols_train, y_train, prompt_points, series_mode)
    if step_series_mode == "around_step" and step_bucket == "step" and step_date != "none":
        series_ctx_step = build_prompt_series_around_step(date_cols_train, y_train, step_date, prompt_points)
        step_series_mode_used = "around_step"
    else:
        series_ctx_step = series_ctx_default
        step_series_mode_used = series_mode

    meta_ctx_common = dict(meta_ctx_base)
    meta_ctx_common.update({
        "train_epochs": int(t_train),
        "total_epochs": int(T),
        "prompt_points": int(prompt_points),
    })

    def mk_id(task_name: str) -> str:
        return f"{tile_id}_{pid}_{task_name}"

    out: List[Dict[str, Any]] = []

    # IMPORTANT: metrics here match v5_6/score.py exactly.
    ex = {
        "tile_id": tile_id, "pid": pid, "coord": coord,
        "meta": {**static_ctx, "train_epochs": int(t_train), "n_obs_train": int(finite.sum())},
        "id": mk_id("A1_trend_dir"),
        "task_name": "A1_trend_dir",
        "bucket_label": trend_bucket,
        "question": "What is the overall trend direction? Choose one: subsiding, uplifting, stable.",
        "answer": trend_bucket,
        "eval": {"metric": "class_exact"},
    }
    attach_context(ex, series_ctx_default, static_ctx, {**meta_ctx_common, "window": "train_only", "series_mode": series_mode})
    out.append(ex)

    ex = {
        "tile_id": tile_id, "pid": pid, "coord": coord,
        "meta": {**static_ctx, "train_epochs": int(t_train), "n_obs_train": int(finite.sum())},
        "id": mk_id("A2_trend_v"),
        "task_name": "A2_trend_v",
        "bucket_label": trend_bucket,
        "question": "What is the mean linear velocity (mm/year)? Give a single number with 1 decimal.",
        "answer": f"{slope:.1f}",
        "answer_numeric": float(f"{slope:.1f}"),
        "eval": {"metric": "num_within_tol", "tolerance": {"type": "abs", "value": float(tol_v)}},
    }
    attach_context(ex, series_ctx_default, static_ctx, {**meta_ctx_common, "window": "train_only", "series_mode": series_mode})
    out.append(ex)

    ex = {
        "tile_id": tile_id, "pid": pid, "coord": coord,
        "meta": {**static_ctx, "train_epochs": int(t_train), "n_obs_train": int(finite.sum())},
        "id": mk_id("B1_seasonality_present"),
        "task_name": "B1_seasonality_present",
        "bucket_label": seas_bucket,
        "question": "Is there clear annual seasonality? Choose one: seasonal, nonseasonal.",
        "answer": seas_bucket,
        "eval": {"metric": "class_exact"},
    }
    attach_context(ex, series_ctx_default, static_ctx, {**meta_ctx_common, "window": "train_only", "series_mode": series_mode})
    out.append(ex)

    amp_ans = "none" if seas_bucket == "nonseasonal" else f"{amp:.1f}"
    ex = {
        "tile_id": tile_id, "pid": pid, "coord": coord,
        "meta": {**static_ctx, "train_epochs": int(t_train), "n_obs_train": int(finite.sum())},
        "id": mk_id("B2_seasonality_amp"),
        "task_name": "B2_seasonality_amp",
        "bucket_label": seas_bucket,
        "question": "If seasonal, what is the annual amplitude (mm)? Otherwise answer none.",
        "answer": amp_ans,
        "answer_numeric": (None if amp_ans == "none" else float(amp_ans)),
        "eval": {"metric": "num_or_none_within_tol", "tolerance": {"type": "abs", "value": float(tol_amp)}},
    }
    attach_context(ex, series_ctx_default, static_ctx, {**meta_ctx_common, "window": "train_only", "series_mode": series_mode})
    out.append(ex)

    ex = {
        "tile_id": tile_id, "pid": pid, "coord": coord,
        "meta": {**static_ctx, "train_epochs": int(t_train), "n_obs_train": int(finite.sum())},
        "id": mk_id("C1_step_time"),
        "task_name": "C1_step_time",
        "bucket_label": step_bucket,
        "question": "Is there an abrupt step change in the series? If yes, approximately when (yyyymmdd); otherwise none.",
        "answer": step_date if step_bucket == "step" else "none",
        "eval": {"metric": "time_or_none_within_k", "tolerance": {"type": "epoch", "value": 1}},
    }
    attach_context(ex, series_ctx_step, static_ctx, {**meta_ctx_common, "window": "train_only", "series_mode": step_series_mode_used})
    out.append(ex)

    mag_ans = "none" if step_bucket == "no_step" else f"{step_mag:.1f}"
    ex = {
        "tile_id": tile_id, "pid": pid, "coord": coord,
        "meta": {**static_ctx, "train_epochs": int(t_train), "n_obs_train": int(finite.sum())},
        "id": mk_id("C2_step_mag"),
        "task_name": "C2_step_mag",
        "bucket_label": step_bucket,
        "question": "If there is a step, what is its magnitude (mm)? Otherwise answer none.",
        "answer": mag_ans,
        "answer_numeric": (None if mag_ans == "none" else float(mag_ans)),
        "eval": {"metric": "num_or_none_within_tol", "tolerance": {"type": "abs", "value": float(tol_step_mag)}},
    }
    attach_context(ex, series_ctx_step, static_ctx, {**meta_ctx_common, "window": "train_only", "series_mode": step_series_mode_used})
    out.append(ex)

    return out


def parse_date_cols_from_df_columns(cols: List[str]) -> List[str]:
    return [c for c in cols if isinstance(c, str) and DATE_RE.match(c)]


def assign_split(tile_id: str, pid: str, seed: int, dev_ratio: float, test_ratio: float) -> str:
    key = f"{tile_id}:{pid}:{seed}"
    r = random.Random(seed ^ hash(key))
    u = r.random()
    if u < dev_ratio:
        return "dev"
    if u < dev_ratio + test_ratio:
        return "test"
    return "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--balanced_eval_per_class", type=int, default=2000)

    ap.add_argument("--chunksize", type=int, default=20000)
    ap.add_argument("--max_train_per_tile", type=int, default=5000)
    ap.add_argument("--min_train_obs", type=int, default=30)

    ap.add_argument("--prompt_points", type=int, default=64)
    ap.add_argument("--series_mode", choices=["last_n", "first_n"], default="last_n")

    ap.add_argument("--step_series_mode", choices=["last_n", "around_step"], default="last_n",
                    help="Only affects C1/C2 prompts when bucket_label=step.")

    args = ap.parse_args()

    in_paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not in_paths:
        raise SystemExit(f"No files matched: {os.path.join(args.input_dir, args.pattern)}")

    out_dir = args.out_dir
    out_jsonl = os.path.join(out_dir, "jsonl")
    out_stats = os.path.join(out_dir, "stats")
    ensure_dir(out_jsonl)
    ensure_dir(out_stats)

    train_path = os.path.join(out_jsonl, "train.jsonl")
    dev_path = os.path.join(out_jsonl, "dev.jsonl")
    test_path = os.path.join(out_jsonl, "test.jsonl")
    devb_path = os.path.join(out_jsonl, "dev_balanced.jsonl")
    testb_path = os.path.join(out_jsonl, "test_balanced.jsonl")

    for p in [train_path, dev_path, test_path, devb_path, testb_path]:
        if os.path.exists(p):
            os.remove(p)

    dev_res_map = BalancedReservoirMap(per_class=args.balanced_eval_per_class, seed=args.seed + 123)
    test_res_map = BalancedReservoirMap(per_class=args.balanced_eval_per_class, seed=args.seed + 456)

    split_counts = Counter()
    bucket_counts = Counter()

    for csv_path in in_paths:
        base = os.path.basename(csv_path)
        m = re.search(r"(E\d+N\d+)", base)
        tile_id = m.group(1) if m else base

        head = pd.read_csv(csv_path, nrows=1)
        date_cols = parse_date_cols_from_df_columns(list(head.columns))
        if not date_cols:
            raise SystemExit(f"[{tile_id}] No date columns found in {csv_path}")

        T = len(date_cols)
        t_train = int(0.8 * T)
        if t_train < 2:
            raise SystemExit(f"[{tile_id}] Too few epochs (T={T})")

        train_written_this_tile = 0

        reader = pd.read_csv(csv_path, chunksize=args.chunksize)
        it = reader if tqdm is None else tqdm(reader, desc=f"{tile_id}", unit="chunk")

        for chunk in it:
            if "pid" not in chunk.columns:
                raise SystemExit(f"[{tile_id}] column 'pid' not found in CSV")

            y_train_mat = chunk[date_cols[:t_train]].to_numpy(dtype=np.float64, copy=False)
            obs_train = np.isfinite(y_train_mat).sum(axis=1)

            elig_mask = obs_train >= int(args.min_train_obs)
            if not np.any(elig_mask):
                continue

            sub = chunk.loc[elig_mask].copy()
            sub_obs_train = obs_train[elig_mask]

            for (idx, row), n_obs in zip(sub.iterrows(), sub_obs_train.tolist()):
                pid = str(row["pid"])
                sp = assign_split(tile_id, pid, args.seed, args.dev_ratio, args.test_ratio)

                if sp == "train" and train_written_this_tile >= int(args.max_train_per_tile):
                    continue

                static_ctx = {
                    "rmse_mm": safe_float(row.get("rmse_mm")) if "rmse_mm" in row else safe_float(row.get("rmse")),
                    "mean_velocity_std": safe_float(row.get("mean_velocity_std")),
                    "seasonality_std": safe_float(row.get("seasonality_std")),
                }
                for k in ["rmse_mm", "mean_velocity_std", "seasonality_std"]:
                    if static_ctx[k] is None:
                        static_ctx[k] = 0.0

                coord = {
                    "easting": int(row.get("easting")) if "easting" in row and not pd.isna(row.get("easting")) else None,
                    "northing": int(row.get("northing")) if "northing" in row and not pd.isna(row.get("northing")) else None,
                }

                meta_base = {"window": "train_only"}

                y_full = row[date_cols].to_numpy(dtype=np.float64, copy=False)

                ex_list = build_examples_for_pid(
                    tile_id=tile_id,
                    pid=pid,
                    coord=coord,
                    static_ctx=static_ctx,
                    meta_ctx_base=meta_base,
                    date_cols=date_cols,
                    y_full=y_full,
                    train_epochs=t_train,
                    prompt_points=args.prompt_points,
                    series_mode=args.series_mode,
                    step_series_mode=args.step_series_mode,
                )
                if not ex_list:
                    continue

                if sp == "train":
                    append_jsonl(train_path, ex_list)
                    train_written_this_tile += 1
                elif sp == "dev":
                    append_jsonl(dev_path, ex_list)
                    for ex in ex_list:
                        dev_res_map.add(ex)
                else:
                    append_jsonl(test_path, ex_list)
                    for ex in ex_list:
                        test_res_map.add(ex)

                split_counts[sp] += len(ex_list)
                for ex in ex_list:
                    bucket_counts[(ex["task_name"], ex["bucket_label"])] += 1

    dev_balanced = dev_res_map.collect_items()
    test_balanced = test_res_map.collect_items()
    write_jsonl(devb_path, dev_balanced)
    write_jsonl(testb_path, test_balanced)

    shortfall_rows = dev_res_map.shortfall_rows("dev_balanced") + test_res_map.shortfall_rows("test_balanced")
    pd.DataFrame(shortfall_rows).to_csv(os.path.join(out_stats, "balanced_shortfall.csv"), index=False)

    pd.DataFrame([{"split": k, "n": int(v)} for k, v in sorted(split_counts.items())]).to_csv(
        os.path.join(out_stats, "split_counts.csv"), index=False
    )
    pd.DataFrame([{"task_name": k[0], "bucket_label": k[1], "n": int(v)} for k, v in sorted(bucket_counts.items())]).to_csv(
        os.path.join(out_stats, "bucket_counts.csv"), index=False
    )

    print(f"DONE: {out_dir}")
    print(f"  train: {split_counts['train']}  dev: {split_counts['dev']}  test: {split_counts['test']}  dev_balanced: {len(dev_balanced)}")


if __name__ == "__main__":
    main()
