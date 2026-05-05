#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from render_prompt_routeA_a3 import render_prompt_routeA_a3  # noqa: E402


DATE_FMT_CANDIDATES = ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_date_any(s: Any) -> Optional[datetime]:
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


def finite_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return None
    return fv if math.isfinite(fv) else None


def valid_indices(d_mm: List[Any]) -> List[int]:
    out: List[int] = []
    for i, v in enumerate(d_mm):
        if finite_or_none(v) is not None:
            out.append(i)
    return out


def bucket_targets(task_name: str, total_n: int) -> Dict[str, int]:
    if task_name in {"A1_trend_dir", "A2_trend_v"}:
        base = total_n // 3
        rem = total_n - base * 3
        labels = ["stable", "subsiding", "uplifting"]
        out = {label: base for label in labels}
        for label in labels[:rem]:
            out[label] += 1
        return out
    if task_name in {"B1_seasonality_present", "B2_seasonality_amp"}:
        return {"nonseasonal": total_n // 2, "seasonal": total_n - total_n // 2}
    if task_name in {"C1_step_time", "C2_step_mag"}:
        return {"no_step": total_n // 2, "step": total_n - total_n // 2}
    raise ValueError(f"Unsupported task: {task_name}")


def sample_core_subset(dataset_path: str, per_task_n: int, seed: int) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    by_task_bucket: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for ex in read_jsonl(dataset_path):
        task = str(ex.get("task_name") or ex.get("task") or "")
        bucket = str(ex.get("bucket_label") or "")
        by_task_bucket[(task, bucket)].append(ex)

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    tasks = ["A1_trend_dir", "A2_trend_v", "B1_seasonality_present", "B2_seasonality_amp", "C1_step_time", "C2_step_mag"]
    for task_name in tasks:
        targets = bucket_targets(task_name, per_task_n)
        for bucket, target_n in targets.items():
            pool = by_task_bucket.get((task_name, bucket), [])
            if len(pool) < target_n:
                raise RuntimeError(f"Not enough examples for {task_name}/{bucket}: need {target_n}, have {len(pool)}")
            idx = list(range(len(pool)))
            rng.shuffle(idx)
            chosen = [copy.deepcopy(pool[i]) for i in idx[:target_n]]
            selected.extend(chosen)
            for ex in chosen:
                manifest_rows.append(
                    {
                        "id": ex["id"],
                        "task_name": task_name,
                        "bucket_label": bucket,
                        "pid": ex.get("pid"),
                        "tile_id": ex.get("tile_id"),
                    }
                )

    selected.sort(key=lambda ex: str(ex["id"]))
    manifest_df = pd.DataFrame(manifest_rows).sort_values(["task_name", "bucket_label", "id"]).reset_index(drop=True)
    return selected, manifest_df


def apply_drop_mask(ex: Dict[str, Any], drop_indices: List[int]) -> Dict[str, Any]:
    out = copy.deepcopy(ex)
    series = (((out.get("context") or {}).get("series")) or {})
    d_mm = list(series.get("d_mm") or [])
    mask = list(series.get("mask") or [1] * len(d_mm))
    drop_set = set(drop_indices)
    for i in drop_set:
        d_mm[i] = None
        if i < len(mask):
            mask[i] = 0
    series["d_mm"] = d_mm
    series["mask"] = mask
    out["context"]["series"] = series
    return out


def drop_random_missingness(ex: Dict[str, Any], rate: float, seed: int, min_keep: int = 6) -> Dict[str, Any]:
    series = (((ex.get("context") or {}).get("series")) or {})
    idx = valid_indices(list(series.get("d_mm") or []))
    if len(idx) <= min_keep:
        return copy.deepcopy(ex)
    rng = random.Random(seed)
    drop_n = int(round(rate * len(idx)))
    drop_n = max(0, min(drop_n, len(idx) - min_keep))
    drop = rng.sample(idx, drop_n)
    return apply_drop_mask(ex, drop)


def drop_sparse_sampling(ex: Dict[str, Any], stride: int, seed_idx: int, min_keep: int = 6) -> Dict[str, Any]:
    series = (((ex.get("context") or {}).get("series")) or {})
    idx = valid_indices(list(series.get("d_mm") or []))
    if len(idx) <= min_keep:
        return copy.deepcopy(ex)
    offset = (seed_idx - 1) % max(1, stride)
    keep = [orig_i for j, orig_i in enumerate(idx) if (j - offset) % stride == 0]
    if len(keep) < min_keep:
        keep = [idx[int(round(p))] for p in np.linspace(0, len(idx) - 1, min_keep)]
        keep = sorted(set(keep))
    drop = [i for i in idx if i not in set(keep)]
    return apply_drop_mask(ex, drop)


def drop_local_block(ex: Dict[str, Any], window_days: int, seed: int, min_keep: int = 6) -> Dict[str, Any]:
    series = (((ex.get("context") or {}).get("series")) or {})
    t_iso = list(series.get("t_iso") or [])
    d_mm = list(series.get("d_mm") or [])
    idx = valid_indices(d_mm)
    if len(idx) <= min_keep:
        return copy.deepcopy(ex)
    valid_dates = [(i, parse_date_any(t_iso[i])) for i in idx]
    valid_dates = [(i, dt) for i, dt in valid_dates if dt is not None]
    if len(valid_dates) <= min_keep:
        return copy.deepcopy(ex)
    rng = random.Random(seed)
    center_i, center_dt = valid_dates[rng.randrange(len(valid_dates))]
    half_window = window_days / 2.0
    drop: List[int] = []
    for i, dt in valid_dates:
        if abs((dt - center_dt).days) <= half_window:
            drop.append(i)
    if len(idx) - len(drop) < min_keep:
        drop = drop[: max(0, len(idx) - min_keep)]
    return apply_drop_mask(ex, drop)


def condition_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [{"condition_id": "clean", "family": "clean", "severity": 0.0, "seed_idx": 0}]
    for rate in [0.10, 0.20, 0.40]:
        for seed_idx in [1, 2, 3]:
            specs.append(
                {
                    "condition_id": f"missing_p{int(rate * 100):02d}_s{seed_idx}",
                    "family": "missingness",
                    "severity": float(rate),
                    "seed_idx": seed_idx,
                    "rate": float(rate),
                }
            )
    for stride in [2, 3, 5]:
        for seed_idx in [1, 2, 3]:
            specs.append(
                {
                    "condition_id": f"sparse_every{stride}_s{seed_idx}",
                    "family": "sparse_sampling",
                    "severity": float(stride),
                    "seed_idx": seed_idx,
                    "stride": int(stride),
                }
            )
    for days in [30, 60, 90]:
        for seed_idx in [1, 2, 3]:
            specs.append(
                {
                    "condition_id": f"block_{days}d_s{seed_idx}",
                    "family": "block_dropout",
                    "severity": float(days),
                    "seed_idx": seed_idx,
                    "window_days": int(days),
                }
            )
    return specs


def perturb_example(ex: Dict[str, Any], spec: Dict[str, Any], global_seed: int) -> Dict[str, Any]:
    cond = spec["family"]
    if cond == "clean":
        return copy.deepcopy(ex)
    ex_seed = hash((ex["id"], spec["condition_id"], global_seed)) & 0xFFFFFFFF
    if cond == "missingness":
        return drop_random_missingness(ex, rate=float(spec["rate"]), seed=ex_seed)
    if cond == "sparse_sampling":
        return drop_sparse_sampling(ex, stride=int(spec["stride"]), seed_idx=int(spec["seed_idx"]))
    if cond == "block_dropout":
        return drop_local_block(ex, window_days=int(spec["window_days"]), seed=ex_seed)
    raise ValueError(f"Unknown family: {cond}")


def render_prompt_record(ex: Dict[str, Any]) -> Dict[str, Any]:
    prompt = render_prompt_routeA_a3(
        ex,
        global_mode="summary",
        table_k=50,
        last_n=64,
        anchors_n=12,
        step_topk=5,
        step_window_n=16,
    )
    return {
        "id": ex["id"],
        "task_name": ex.get("task_name") or ex.get("task"),
        "prompt": prompt,
    }


def dataset_condition_summary(examples: List[Dict[str, Any]], spec: Dict[str, Any]) -> Dict[str, Any]:
    obs_counts: List[int] = []
    total_counts: List[int] = []
    for ex in examples:
        series = (((ex.get("context") or {}).get("series")) or {})
        d_mm = list(series.get("d_mm") or [])
        obs_counts.append(len(valid_indices(d_mm)))
        total_counts.append(len(d_mm))
    total_obs = float(np.mean(obs_counts)) if obs_counts else 0.0
    total_pts = float(np.mean(total_counts)) if total_counts else 0.0
    return {
        "condition_id": spec["condition_id"],
        "family": spec["family"],
        "severity": spec["severity"],
        "seed_idx": spec["seed_idx"],
        "n_examples": len(examples),
        "mean_observed_points": total_obs,
        "mean_total_points": total_pts,
        "mean_missing_rate": 1.0 - (total_obs / total_pts if total_pts > 0 else 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_task_n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260503)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    conditions_dir = os.path.join(args.out_dir, "conditions")
    ensure_dir(conditions_dir)

    subset, manifest_df = sample_core_subset(args.dataset_jsonl, per_task_n=int(args.per_task_n), seed=int(args.seed))
    clean_dataset_path = os.path.join(args.out_dir, "core_subset_clean.jsonl")
    clean_prompts_path = os.path.join(args.out_dir, "core_subset_clean_routeA_a3_prompts.jsonl")
    write_jsonl(clean_dataset_path, subset)
    write_jsonl(clean_prompts_path, (render_prompt_record(ex) for ex in subset))
    manifest_df.to_csv(os.path.join(args.out_dir, "core_subset_manifest.csv"), index=False)

    condition_rows: List[Dict[str, Any]] = []
    for spec in condition_specs():
        cond_dir = os.path.join(conditions_dir, spec["condition_id"])
        ensure_dir(cond_dir)
        perturbed = [perturb_example(ex, spec, global_seed=int(args.seed)) for ex in subset]
        dataset_path = os.path.join(cond_dir, "dataset.jsonl")
        prompts_path = os.path.join(cond_dir, "prompts_routeA_a3.jsonl")
        write_jsonl(dataset_path, perturbed)
        write_jsonl(prompts_path, (render_prompt_record(ex) for ex in perturbed))
        row = dataset_condition_summary(perturbed, spec)
        row["dataset_jsonl"] = dataset_path
        row["prompts_jsonl"] = prompts_path
        condition_rows.append(row)

    pd.DataFrame(condition_rows).sort_values(["family", "severity", "seed_idx", "condition_id"]).to_csv(
        os.path.join(args.out_dir, "condition_manifest.csv"),
        index=False,
    )
    summary = {
        "dataset_jsonl": args.dataset_jsonl,
        "per_task_n": int(args.per_task_n),
        "seed": int(args.seed),
        "n_examples": len(subset),
        "clean_dataset_jsonl": clean_dataset_path,
        "clean_prompts_jsonl": clean_prompts_path,
        "n_conditions": len(condition_rows),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Wrote robustness subset to:", args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
