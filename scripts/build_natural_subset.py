#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from render_prompt_routeA_a3 import render_prompt_routeA_a3  # noqa: E402


EXPECTED_TASKS = [
    "A1_trend_dir",
    "A2_trend_v",
    "B1_seasonality_present",
    "B2_seasonality_amp",
    "C1_step_time",
    "C2_step_mag",
]


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


def pid_key(ex: Dict[str, Any]) -> Tuple[str, str]:
    return str(ex.get("tile_id") or ""), str(ex.get("pid") or "")


def largest_remainder_allocation(counts_by_tile: Dict[str, int], target_total: int) -> Dict[str, int]:
    total = int(sum(counts_by_tile.values()))
    if total <= 0:
        raise RuntimeError("No eligible pid groups found for natural subset sampling.")
    if target_total <= 0:
        raise RuntimeError("target_total must be positive.")
    if target_total > total:
        raise RuntimeError(f"Requested {target_total} pid groups, but only {total} are eligible.")

    raw = {tile: (target_total * count / total) for tile, count in counts_by_tile.items()}
    alloc = {tile: int(math.floor(v)) for tile, v in raw.items()}
    remainder = target_total - sum(alloc.values())
    ranked = sorted(raw.items(), key=lambda kv: (-(kv[1] - math.floor(kv[1])), kv[0]))
    for tile, _ in ranked[:remainder]:
        alloc[tile] += 1

    for tile, count in sorted(counts_by_tile.items(), key=lambda kv: kv[0]):
        if alloc[tile] > count:
            overflow = alloc[tile] - count
            alloc[tile] = count
            for other, _ in ranked:
                spare = counts_by_tile[other] - alloc[other]
                if spare <= 0:
                    continue
                take = min(spare, overflow)
                alloc[other] += take
                overflow -= take
                if overflow == 0:
                    break
            if overflow != 0:
                raise RuntimeError("Tile allocation overflow could not be resolved.")

    if sum(alloc.values()) != target_total:
        raise RuntimeError("Tile allocation failed to match target_total.")
    return alloc


def render_prompt_record(ex: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": ex["id"],
        "task_name": ex.get("task_name") or ex.get("task"),
        "prompt": render_prompt_routeA_a3(ex),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--baseline_preds_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_examples", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=20260504)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    target_pid_count = int(round(float(args.target_examples) / len(EXPECTED_TASKS)))

    group_info: Dict[Tuple[str, str], Dict[str, Any]] = {}
    full_task_bucket = Counter()
    full_tile_group_counts = Counter()
    full_tile_example_counts = Counter()

    for ex in read_jsonl(args.dataset_jsonl):
        tile_id, pid = pid_key(ex)
        key = (tile_id, pid)
        task_name = str(ex.get("task_name") or ex.get("task") or "")
        bucket_label = str(ex.get("bucket_label") or "")
        info = group_info.setdefault(
            key,
            {
                "tile_id": tile_id,
                "pid": pid,
                "tasks": set(),
                "n_examples": 0,
            },
        )
        info["tasks"].add(task_name)
        info["n_examples"] += 1
        full_task_bucket[(task_name, bucket_label)] += 1
        full_tile_example_counts[tile_id] += 1

    eligible_by_tile: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    incomplete_rows: List[Dict[str, Any]] = []
    for key, info in group_info.items():
        task_set = set(info["tasks"])
        if task_set == set(EXPECTED_TASKS) and int(info["n_examples"]) == len(EXPECTED_TASKS):
            eligible_by_tile[info["tile_id"]].append(key)
            full_tile_group_counts[info["tile_id"]] += 1
        else:
            incomplete_rows.append(
                {
                    "tile_id": info["tile_id"],
                    "pid": info["pid"],
                    "n_examples": int(info["n_examples"]),
                    "tasks_present": ",".join(sorted(task_set)),
                }
            )

    alloc = largest_remainder_allocation(
        counts_by_tile={tile: len(keys) for tile, keys in eligible_by_tile.items()},
        target_total=target_pid_count,
    )

    rng = random.Random(int(args.seed))
    selected_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    alloc_rows: List[Dict[str, Any]] = []
    for tile_id in sorted(eligible_by_tile.keys()):
        pool = list(eligible_by_tile[tile_id])
        rng.shuffle(pool)
        chosen = sorted(pool[: alloc[tile_id]])
        for key in chosen:
            selected_groups[key] = {"tile_id": key[0], "pid": key[1]}
        alloc_rows.append(
            {
                "tile_id": tile_id,
                "eligible_pid_groups": int(len(pool)),
                "selected_pid_groups": int(len(chosen)),
                "selected_examples_expected": int(len(chosen) * len(EXPECTED_TASKS)),
                "selected_pid_frac": float(len(chosen) / max(len(pool), 1)),
            }
        )

    subset_path = os.path.join(args.out_dir, "natural_subset.jsonl")
    prompts_path = os.path.join(args.out_dir, "natural_subset_routeA_a3_prompts.jsonl")
    baseline_subset_path = os.path.join(args.out_dir, "baseline_preds_subset.jsonl")
    pid_manifest_path = os.path.join(args.out_dir, "selected_pid_manifest.csv")
    alloc_path = os.path.join(args.out_dir, "tile_allocation.csv")
    compare_path = os.path.join(args.out_dir, "task_bucket_compare.csv")
    incomplete_path = os.path.join(args.out_dir, "incomplete_pid_groups.csv")
    summary_path = os.path.join(args.out_dir, "summary.json")

    selected_ids = set()
    subset_task_bucket = Counter()
    subset_tile_example_counts = Counter()
    subset_group_counts = Counter()
    prompt_rows: List[Dict[str, Any]] = []

    with open(subset_path, "w", encoding="utf-8") as f_out:
        for ex in read_jsonl(args.dataset_jsonl):
            key = pid_key(ex)
            if key not in selected_groups:
                continue
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            selected_ids.add(str(ex["id"]))
            subset_group_counts[key] += 1
            task_name = str(ex.get("task_name") or ex.get("task") or "")
            bucket_label = str(ex.get("bucket_label") or "")
            subset_task_bucket[(task_name, bucket_label)] += 1
            subset_tile_example_counts[str(ex.get("tile_id") or "")] += 1
            prompt_rows.append(render_prompt_record(ex))

    write_jsonl(prompts_path, prompt_rows)

    kept = 0
    with open(baseline_subset_path, "w", encoding="utf-8") as f_out:
        for row in read_jsonl(args.baseline_preds_jsonl):
            if str(row.get("id")) not in selected_ids:
                continue
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    pd.DataFrame(sorted(selected_groups.values(), key=lambda r: (r["tile_id"], r["pid"]))).to_csv(pid_manifest_path, index=False)
    pd.DataFrame(alloc_rows).sort_values("tile_id").to_csv(alloc_path, index=False)
    incomplete_df = pd.DataFrame(incomplete_rows)
    if incomplete_df.empty:
        incomplete_df = pd.DataFrame(columns=["tile_id", "pid", "n_examples", "tasks_present"])
    else:
        incomplete_df = incomplete_df.sort_values(["tile_id", "pid"])
    incomplete_df.to_csv(incomplete_path, index=False)

    compare_rows: List[Dict[str, Any]] = []
    keys = sorted(set(full_task_bucket.keys()) | set(subset_task_bucket.keys()))
    for task_name, bucket_label in keys:
        full_n = int(full_task_bucket[(task_name, bucket_label)])
        subset_n = int(subset_task_bucket[(task_name, bucket_label)])
        compare_rows.append(
            {
                "task_name": task_name,
                "bucket_label": bucket_label,
                "full_n": full_n,
                "subset_n": subset_n,
                "full_frac_within_task": float(full_n / max(sum(v for (t, _), v in full_task_bucket.items() if t == task_name), 1)),
                "subset_frac_within_task": float(subset_n / max(sum(v for (t, _), v in subset_task_bucket.items() if t == task_name), 1)),
            }
        )
    pd.DataFrame(compare_rows).sort_values(["task_name", "bucket_label"]).to_csv(compare_path, index=False)

    bad_groups = [f"{tile}:{pid}" for (tile, pid), n in subset_group_counts.items() if int(n) != len(EXPECTED_TASKS)]
    if bad_groups:
        raise RuntimeError(f"Subset contains incomplete pid groups: {bad_groups[:10]}")

    summary = {
        "seed": int(args.seed),
        "requested_examples": int(args.target_examples),
        "target_pid_groups": int(target_pid_count),
        "selected_pid_groups": int(len(selected_groups)),
        "selected_examples": int(len(prompt_rows)),
        "selected_ids": int(len(selected_ids)),
        "baseline_subset_preds": int(kept),
        "n_tiles": int(len(eligible_by_tile)),
        "eligible_pid_groups_total": int(sum(len(v) for v in eligible_by_tile.values())),
        "tile_example_counts_full": {k: int(v) for k, v in sorted(full_tile_example_counts.items())},
        "tile_pid_counts_full": {k: int(v) for k, v in sorted(full_tile_group_counts.items())},
        "tile_example_counts_subset": {k: int(v) for k, v in sorted(subset_tile_example_counts.items())},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Wrote natural subset to:", args.out_dir)


if __name__ == "__main__":
    main()
