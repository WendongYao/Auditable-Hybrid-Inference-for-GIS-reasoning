#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


PURE_RULES_MODEL = "baseline_v3"
PURE_RULES_VARIANT = "ALL_base"

LLM_VARIANT_CANDIDATES = ["ALL_base", "llmonly"]
RULES_VARIANT_CANDIDATES = ["ABL_plus_B1B2C1C2", "plus_B2C2_no_gate"]
FULL_VARIANT_CANDIDATES = ["ABL_plus_B1B2C1C2_plus_gatev3", "full_B2C2_gatev3", "ABL_plus_B1B2C1C2"]

DEFAULT_MODELS = [
    "gemini25flash",
    "gemini3flash",
    "gemini3pro",
    "deepseekv32a3v1",
    "gpt52",
    "llama4maverick",
]


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_summary(scores_root: str, model: str, variant: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(scores_root, model, variant, "summary.json")
    if not os.path.exists(path):
        return None
    obj = read_json(path)
    obj["path"] = os.path.dirname(path)
    return obj


def load_by_task(scores_root: str, model: str, variant: str) -> pd.DataFrame:
    path = os.path.join(scores_root, model, variant, "by_task.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["task_name", "acc"])
    df = pd.read_csv(path)
    return df[["task_name", "acc"]]


def load_per_example(scores_root: str, model: str, variant: str) -> pd.DataFrame:
    path = os.path.join(scores_root, model, variant, "per_example.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["id", "task_name", "ok"])
    df = pd.read_csv(path)
    return df[["id", "task_name", "ok"]]


def pick_variant(scores_root: str, model: str, candidates: List[str]) -> Optional[str]:
    for variant in candidates:
        if load_summary(scores_root, model, variant) is not None:
            return variant
    return None


def summary_pair(scores_root: str, model: str, variant: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    if variant is None:
        return None, None
    return load_summary(scores_root, model, variant), load_by_task(scores_root, model, variant)


def task_map(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    return {str(r["task_name"]): float(r["acc"]) for _, r in df.iterrows()}


def safe_metric(summary: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not summary:
        return None
    v = summary.get(key)
    return float(v) if v is not None else None


def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a - b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    pure_rules_summary, pure_rules_task_df = summary_pair(args.scores_root, PURE_RULES_MODEL, PURE_RULES_VARIANT)
    pure_rules_task = task_map(pure_rules_task_df)
    pure_rules_per_example = load_per_example(args.scores_root, PURE_RULES_MODEL, PURE_RULES_VARIANT)

    summary_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []

    for model in models:
        llm_variant = pick_variant(args.scores_root, model, LLM_VARIANT_CANDIDATES)
        rules_variant = pick_variant(args.scores_root, model, RULES_VARIANT_CANDIDATES)
        full_variant = pick_variant(args.scores_root, model, FULL_VARIANT_CANDIDATES)

        llm_summary, llm_task_df = summary_pair(args.scores_root, model, llm_variant)
        rules_summary, rules_task_df = summary_pair(args.scores_root, model, rules_variant)
        full_summary, full_task_df = summary_pair(args.scores_root, model, full_variant)

        llm_task = task_map(llm_task_df)
        rules_task = task_map(rules_task_df)
        full_task = task_map(full_task_df)

        summary_rows.append(
            {
                "model": model,
                "llm_variant": llm_variant,
                "rules_variant": rules_variant,
                "full_variant": full_variant,
                "pure_rules_macro": safe_metric(pure_rules_summary, "macro_acc_over_tasks"),
                "pure_rules_micro": safe_metric(pure_rules_summary, "micro_acc"),
                "llm_macro": safe_metric(llm_summary, "macro_acc_over_tasks"),
                "llm_micro": safe_metric(llm_summary, "micro_acc"),
                "rules_macro": safe_metric(rules_summary, "macro_acc_over_tasks"),
                "rules_micro": safe_metric(rules_summary, "micro_acc"),
                "full_macro": safe_metric(full_summary, "macro_acc_over_tasks"),
                "full_micro": safe_metric(full_summary, "micro_acc"),
                "full_minus_llm_macro": delta(safe_metric(full_summary, "macro_acc_over_tasks"), safe_metric(llm_summary, "macro_acc_over_tasks")),
                "full_minus_llm_micro": delta(safe_metric(full_summary, "micro_acc"), safe_metric(llm_summary, "micro_acc")),
                "full_minus_pure_rules_macro": delta(safe_metric(full_summary, "macro_acc_over_tasks"), safe_metric(pure_rules_summary, "macro_acc_over_tasks")),
                "full_minus_pure_rules_micro": delta(safe_metric(full_summary, "micro_acc"), safe_metric(pure_rules_summary, "micro_acc")),
                "rules_minus_pure_rules_macro": delta(safe_metric(rules_summary, "macro_acc_over_tasks"), safe_metric(pure_rules_summary, "macro_acc_over_tasks")),
                "rules_minus_pure_rules_micro": delta(safe_metric(rules_summary, "micro_acc"), safe_metric(pure_rules_summary, "micro_acc")),
            }
        )

        all_tasks = sorted(set(pure_rules_task) | set(llm_task) | set(rules_task) | set(full_task))
        for task_name in all_tasks:
            task_rows.append(
                {
                    "model": model,
                    "task_name": task_name,
                    "pure_rules_acc": pure_rules_task.get(task_name),
                    "llm_acc": llm_task.get(task_name),
                    "rules_acc": rules_task.get(task_name),
                    "full_acc": full_task.get(task_name),
                    "full_minus_llm": delta(full_task.get(task_name), llm_task.get(task_name)),
                    "full_minus_pure_rules": delta(full_task.get(task_name), pure_rules_task.get(task_name)),
                    "rules_minus_pure_rules": delta(rules_task.get(task_name), pure_rules_task.get(task_name)),
                }
            )

        if full_variant is not None and not pure_rules_per_example.empty:
            full_per_example = load_per_example(args.scores_root, model, full_variant)
            if not full_per_example.empty:
                joined = pure_rules_per_example.merge(
                    full_per_example,
                    on=["id", "task_name"],
                    suffixes=("_pure_rules", "_full"),
                )
                for task_name, sub in joined.groupby("task_name", sort=True):
                    both_correct = int(((sub["ok_pure_rules"] == 1) & (sub["ok_full"] == 1)).sum())
                    full_only = int(((sub["ok_pure_rules"] == 0) & (sub["ok_full"] == 1)).sum())
                    rules_only = int(((sub["ok_pure_rules"] == 1) & (sub["ok_full"] == 0)).sum())
                    both_wrong = int(((sub["ok_pure_rules"] == 0) & (sub["ok_full"] == 0)).sum())
                    overlap_rows.append(
                        {
                            "model": model,
                            "task_name": task_name,
                            "n": int(len(sub)),
                            "both_correct": both_correct,
                            "full_only_correct": full_only,
                            "pure_rules_only_correct": rules_only,
                            "both_wrong": both_wrong,
                            "net_full_gain": int(full_only - rules_only),
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    task_df = pd.DataFrame(task_rows)
    overlap_df = pd.DataFrame(overlap_rows)

    summary_df.to_csv(os.path.join(args.out_dir, "hybrid_vs_rules_summary.csv"), index=False)
    task_df.to_csv(os.path.join(args.out_dir, "hybrid_vs_rules_task_deltas.csv"), index=False)
    overlap_df.to_csv(os.path.join(args.out_dir, "hybrid_vs_rules_overlap.csv"), index=False)

    avg_summary = {
        "models": models,
        "avg_full_minus_llm_macro": float(summary_df["full_minus_llm_macro"].dropna().mean()) if not summary_df.empty else None,
        "avg_full_minus_llm_micro": float(summary_df["full_minus_llm_micro"].dropna().mean()) if not summary_df.empty else None,
        "avg_full_minus_pure_rules_macro": float(summary_df["full_minus_pure_rules_macro"].dropna().mean()) if not summary_df.empty else None,
        "avg_full_minus_pure_rules_micro": float(summary_df["full_minus_pure_rules_micro"].dropna().mean()) if not summary_df.empty else None,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(avg_summary, f, ensure_ascii=False, indent=2)

    print("Saved comparison to:", args.out_dir)
    print("Summary:", json.dumps(avg_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
