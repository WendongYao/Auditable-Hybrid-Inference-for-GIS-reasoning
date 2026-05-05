#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_cmd(cmd: List[str], log_path: str) -> None:
    ensure_dir(os.path.dirname(log_path) or ".")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}; see {log_path}")


def maybe_run(cmd: List[str], log_path: str, target_path: str, force: bool) -> None:
    if (not force) and os.path.exists(target_path):
        print(f"[SKIP] {target_path}")
        return
    print(f"[RUN ] {os.path.basename(target_path)}")
    run_cmd(cmd, log_path)


def maybe_run_with_marker(cmd: List[str], log_path: str, done_marker: str, force: bool) -> None:
    if (not force) and os.path.exists(done_marker):
        print(f"[SKIP] {done_marker}")
        return
    print(f"[RUN ] {os.path.basename(done_marker)}")
    run_cmd(cmd, log_path)
    with open(done_marker, "w", encoding="utf-8") as f:
        f.write("done\n")


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarise_scores(out_dir: str, system_score_dirs: Dict[str, str]) -> None:
    summary_rows = []
    by_task_frames = []

    for system_name, score_dir in system_score_dirs.items():
        summary_path = os.path.join(score_dir, "summary.json")
        by_task_path = os.path.join(score_dir, "by_task.csv")
        if not os.path.exists(summary_path):
            raise RuntimeError(f"Missing score summary: {summary_path}")
        if not os.path.exists(by_task_path):
            raise RuntimeError(f"Missing by_task.csv: {by_task_path}")

        summary = read_json(summary_path)
        summary_rows.append(
            {
                "system": system_name,
                "micro_acc": float(summary["micro_acc"]),
                "macro_acc": float(summary["macro_acc_over_tasks"]),
                "coverage_pred_over_gold": float(summary["coverage_pred_over_gold"]),
                "n_gold": int(summary["n_gold"]),
                "n_pred": int(summary["n_pred"]),
                "n_scored": int(summary["n_scored"]),
            }
        )

        by_task = pd.read_csv(by_task_path).copy()
        by_task["system"] = system_name
        by_task_frames.append(by_task)

    score_summary = pd.DataFrame(summary_rows).sort_values(["micro_acc", "system"], ascending=[False, True])
    score_summary.to_csv(os.path.join(out_dir, "score_summary.csv"), index=False)

    task_acc = pd.concat(by_task_frames, ignore_index=True)
    task_acc = task_acc.rename(columns={"acc": "task_acc"})
    task_acc.to_csv(os.path.join(out_dir, "task_accuracy_summary.csv"), index=False)

    base_map = score_summary.set_index("system")[["micro_acc", "macro_acc"]].to_dict(orient="index")
    for ref_system, out_name in [
        ("baseline_v3", "delta_vs_baseline.csv"),
        ("llm_direct", "delta_vs_llm.csv"),
    ]:
        if ref_system not in base_map:
            continue
        ref = base_map[ref_system]
        rows = []
        for _, row in score_summary.iterrows():
            rows.append(
                {
                    "system": row["system"],
                    "delta_micro_acc": float(row["micro_acc"] - ref["micro_acc"]),
                    "delta_macro_acc": float(row["macro_acc"] - ref["macro_acc"]),
                }
            )
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, out_name), index=False)

    pivot = task_acc.pivot_table(index="task_name", columns="system", values="task_acc")
    for ref_system, out_name in [
        ("baseline_v3", "task_delta_vs_baseline.csv"),
        ("llm_direct", "task_delta_vs_llm.csv"),
    ]:
        if ref_system not in pivot.columns:
            continue
        delta = pivot.subtract(pivot[ref_system], axis=0).reset_index()
        delta.to_csv(os.path.join(out_dir, out_name), index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--baseline_preds", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hf_python", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = sys.executable
    run_llm_hf = os.path.join(repo_root, "scripts", "run_llm_hf_local.py")
    predict_b1 = os.path.join(repo_root, "scripts", "predict_b1_from_series.py")
    predict_b2 = os.path.join(repo_root, "scripts", "predict_b2_from_series.py")
    predict_c1 = os.path.join(repo_root, "scripts", "predict_c1_none_guard_from_series.py")
    predict_c2 = os.path.join(repo_root, "scripts", "predict_c2_from_routeA_a3_prompt.py")
    merge_taskwise = os.path.join(repo_root, "scripts", "merge_preds_taskwise.py")
    merge_taskwise_cond = os.path.join(repo_root, "scripts", "merge_preds_taskwise_conditional.py")
    apply_gate = os.path.join(repo_root, "scripts", "apply_gate_v3_consistency.py")
    score_py = os.path.join(repo_root, "scripts", "score.py")

    ensure_dir(args.out_dir)
    logs_dir = os.path.join(args.out_dir, "logs")
    rules_dir = os.path.join(args.out_dir, "rules")
    stages_dir = os.path.join(args.out_dir, "stages")
    scores_dir = os.path.join(args.out_dir, "scores")
    ensure_dir(logs_dir)
    ensure_dir(rules_dir)
    ensure_dir(stages_dir)
    ensure_dir(scores_dir)

    llm_preds = os.path.join(args.out_dir, "llm_preds.jsonl")
    llm_done = os.path.join(args.out_dir, "llm_done.marker")
    hybrid_rules_preds = os.path.join(args.out_dir, "hybrid_rules_preds.jsonl")
    hybrid_preds = os.path.join(args.out_dir, "hybrid_preds.jsonl")
    b1_rule_preds = os.path.join(rules_dir, "b1_rule_preds.jsonl")
    b2_rule_preds = os.path.join(rules_dir, "b2_rule_preds.jsonl")
    c1_rule_preds = os.path.join(rules_dir, "c1_reth_rule_preds.jsonl")
    c2_rule_preds = os.path.join(rules_dir, "c2_rule_preds.jsonl")
    stage_b1 = os.path.join(stages_dir, "plus_B1.jsonl")
    stage_b1b2 = os.path.join(stages_dir, "plus_B1B2.jsonl")
    stage_b1b2c1 = os.path.join(stages_dir, "plus_B1B2C1.jsonl")

    score_dirs = {
        "baseline_v3": os.path.join(scores_dir, "baseline_v3"),
        "llm_direct": os.path.join(scores_dir, "llm_direct"),
        "hybrid_rules": os.path.join(scores_dir, "hybrid_rules"),
        "full_hybrid": os.path.join(scores_dir, "full_hybrid"),
    }

    maybe_run(
        [
            py,
            score_py,
            "--dataset",
            args.dataset_jsonl,
            "--pred",
            args.baseline_preds,
            "--out_dir",
            score_dirs["baseline_v3"],
            "--mode",
            "all",
        ],
        os.path.join(logs_dir, "00_score_baseline.log"),
        os.path.join(score_dirs["baseline_v3"], "summary.json"),
        force=bool(args.force),
    )

    maybe_run_with_marker(
        [
            args.hf_python,
            run_llm_hf,
            "--in_prompts",
            args.prompts_jsonl,
            "--out",
            llm_preds,
            "--model_id",
            args.model_id,
            "--batch_size",
            str(args.batch_size),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--use_chat_template",
            "--load_in_4bit",
            "--resume",
        ],
        os.path.join(logs_dir, "01_llm.log"),
        llm_done,
        force=bool(args.force),
    )

    maybe_run(
        [
            py,
            predict_b1,
            "--ids_jsonl",
            args.dataset_jsonl,
            "--dataset",
            args.dataset_jsonl,
            "--out",
            b1_rule_preds,
            "--amp_mode",
            "sin",
            "--sin_period_days",
            "365.25",
            "--amp_thr",
            "0.9",
            "--snr_thr",
            "1.2",
            "--noise_mode",
            "rmse",
        ],
        os.path.join(logs_dir, "02_rule_b1.log"),
        b1_rule_preds,
        force=bool(args.force),
    )

    maybe_run(
        [
            py,
            merge_taskwise,
            "--pred_main",
            llm_preds,
            "--pred_alt",
            b1_rule_preds,
            "--use_alt_tasks",
            "B1_seasonality_present",
            "--out",
            stage_b1,
        ],
        os.path.join(logs_dir, "03_merge_b1.log"),
        stage_b1,
        force=bool(args.force),
    )

    maybe_run(
        [
            py,
            predict_b2,
            "--ids_jsonl",
            args.dataset_jsonl,
            "--dataset",
            args.dataset_jsonl,
            "--out",
            b2_rule_preds,
            "--amp_thr",
            "1.0",
            "--snr_thr",
            "1.2",
            "--amp_mode",
            "sin",
            "--sin_period_days",
            "365.25",
            "--noise_mode",
            "rmse",
            "--round_to",
            "0.25",
            "--dump_debug",
        ],
        os.path.join(logs_dir, "04_rule_b2.log"),
        b2_rule_preds,
        force=bool(args.force),
    )

    maybe_run(
        [
            py,
            merge_taskwise,
            "--pred_main",
            stage_b1,
            "--pred_alt",
            b2_rule_preds,
            "--use_alt_tasks",
            "B2_seasonality_amp",
            "--out",
            stage_b1b2,
        ],
        os.path.join(logs_dir, "05_merge_b2.log"),
        stage_b1b2,
        force=bool(args.force),
    )

    maybe_run(
        [
            py,
            predict_c1,
            "--ids_jsonl",
            args.dataset_jsonl,
            "--dataset",
            args.dataset_jsonl,
            "--out",
            c1_rule_preds,
            "--none_mag_max",
            "1.2",
            "--none_snr_max",
            "2.5",
            "--none_pf_max",
            "0.75",
            "--none_edge_max",
            "2.0",
            "--none_score_max",
            "1.5",
            "--dump_debug",
        ],
        os.path.join(logs_dir, "06_rule_c1.log"),
        c1_rule_preds,
        force=bool(args.force),
    )

    maybe_run(
        [
            py,
            merge_taskwise_cond,
            "--pred_main",
            stage_b1b2,
            "--pred_alt",
            c1_rule_preds,
            "--use_alt_tasks",
            "C1_step_time",
            "--out",
            stage_b1b2c1,
            "--alt_flag_field",
            "use_alt",
        ],
        os.path.join(logs_dir, "07_merge_c1.log"),
        stage_b1b2c1,
        force=bool(args.force),
    )

    maybe_run(
        [
            py,
            predict_c2,
            "--ids_jsonl",
            args.dataset_jsonl,
            "--routeA_prompts_jsonl",
            args.prompts_jsonl,
            "--out",
            c2_rule_preds,
            "--use_abs_delta",
            "--mag_thr",
            "2.7",
            "--snr_thr",
            "2.2",
            "--persist_frac_thr",
            "0.8",
            "--min_post_points",
            "5",
            "--support_gap_max",
            "0.8",
            "--persist_frac_soft_min",
            "0.55",
            "--pf_override_snr_margin",
            "0.5",
            "--pf_override_mag_margin",
            "0.5",
            "--dump_debug",
        ],
        os.path.join(logs_dir, "08_rule_c2.log"),
        c2_rule_preds,
        force=bool(args.force),
    )

    maybe_run(
        [
            py,
            merge_taskwise,
            "--pred_main",
            stage_b1b2c1,
            "--pred_alt",
            c2_rule_preds,
            "--use_alt_tasks",
            "C2_step_mag",
            "--out",
            hybrid_rules_preds,
        ],
        os.path.join(logs_dir, "09_merge_c2.log"),
        hybrid_rules_preds,
        force=bool(args.force),
    )

    maybe_run(
        [
            py,
            apply_gate,
            "--in_pred",
            hybrid_rules_preds,
            "--out_pred",
            hybrid_preds,
            "--gate_b1b2",
            "--gate_c1c2",
        ],
        os.path.join(logs_dir, "10_gate_v3.log"),
        hybrid_preds,
        force=bool(args.force),
    )

    maybe_run(
        [
            py,
            score_py,
            "--dataset",
            args.dataset_jsonl,
            "--pred",
            llm_preds,
            "--out_dir",
            score_dirs["llm_direct"],
            "--mode",
            "all",
        ],
        os.path.join(logs_dir, "11_score_llm.log"),
        os.path.join(score_dirs["llm_direct"], "summary.json"),
        force=bool(args.force),
    )

    maybe_run(
        [
            py,
            score_py,
            "--dataset",
            args.dataset_jsonl,
            "--pred",
            hybrid_rules_preds,
            "--out_dir",
            score_dirs["hybrid_rules"],
            "--mode",
            "all",
        ],
        os.path.join(logs_dir, "12_score_hybrid_rules.log"),
        os.path.join(score_dirs["hybrid_rules"], "summary.json"),
        force=bool(args.force),
    )

    maybe_run(
        [
            py,
            score_py,
            "--dataset",
            args.dataset_jsonl,
            "--pred",
            hybrid_preds,
            "--out_dir",
            score_dirs["full_hybrid"],
            "--mode",
            "all",
        ],
        os.path.join(logs_dir, "13_score_hybrid_full.log"),
        os.path.join(score_dirs["full_hybrid"], "summary.json"),
        force=bool(args.force),
    )

    summarise_scores(args.out_dir, score_dirs)
    print("Completed run in:", args.out_dir)


if __name__ == "__main__":
    main()
