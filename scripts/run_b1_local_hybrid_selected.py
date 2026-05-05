#!/usr/bin/env python3
import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition_manifest_csv", required=True)
    ap.add_argument("--condition_ids", required=True, help="Comma-separated list")
    ap.add_argument("--baseline_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--hf_python", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_llm_hf = os.path.join(repo_root, "scripts", "run_llm_hf_local.py")
    predict_b1 = os.path.join(repo_root, "scripts", "predict_b1_from_series.py")
    predict_b2 = os.path.join(repo_root, "scripts", "predict_b2_from_series.py")
    predict_c1 = os.path.join(repo_root, "scripts", "predict_c1_none_guard_from_series.py")
    predict_c2 = os.path.join(repo_root, "scripts", "predict_c2_from_routeA_a3_prompt.py")
    merge_taskwise = os.path.join(repo_root, "scripts", "merge_preds_taskwise.py")
    merge_taskwise_cond = os.path.join(repo_root, "scripts", "merge_preds_taskwise_conditional.py")
    apply_gate = os.path.join(repo_root, "scripts", "apply_gate_v3_consistency.py")
    score_py = os.path.join(repo_root, "scripts", "score.py")
    py = sys.executable

    ensure_dir(args.out_root)
    wanted = [x.strip() for x in args.condition_ids.split(",") if x.strip()]
    manifest = pd.read_csv(args.condition_manifest_csv)
    manifest = manifest[manifest["condition_id"].isin(wanted)].copy()
    if manifest.empty:
        raise RuntimeError("No matching condition_ids found.")
    manifest["_order"] = manifest["condition_id"].apply(lambda x: wanted.index(x))
    manifest = manifest.sort_values("_order")

    for _, row in manifest.iterrows():
        cond = str(row["condition_id"])
        cond_dir = os.path.join(args.out_root, cond)
        logs_dir = os.path.join(cond_dir, "logs")
        rules_dir = os.path.join(cond_dir, "rules")
        stages_dir = os.path.join(cond_dir, "stages")
        ensure_dir(logs_dir)
        ensure_dir(rules_dir)
        ensure_dir(stages_dir)
        llm_preds = os.path.join(cond_dir, "llm_preds.jsonl")
        hybrid_rules_preds = os.path.join(cond_dir, "hybrid_rules_preds.jsonl")
        hybrid_preds = os.path.join(cond_dir, "hybrid_preds.jsonl")
        b1_rule_preds = os.path.join(rules_dir, "b1_rule_preds.jsonl")
        b2_rule_preds = os.path.join(rules_dir, "b2_rule_preds.jsonl")
        c1_rule_preds = os.path.join(rules_dir, "c1_reth_rule_preds.jsonl")
        c2_rule_preds = os.path.join(rules_dir, "c2_rule_preds.jsonl")
        stage_b1 = os.path.join(stages_dir, "plus_B1.jsonl")
        stage_b1b2 = os.path.join(stages_dir, "plus_B1B2.jsonl")
        stage_b1b2c1 = os.path.join(stages_dir, "plus_B1B2C1.jsonl")
        llm_score_dir = os.path.join(cond_dir, "llm_score")
        hybrid_rules_score_dir = os.path.join(cond_dir, "hybrid_rules_score")
        hybrid_score_dir = os.path.join(cond_dir, "hybrid_score")
        baseline_preds = os.path.join(args.baseline_root, cond, "baseline_preds.jsonl")
        dataset_jsonl = str(row["dataset_jsonl"])
        prompts_jsonl = str(row["prompts_jsonl"])

        maybe_run(
            [
                args.hf_python,
                run_llm_hf,
                "--in_prompts",
                prompts_jsonl,
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
            ],
            os.path.join(logs_dir, "01_llm.log"),
            llm_preds,
            force=bool(args.force),
        )

        maybe_run(
            [
                py,
                predict_b1,
                "--ids_jsonl",
                dataset_jsonl,
                "--dataset",
                dataset_jsonl,
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
                dataset_jsonl,
                "--dataset",
                dataset_jsonl,
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
                dataset_jsonl,
                "--dataset",
                dataset_jsonl,
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
                dataset_jsonl,
                "--routeA_prompts_jsonl",
                prompts_jsonl,
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
                dataset_jsonl,
                "--pred",
                llm_preds,
                "--out_dir",
                llm_score_dir,
                "--mode",
                "all",
            ],
            os.path.join(logs_dir, "11_score_llm.log"),
            os.path.join(llm_score_dir, "summary.json"),
            force=bool(args.force),
        )

        maybe_run(
            [
                py,
                score_py,
                "--dataset",
                dataset_jsonl,
                "--pred",
                hybrid_rules_preds,
                "--out_dir",
                hybrid_rules_score_dir,
                "--mode",
                "all",
            ],
            os.path.join(logs_dir, "12_score_hybrid_rules.log"),
            os.path.join(hybrid_rules_score_dir, "summary.json"),
            force=bool(args.force),
        )

        maybe_run(
            [
                py,
                score_py,
                "--dataset",
                dataset_jsonl,
                "--pred",
                hybrid_preds,
                "--out_dir",
                hybrid_score_dir,
                "--mode",
                "all",
            ],
            os.path.join(logs_dir, "13_score_hybrid_full.log"),
            os.path.join(hybrid_score_dir, "summary.json"),
            force=bool(args.force),
        )

    print("Completed selected conditions:", ",".join(wanted))


if __name__ == "__main__":
    main()
