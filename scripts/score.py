# scripts/score.py
import argparse, json, os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def norm_str(x: Any) -> str:
    return str(x).strip().lower()


def try_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and np.isfinite(x):
        return float(x)
    s = str(x).strip()
    if s.lower() in {"none", "null", "nan", ""}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_yyyymmdd(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    ss = str(s).strip()
    if ss.lower() in {"none", "null", ""}:
        return None
    try:
        if len(ss) == 8 and ss.isdigit():
            return datetime.strptime(ss, "%Y%m%d")
        if len(ss) == 10 and ss[4] == "-" and ss[7] == "-":
            return datetime.strptime(ss, "%Y-%m-%d")
    except Exception:
        return None
    return None


def median_cadence_days(ex: Dict[str, Any], fallback: float = 6.0) -> float:
    t_iso = (ex.get("context", {}) or {}).get("series", {}).get("t_iso", [])
    if not t_iso or len(t_iso) < 2:
        return fallback
    dts = []
    for a, b in zip(t_iso[:-1], t_iso[1:]):
        da = parse_yyyymmdd(a)
        db = parse_yyyymmdd(b)
        if da and db:
            dts.append((db - da).days)
    if not dts:
        return fallback
    return float(np.median(dts))


def score_one(gold_ex: Dict[str, Any], pred_ex: Dict[str, Any]) -> Tuple[bool, str]:
    metric = (gold_ex.get("eval") or {}).get("metric")
    tol = (gold_ex.get("eval") or {}).get("tolerance") or {}
    gold_ans = gold_ex.get("answer")
    pred_ans = pred_ex.get("answer_parsed", None)

    parse_err = pred_ex.get("parse_error")
    # 如果有 parse_error 且 answer 为空，直接判错（但 reason 记录清楚）
    if parse_err is not None and pred_ans is None:
        return False, f"parse:{parse_err}"

    if metric == "class_exact":
        ok = norm_str(pred_ans) == norm_str(gold_ans)
        return ok, "ok" if ok else "class_mismatch"

    if metric == "num_within_tol":
        g = try_float(gold_ex.get("answer_numeric", gold_ans))
        p = try_float(pred_ans)
        if g is None or p is None:
            return False, "num_missing"
        abs_tol = float(tol.get("value", 0.5))
        ok = abs(p - g) <= abs_tol
        return ok, "ok" if ok else f"num_outside_tol(abs_tol={abs_tol})"

    if metric == "num_or_none_within_tol":
        gold_none = (gold_ex.get("answer_numeric") is None) or (norm_str(gold_ex.get("answer")) == "none")
        pred_none = (pred_ans is None) or (norm_str(pred_ans) == "none")
        if gold_none:
            ok = pred_none
            return ok, "ok" if ok else "expected_none"
        g = try_float(gold_ex.get("answer_numeric", gold_ans))
        p = try_float(pred_ans)
        if g is None or p is None:
            return False, "num_missing"
        abs_tol = float(tol.get("value", 0.5))
        ok = abs(p - g) <= abs_tol
        return ok, "ok" if ok else f"num_outside_tol(abs_tol={abs_tol})"

    if metric == "time_or_none_within_k":
        gold_none = norm_str(gold_ans) == "none"
        pred_none = (pred_ans is None) or (norm_str(pred_ans) == "none")
        if gold_none:
            ok = pred_none
            return ok, "ok" if ok else "expected_none"

        gd = parse_yyyymmdd(gold_ans)
        pd_ = parse_yyyymmdd(pred_ans)
        if (gd is None) or (pd_ is None):
            return False, "time_parse_fail"

        k = int(tol.get("value", 1))
        cadence = median_cadence_days(gold_ex, fallback=6.0)
        day_tol = max(1.0, cadence) * float(k)
        ok = abs((pd_ - gd).days) <= day_tol
        return ok, "ok" if ok else f"time_outside_tol(days_tol={day_tol:.1f})"

    return False, f"unknown_metric:{metric}"


def _print_file_header(title: str):
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def _print_text_file(path: str, title: str):
    _print_file_header(title)
    try:
        with open(path, "r", encoding="utf-8") as f:
            print(f.read().rstrip("\n"))
    except Exception as e:
        print(f"[WARN] failed to read {path}: {e}")


def _print_df(df: pd.DataFrame, title: str, max_rows: int = 500):
    _print_file_header(title)
    if df is None or len(df) == 0:
        print("(empty)")
        return
    with pd.option_context(
        "display.max_rows", max_rows,
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 200,
    ):
        print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dev_balanced/test_balanced JSONL with gold labels")
    ap.add_argument("--pred", required=True, help="Predictions JSONL from run_llm.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--mode",
        choices=["all", "intersection"],
        default="intersection",
        help="all=score every gold example (missing_pred counted wrong); intersection=score only ids with predictions.",
    )
    # NEW: optional knobs for printing
    ap.add_argument("--print_outputs", action="store_true", help="Print by_task/by_task_bucket/summary to stdout.")
    ap.add_argument("--print_max_rows", type=int, default=500, help="Max rows to print for each CSV.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gold = {ex["id"]: ex for ex in read_jsonl(args.dataset)}

    # 允许 pred 重复 id：保留最后一次（便于 retry 覆盖）
    pred = {}
    for ex in read_jsonl(args.pred):
        pred[ex["id"]] = ex

    gold_ids = set(gold.keys())
    pred_ids = set(pred.keys())
    inter_ids = sorted(gold_ids & pred_ids)

    coverage = len(inter_ids) / max(1, len(gold_ids))

    if args.mode == "intersection":
        score_ids = inter_ids
    else:
        score_ids = sorted(gold_ids)

    rows = []
    for ex_id in score_ids:
        g = gold[ex_id]
        p = pred.get(ex_id, {"id": ex_id, "answer_parsed": None, "parse_error": "missing_pred"})
        ok, reason = score_one(g, p)
        rows.append(
            {
                "id": ex_id,
                "task_name": g.get("task_name", g.get("task")),
                "bucket_label": g.get("bucket_label"),
                "metric": (g.get("eval") or {}).get("metric"),
                "gold_answer": g.get("answer"),
                "pred_answer": p.get("answer_parsed"),
                "parse_error": p.get("parse_error"),
                "ok": int(ok),
                "reason": reason,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "per_example.csv"), index=False)

    by_task = (
        df.groupby(["task_name"])
        .agg(
            n=("ok", "size"),
            acc=("ok", "mean"),
            format_fail=("parse_error", lambda s: float(np.mean(pd.notna(s)))),
        )
        .reset_index()
    )
    by_task_path = os.path.join(args.out_dir, "by_task.csv")
    by_task.to_csv(by_task_path, index=False)

    by_bucket = (
        df.groupby(["task_name", "bucket_label"])
        .agg(
            n=("ok", "size"),
            acc=("ok", "mean"),
        )
        .reset_index()
    )
    by_bucket_path = os.path.join(args.out_dir, "by_task_bucket.csv")
    by_bucket.to_csv(by_bucket_path, index=False)

    macro_acc = float(by_task["acc"].mean()) if len(by_task) else 0.0
    micro_acc = float(df["ok"].mean()) if len(df) else 0.0

    summary = {
        "mode": args.mode,
        "coverage_pred_over_gold": coverage,
        "micro_acc": micro_acc,
        "macro_acc_over_tasks": macro_acc,
        "n_gold": int(len(gold_ids)),
        "n_pred": int(len(pred_ids)),
        "n_scored": int(len(df)),
        "n_tasks": int(df["task_name"].nunique()) if len(df) else 0,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved:", args.out_dir)
    print("Summary:", summary)

    # NEW: print generated outputs
    if args.print_outputs:
        _print_df(by_task, f"by_task.csv (path={by_task_path})", max_rows=int(args.print_max_rows))
        _print_df(by_bucket, f"by_task_bucket.csv (path={by_bucket_path})", max_rows=int(args.print_max_rows))
        _print_text_file(summary_path, f"summary.json (path={summary_path})")


if __name__ == "__main__":
    main()
