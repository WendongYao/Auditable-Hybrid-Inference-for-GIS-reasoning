# scripts/render_prompt.py
import argparse, json, os, random
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def allowed_for_task(task: str) -> str:
    if task == "A1_trend_dir":
        return 'One of: "subsiding", "uplifting", "stable".'
    if task == "A2_trend_v":
        return "A JSON number (mm/year)."
    if task == "B1_seasonality_present":
        return 'One of: "seasonal", "nonseasonal".'
    if task == "B2_seasonality_amp":
        return 'Either "none" or a JSON number (mm).'
    if task == "C1_step_time":
        return 'Either "none" or an 8-digit date string "yyyymmdd".'
    if task == "C2_step_mag":
        return 'Either "none" or a JSON number (mm).'
    return "Return JSON only with key 'answer'."


def format_table(t_iso, d_mm, max_rows=50) -> str:
    rows = ["date,disp_mm"]
    for t, d in zip(t_iso[:max_rows], d_mm[:max_rows]):
        if d is None or (isinstance(d, float) and np.isnan(d)):
            rows.append(f"{t},NA")
        else:
            rows.append(f"{t},{float(d):.3f}")
    return "\n".join(rows)


def parse_date_any(s: str) -> Optional[datetime]:
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    # last resort: fromisoformat for "YYYY-MM-DD"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _valid_indices(d_mm: List) -> List[int]:
    idx = []
    for i, v in enumerate(d_mm):
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            idx.append(i)
    return idx


def _pick_indices_linspace(valid_idx: List[int], n: int) -> List[int]:
    if n <= 0 or len(valid_idx) == 0:
        return []
    if len(valid_idx) <= n:
        return valid_idx[:]
    pos = np.linspace(0, len(valid_idx) - 1, n)
    picked = sorted({valid_idx[int(round(p))] for p in pos})
    return picked[:n]


def _pick_indices_quantile(valid_idx: List[int], n: int) -> List[int]:
    if n <= 0 or len(valid_idx) == 0:
        return []
    if len(valid_idx) <= n:
        return valid_idx[:]
    qs = np.linspace(0, 1, n)
    picked = []
    for q in qs:
        j = int(round(q * (len(valid_idx) - 1)))
        picked.append(valid_idx[j])
    return sorted(set(picked))[:n]


def _pick_indices_step_window(t_iso: List[str], d_mm: List, n: int) -> List[int]:
    # find max abs jump between consecutive valid points, then take a window around it
    valid = _valid_indices(d_mm)
    if n <= 0 or len(valid) < 3:
        return _pick_indices_linspace(valid, n)

    # compute diffs on consecutive original indices
    best_i = None
    best_jump = -1.0
    for i in range(len(d_mm) - 1):
        if i not in set(valid) or (i + 1) not in set(valid):
            continue
        a = float(d_mm[i]); b = float(d_mm[i + 1])
        jump = abs(b - a)
        if jump > best_jump:
            best_jump = jump
            best_i = i

    if best_i is None:
        return _pick_indices_linspace(valid, n)

    # center window around boundary i|i+1
    half = max(1, n // 2)
    start = max(0, best_i - half)
    end = min(len(d_mm), best_i + 1 + half)
    window = [i for i in range(start, end) if i in set(valid)]

    # if still not enough, pad with quantiles
    if len(window) < n:
        pad = _pick_indices_quantile(valid, n - len(window))
        window = sorted(set(window + pad))
    return window[:n]


def _pick_indices_month_anchors(t_iso: List[str], d_mm: List, n: int) -> List[int]:
    # pick one representative per month-of-year (median index), then subsample to n
    valid = _valid_indices(d_mm)
    if n <= 0 or len(valid) == 0:
        return []

    buckets = {m: [] for m in range(1, 13)}
    for i in valid:
        dt = parse_date_any(t_iso[i])
        if not dt:
            continue
        buckets[dt.month].append(i)

    month_reps = []
    for m in range(1, 13):
        lst = buckets[m]
        if not lst:
            continue
        lst_sorted = sorted(lst)
        month_reps.append(lst_sorted[len(lst_sorted) // 2])

    month_reps = sorted(set(month_reps))
    if len(month_reps) == 0:
        return _pick_indices_linspace(valid, n)

    # if we have more reps than needed, subsample months evenly
    if len(month_reps) > n:
        pos = np.linspace(0, len(month_reps) - 1, n)
        picked = sorted({month_reps[int(round(p))] for p in pos})
        return picked[:n]

    # if fewer, pad with linspace
    if len(month_reps) < n:
        pad = _pick_indices_linspace(valid, n - len(month_reps))
        month_reps = sorted(set(month_reps + pad))
    return month_reps[:n]


def pick_anchor_indices(task: str, t_iso: List[str], d_mm: List, n: int, strategy: str) -> List[int]:
    valid = _valid_indices(d_mm)
    if n <= 0:
        return []

    if strategy == "auto":
        if task in {"B1_seasonality_present", "B2_seasonality_amp"}:
            return _pick_indices_month_anchors(t_iso, d_mm, n)
        if task in {"C1_step_time", "C2_step_mag"}:
            return _pick_indices_step_window(t_iso, d_mm, n)
        # trend tasks
        return _pick_indices_quantile(valid, n)

    if strategy == "linspace":
        return _pick_indices_linspace(valid, n)
    if strategy == "quantile":
        return _pick_indices_quantile(valid, n)
    if strategy == "step_window":
        return _pick_indices_step_window(t_iso, d_mm, n)
    if strategy == "month":
        return _pick_indices_month_anchors(t_iso, d_mm, n)

    raise ValueError("anchors_strategy must be one of: auto/linspace/quantile/step_window/month")


def build_summary_block(t_iso: List[str], d_mm: List) -> str:
    d_arr = np.array([np.nan if v is None else float(v) for v in d_mm], dtype=np.float64)
    finite = np.isfinite(d_arr)
    coverage = float(finite.mean()) if len(d_arr) else 0.0
    return (
        "SUMMARY\n"
        f"- start_date: {t_iso[0]}\n"
        f"- end_date:   {t_iso[-1]}\n"
        f"- num_points: {len(d_arr)}\n"
        f"- coverage:   {coverage:.3f}\n"
        f"- min_mm: {np.nanmin(d_arr):.3f}\n"
        f"- max_mm: {np.nanmax(d_arr):.3f}\n"
        f"- mean_mm:{np.nanmean(d_arr):.3f}\n"
        f"- std_mm: {np.nanstd(d_arr):.3f}\n"
    )


def build_anchors_block(task: str, t_iso: List[str], d_mm: List, anchors_n: int, anchors_strategy: str) -> str:
    idxs = pick_anchor_indices(task, t_iso, d_mm, anchors_n, anchors_strategy)
    if len(idxs) == 0:
        return ""
    t_view = [t_iso[i] for i in idxs]
    d_view = [d_mm[i] for i in idxs]
    return (
        f"ANCHORS (n={len(idxs)}, strategy={anchors_strategy})\n"
        + format_table(t_view, d_view, max_rows=len(idxs))
    )


def render_prompt(ex, mode="summary", table_k=50, last_n=32, anchors_n=0, anchors_strategy="auto"):
    task = ex.get("task_name", ex.get("task", "unknown"))
    q = ex.get("question", "")

    series = (ex.get("context", {}) or {}).get("series", None)
    if series is None:
        raise RuntimeError("Example has no context.series.")

    t_iso = series.get("t_iso", [])
    d_mm  = series.get("d_mm", [])
    if not t_iso or not d_mm:
        raise RuntimeError("context.series missing t_iso/d_mm")

    # main observation view
    if mode == "last_n":
        t_view = t_iso[-last_n:]
        d_view = d_mm[-last_n:]
        obs_block = "LAST_N_POINTS\n" + format_table(t_view, d_view, max_rows=last_n)
    elif mode == "table_k":
        obs_block = "TABLE_K_POINTS\n" + format_table(t_iso, d_mm, max_rows=table_k)
    elif mode == "summary":
        obs_block = build_summary_block(t_iso, d_mm)
    else:
        raise ValueError("mode must be one of: summary/table_k/last_n")

    # STRICT APPEND: anchors are always appended after the main block
    anchors_block = ""
    if anchors_n and anchors_n > 0:
        anchors_block = "\n\n" + build_anchors_block(task, t_iso, d_mm, anchors_n, anchors_strategy)

    allowed = allowed_for_task(task)
    instruction = (
        "You are given an InSAR displacement time series (mm) at regular acquisition epochs.\n"
        "Answer the question using only the provided information.\n"
        "Return JSON only with the key 'answer'.\n"
        f"Allowed answer format for this task: {allowed}\n"
        "Do not add extra keys. Do not add explanation.\n"
    )

    return f"{instruction}\nTASK: {task}\nQUESTION: {q}\n\n{obs_block}{anchors_block}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["summary","table_k","last_n"], default="summary")
    ap.add_argument("--table_k", type=int, default=50)
    ap.add_argument("--last_n", type=int, default=32)

    # anchors
    ap.add_argument("--anchors_n", type=int, default=0)
    ap.add_argument("--anchors_strategy", choices=["auto","linspace","quantile","step_window","month"], default="auto")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    data = list(read_jsonl(args.in_jsonl))
    rng.shuffle(data)
    subset = data[:args.n]

    out_path = os.path.join(args.out_dir, f"prompts_{args.mode}_{args.n}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in subset:
            p = render_prompt(
                ex,
                mode=args.mode,
                table_k=args.table_k,
                last_n=args.last_n,
                anchors_n=args.anchors_n,
                anchors_strategy=args.anchors_strategy,
            )
            rec = {"id": ex["id"], "task_name": ex.get("task_name"), "prompt": p}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
