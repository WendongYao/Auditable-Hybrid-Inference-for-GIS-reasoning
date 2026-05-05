# scripts/render_prompt_routeA_a3.py
import argparse, json, os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DATE_YYYYMMDD = "%Y%m%d"


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_date_any(s: str) -> Optional[datetime]:
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def yyyymmdd_from_iso(s: str) -> str:
    dt = parse_date_any(s)
    if not dt:
        # fallback: strip non-digits
        digits = "".join([c for c in str(s) if c.isdigit()])
        return digits[:8] if len(digits) >= 8 else str(s)
    return dt.strftime(DATE_YYYYMMDD)


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


def format_table(t_iso: List[str], d_mm: List[Any], max_rows: int = 64) -> str:
    rows = ["date,disp_mm"]
    for t, d in zip(t_iso[:max_rows], d_mm[:max_rows]):
        if d is None or (isinstance(d, float) and np.isnan(d)):
            rows.append(f"{t},NA")
        else:
            rows.append(f"{t},{float(d):.3f}")
    return "\n".join(rows)


def _finite_idx(d_mm: List[Any]) -> List[int]:
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


def build_summary_block(t_iso: List[str], d_mm: List[Any]) -> str:
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


def build_static_block(ex: Dict[str, Any]) -> str:
    static = (ex.get("context", {}) or {}).get("static") or {}
    rmse = static.get("rmse_mm")
    vstd = static.get("mean_velocity_std")
    sstd = static.get("seasonality_std")
    if rmse is None and vstd is None and sstd is None:
        return ""
    return (
        "STATIC (optional)\n"
        f"- rmse_mm: {rmse}\n"
        f"- mean_velocity_std: {vstd}\n"
        f"- seasonality_std: {sstd}\n"
    )


def _pick_window_around_boundary(valid: List[int], boundary_i: int, n: int) -> List[int]:
    # boundary is between i and i+1 in original indices
    if n <= 0:
        return []
    if len(valid) == 0:
        return []
    # center around i|i+1, take +/- half
    half = max(1, n // 2)
    start = max(0, boundary_i - half)
    end = min(max(valid) + 1, boundary_i + 1 + half + 1)

    valid_set = set(valid)
    window = [i for i in range(start, end) if i in valid_set]
    # pad if needed using quantiles
    if len(window) < n:
        # quantile pad on valid indices
        qs = np.linspace(0, 1, n - len(window))
        pad = []
        for q in qs:
            j = int(round(q * (len(valid) - 1)))
            pad.append(valid[j])
        window = sorted(set(window + pad))
    return window[:n]


def _estimate_step_mag(d_mm: List[Any], valid: List[int], boundary_i: int, k_side: int = 3) -> Optional[float]:
    # estimate abs median(after) - median(before), using up to k_side points each side
    valid_set = set(valid)
    pre = [i for i in range(boundary_i - 50, boundary_i + 1) if i in valid_set]
    post = [i for i in range(boundary_i + 1, boundary_i + 1 + 50) if i in valid_set]
    if len(pre) == 0 or len(post) == 0:
        return None
    pre = pre[-k_side:]
    post = post[:k_side]
    try:
        pre_vals = [float(d_mm[i]) for i in pre]
        post_vals = [float(d_mm[i]) for i in post]
        return float(abs(np.median(post_vals) - np.median(pre_vals)))
    except Exception:
        return None


def build_step_candidates_block(
    t_iso: List[str],
    d_mm: List[Any],
    topk: int = 5,
    window_n: int = 16,
) -> str:
    valid = _finite_idx(d_mm)
    if len(valid) < 2:
        return "STEP_CANDIDATES\n- insufficient_valid_points\n"

    valid_set = set(valid)
    # compute diffs on consecutive original indices where both finite
    candidates: List[Tuple[float, int, float]] = []
    for i in range(len(d_mm) - 1):
        if i not in valid_set or (i + 1) not in valid_set:
            continue
        a = float(d_mm[i])
        b = float(d_mm[i + 1])
        delta = b - a
        candidates.append((abs(delta), i, delta))

    if not candidates:
        return "STEP_CANDIDATES\n- no_consecutive_finite_pairs\n"

    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = candidates[: max(1, int(topk))]

    lines = ["STEP_CANDIDATES (top by |delta| on consecutive epochs)"]
    for rank, (absd, i, delta) in enumerate(candidates, start=1):
        left = t_iso[i]
        right = t_iso[i + 1]
        step_date = yyyymmdd_from_iso(right)  # IMPORTANT: answer tends to be the "new level" epoch
        est = _estimate_step_mag(d_mm, valid, i, k_side=3)
        est_s = "NA" if est is None else f"{est:.3f}"
        lines.append(
            f"- #{rank}: {left} -> {right} | delta_mm={delta:.3f} | step_date_candidate={step_date} | est_step_mag_abs_mm≈{est_s}"
        )

    # also provide a window around the strongest candidate
    best_i = candidates[0][1]
    win_idx = _pick_window_around_boundary(valid, best_i, window_n)
    t_view = [t_iso[j] for j in win_idx]
    d_view = [d_mm[j] for j in win_idx]
    window_block = "WINDOW_AROUND_TOP1_JUMP\n" + format_table(t_view, d_view, max_rows=len(t_view))

    return "\n".join(lines) + "\n\n" + window_block + "\n"


def pick_anchor_indices_auto(task: str, t_iso: List[str], d_mm: List[Any], n: int) -> List[int]:
    valid = _finite_idx(d_mm)
    if n <= 0 or len(valid) == 0:
        return []
    if len(valid) <= n:
        return valid[:]

    if task in {"B1_seasonality_present", "B2_seasonality_amp"}:
        # one rep per month if possible
        buckets = {m: [] for m in range(1, 13)}
        for i in valid:
            dt = parse_date_any(t_iso[i])
            if dt:
                buckets[dt.month].append(i)
        reps = []
        for m in range(1, 13):
            lst = sorted(buckets[m])
            if lst:
                reps.append(lst[len(lst) // 2])
        reps = sorted(set(reps))
        if len(reps) >= n:
            pos = np.linspace(0, len(reps) - 1, n)
            return sorted({reps[int(round(p))] for p in pos})[:n]
        # pad with quantiles
        pos = np.linspace(0, len(valid) - 1, n - len(reps))
        pad = sorted({valid[int(round(p))] for p in pos})
        return sorted(set(reps + pad))[:n]

    if task in {"C1_step_time", "C2_step_mag"}:
        # for step tasks, anchors are handled by step candidates + window; still give some global quantiles
        pos = np.linspace(0, len(valid) - 1, n)
        return sorted({valid[int(round(p))] for p in pos})[:n]

    # trend tasks
    pos = np.linspace(0, len(valid) - 1, n)
    return sorted({valid[int(round(p))] for p in pos})[:n]


def render_prompt_routeA_a3(
    ex: Dict[str, Any],
    global_mode: str = "summary",
    table_k: int = 50,
    last_n: int = 64,
    anchors_n: int = 12,
    step_topk: int = 5,
    step_window_n: int = 16,
) -> str:
    task = ex.get("task_name") or ex.get("task") or "unknown"
    q = ex.get("question", "")

    series = (ex.get("context", {}) or {}).get("series", None)
    if series is None:
        raise RuntimeError("Example has no context.series.")
    t_iso = series.get("t_iso", [])
    d_mm = series.get("d_mm", [])
    if not t_iso or not d_mm:
        raise RuntimeError("context.series missing t_iso/d_mm")

    allowed = allowed_for_task(task)

    # base instruction
    instruction = (
        "You are given an InSAR displacement time series (mm) at regular acquisition epochs.\n"
        "Answer the question using only the provided information.\n"
        "Output MUST be exactly one JSON object and nothing else.\n"
        "Return JSON only with the key 'answer'. Do not add extra keys. Do not add explanation.\n"
        f"Allowed answer format for this task: {allowed}\n"
    )

    # extra A3 guidance for step tasks
    if task == "C1_step_time":
        instruction += (
            "\nSTEP DECISION (do this first, before choosing \"none\" or a date):Definition:\n"
            "- A \"step\" is a PERSISTENT level shift between two consecutive observed epochs (even if the time gap is large).\n"
            "- A large time gap does NOT invalidate a step. If the series jumps and then stays near a new level in the window, it can be a step.\n\n"
            "How to decide (use only what is provided below):1) Look at STEP_CANDIDATES. Start from #1 (largest |delta|).2) Check PERSISTENCE using WINDOW_AROUND_TOP1_JUMP:\n"
            "   - Compare the few points BEFORE the jump vs the few points AFTER the jump.\n"
            "   - A true step should show that after the jump, values mostly remain near the new level (do NOT quickly return toward the old level).3) Check MAGNITUDE vs typical variation:\n"
            "   - If the jump size looks similar to normal point-to-point variation in the window, treat it as NOT a step.\n"
            "   - If the jump is clearly larger than typical variation AND the new level persists, treat it as a step.\n\n"
            "Output rule:\n"
            "- Output \"none\" ONLY IF the top jump candidate does NOT look like a persistent level shift\n"
            "  (e.g., quickly reverts back, or is comparable to typical noise/variation).\n"
            "- If you decide a step exists, output the step_date_candidate (yyyymmdd) of the FIRST epoch on the new level\n"
            "  (the right side of the jump).\n"
        )
    if task == "C2_step_mag":
        instruction += (
        "\nSTEP DECISION (do this first, before choosing \"none\" or a number):Definition:\n"
        "- A \"step\" is a PERSISTENT level shift between two consecutive observed epochs (even if the time gap is large).\n"
        "- A large time gap does NOT invalidate a step.\n\n"
        "How to decide:1) Use STEP_CANDIDATES (#1 first) + WINDOW_AROUND_TOP1_JUMP to check persistence (post-jump points stay near a new level).2) If the jump is comparable to typical variation/noise and not persistent, treat as NO step.\n\n"
        "Output rule:\n"
        "- Output \"none\" ONLY IF the top jump candidate does NOT look persistent in the window.\n"
        "- If you decide a step exists:\n"
        "  - Output a POSITIVE step magnitude (mm).\n"
        "  - Prefer est_step_mag_abs_mm≈... from the top candidate; if it is NA, use abs(delta_mm) from the top candidate.\n"
    )

    # observation block
    if global_mode == "last_n":
        t_view = t_iso[-last_n:]
        d_view = d_mm[-last_n:]
        obs_block = "LAST_N_POINTS\n" + format_table(t_view, d_view, max_rows=last_n)
    elif global_mode == "table_k":
        obs_block = "TABLE_K_POINTS\n" + format_table(t_iso, d_mm, max_rows=table_k)
    elif global_mode == "summary":
        obs_block = build_summary_block(t_iso, d_mm)
    else:
        raise ValueError("global_mode must be one of: summary/table_k/last_n")

    static_block = build_static_block(ex)
    if static_block:
        obs_block = obs_block + "\n" + static_block

    # anchors (generic)
    anchor_idxs = pick_anchor_indices_auto(task, t_iso, d_mm, anchors_n)
    anchors_block = ""
    if anchors_n > 0 and len(anchor_idxs) > 0:
        t_a = [t_iso[i] for i in anchor_idxs]
        d_a = [d_mm[i] for i in anchor_idxs]
        anchors_block = "\n\nANCHORS (auto)\n" + format_table(t_a, d_a, max_rows=len(t_a))

    # A3: step candidates block for C tasks
    step_block = ""
    if task in {"C1_step_time", "C2_step_mag"}:
        step_block = "\n\n" + build_step_candidates_block(
            t_iso=t_iso, d_mm=d_mm, topk=step_topk, window_n=step_window_n
        )

    prompt = (
        f"{instruction}\n"
        f"TASK: {task}\n"
        f"QUESTION: {q}\n\n"
        f"{obs_block}"
        f"{anchors_block}"
        f"{step_block}\n"
    )
    return prompt


def unwrap_if_pool_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    # support a3_pool samples.jsonl format: {"dataset": {...}, "pred": {...}, ...}
    ds = rec.get("dataset")
    if isinstance(ds, dict) and ("context" in ds or "question" in ds):
        return ds
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", required=True, help="Full dataset jsonl (e.g., dev_balanced.jsonl).")
    ap.add_argument("--ids_jsonl", required=True, help="A jsonl file containing the 560 ids (e.g., prompts_summary_560.jsonl).")
    ap.add_argument("--out_jsonl", required=True, help="Output prompts jsonl.")

    ap.add_argument("--mode", choices=["summary", "table_k", "last_n"], default="summary")
    ap.add_argument("--table_k", type=int, default=50)
    ap.add_argument("--last_n", type=int, default=64)

    ap.add_argument("--anchors_n", type=int, default=12)

    # A3 knobs (step tasks)
    ap.add_argument("--step_topk", type=int, default=5)
    ap.add_argument("--step_window_n", type=int, default=16)

    args = ap.parse_args()

    # 1) read ids in order
    id_rows = list(read_jsonl(args.ids_jsonl))
    ids: List[str] = []
    for r in id_rows:
        if "id" not in r:
            raise RuntimeError(f"ids_jsonl row missing 'id': {r}")
        ids.append(r["id"])

    id_set = set(ids)
    if len(ids) != len(id_set):
        raise RuntimeError("ids_jsonl contains duplicate ids; please dedup first.")

    # 2) load only needed examples
    ex_map: Dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(args.dataset_jsonl):
        rec = unwrap_if_pool_row(rec)
        rid = rec.get("id")
        if rid in id_set:
            ex_map[rid] = rec

    missing = [i for i in ids if i not in ex_map]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} ids in dataset_jsonl. First 5: {missing[:5]}")

    # 3) render prompts in the SAME order as ids_jsonl
    out_rows: List[Dict[str, Any]] = []
    for rid in ids:
        ex = ex_map[rid]
        p = render_prompt_routeA_a3(
            ex,
            global_mode=args.mode,
            table_k=args.table_k,
            last_n=args.last_n,
            anchors_n=args.anchors_n,
            step_topk=args.step_topk,
            step_window_n=args.step_window_n,
        )
        out_rows.append({"id": rid, "task_name": ex.get("task_name"), "prompt": p})

    write_jsonl(args.out_jsonl, out_rows)
    print(f"[OK] Wrote: {args.out_jsonl}  (n={len(out_rows)})")


if __name__ == "__main__":
    main()
