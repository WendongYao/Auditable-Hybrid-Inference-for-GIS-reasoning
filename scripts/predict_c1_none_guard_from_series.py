#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Conditional rule predictor for EGMS-QA task: C1_step_time.

This script is meant to *only* overwrite the model's C1 prediction when the
time series provides weak evidence for a step (i.e., likely no-step). It emits
one record per id with an extra boolean flag `use_alt`:

- use_alt = True  -> downstream merge overrides C1 answer with 'none'
- use_alt = False -> downstream merge keeps main model answer

Output schema (per line):
  {
    "id": ...,
    "task_name": "C1_step_time",
    "model": "c1_none_guard_from_series_v1",
    "answer_parsed": "none",
    "answer_numeric": null,
    "parse_error": false,
    "use_alt": true|false,
    "debug": {...}   # optional when --dump_debug
  }

PowerShell example:
  python scripts/predict_c1_none_guard_from_series.py `
    --ids_jsonl $IDS `
    --dataset $DATA `
    --out $P_C1G `
    --none_mag_max 1.4 `
    --none_snr_max 1.2 `
    --none_pf_max 0.65 `
    --none_edge_max 1.2 `
    --none_score_max 0.85 `
    --dump_debug
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TASK_NAME = "C1_step_time"


# -----------------------------
# IO helpers
# -----------------------------
def read_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: str, records: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_id_list(ids_jsonl: str) -> List[str]:
    ids: List[str] = []
    with open(ids_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, str):
                ids.append(obj)
            elif isinstance(obj, dict) and "id" in obj:
                ids.append(str(obj["id"]))
            else:
                for k in ("qid", "sample_id", "uid"):
                    if isinstance(obj, dict) and k in obj:
                        ids.append(str(obj[k]))
                        break
    return ids


# -----------------------------
# Series helpers
# -----------------------------
def _parse_iso_date_to_ordinal_days(iso: str) -> Optional[int]:
    if not iso:
        return None
    s = iso.strip()
    if len(s) >= 10:
        s = s[:10]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.toordinal()
        except ValueError:
            continue
    return None


def extract_series_xy(row: dict) -> Tuple[np.ndarray, np.ndarray]:
    ctx = row.get("context", {}) or {}
    series = (ctx.get("series", {}) or {})
    t_iso = series.get("t_iso", []) or []
    y = series.get("d_mm", []) or []
    mask = series.get("mask", None)

    n = min(len(t_iso), len(y))
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    t_ord: List[int] = []
    yy: List[float] = []
    for i in range(n):
        if mask is not None:
            try:
                if int(mask[i]) == 0:
                    continue
            except Exception:
                pass
        td = _parse_iso_date_to_ordinal_days(str(t_iso[i]))
        if td is None:
            continue
        try:
            val = float(y[i])
        except Exception:
            continue
        if not math.isfinite(val):
            continue
        t_ord.append(td)
        yy.append(val)

    if not t_ord:
        return np.array([], dtype=float), np.array([], dtype=float)

    t0 = min(t_ord)
    t_days = np.array([d - t0 for d in t_ord], dtype=float)
    y_mm = np.array(yy, dtype=float)
    return t_days, y_mm


def robust_mad_sigma(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = mad / 0.6744897501960817 if mad > 0 else 0.0
    return float(sigma)


# -----------------------------
# Step scan
# -----------------------------
@dataclass
class StepCandidate:
    k: int
    t_split_days: float
    mag: float
    abs_mag: float
    noise: float
    snr: float
    pf: float
    edge: float
    score: float


def _persist_fraction(pre_med: float, post_med: float, post_vals: np.ndarray) -> float:
    """Fraction of post points closer to post median than pre median."""
    if post_vals.size == 0:
        return 0.0
    d_post = np.abs(post_vals - post_med)
    d_pre = np.abs(post_vals - pre_med)
    return float(np.mean(d_post < d_pre))


def scan_best_step(
    t_days: np.ndarray,
    y: np.ndarray,
    min_pre: int,
    min_post: int,
    max_gap_days: float,
) -> Optional[StepCandidate]:
    """Scan split points and return the candidate with max score=snr*pf.

    Constraints:
    - at least min_pre points before, min_post points after
    - ignore splits where time gap across the split is too large (>max_gap_days)
    """
    n = int(y.size)
    if n < (min_pre + min_post):
        return None

    best: Optional[StepCandidate] = None

    for k in range(min_pre, n - min_post + 1):
        # require reasonable continuity across split (avoid large missing gaps)
        gap = t_days[k] - t_days[k - 1]
        if not math.isfinite(gap) or gap > max_gap_days:
            continue

        pre = y[:k]
        post = y[k:]

        pre_med = float(np.median(pre))
        post_med = float(np.median(post))

        mag = float(post_med - pre_med)
        abs_mag = float(abs(mag))

        # noise: robust sigma of first differences (pre+post)
        dy = np.diff(y)
        noise = robust_mad_sigma(dy)
        noise = max(noise, 1e-6)

        snr = float(abs_mag / noise)

        pf = _persist_fraction(pre_med, post_med, post)

        edge = float(abs(y[k] - y[k - 1]))

        score = float(snr * pf)

        cand = StepCandidate(
            k=int(k),
            t_split_days=float((t_days[k] + t_days[k - 1]) / 2.0),
            mag=mag,
            abs_mag=abs_mag,
            noise=noise,
            snr=snr,
            pf=pf,
            edge=edge,
            score=score,
        )

        if best is None:
            best = cand
        else:
            if cand.score > best.score:
                best = cand
            elif cand.score == best.score and cand.abs_mag > best.abs_mag:
                best = cand

    return best


# -----------------------------
# Prediction record
# -----------------------------
def make_pred_record(_id: str, use_alt: bool, debug: Optional[dict]) -> dict:
    rec = {
        "id": _id,
        "task_name": TASK_NAME,
        "model": "c1_none_guard_from_series_v1",
        "answer_parsed": "none",     # always 'none'; downstream uses use_alt to decide override
        "answer_numeric": None,
        "parse_error": False,
        "use_alt": bool(use_alt),
    }
    if debug is not None:
        rec["debug"] = debug
    return rec


# -----------------------------
# Main
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_jsonl", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)

    # none guard thresholds
    ap.add_argument("--none_mag_max", type=float, default=1.4)
    ap.add_argument("--none_snr_max", type=float, default=1.2)
    ap.add_argument("--none_pf_max", type=float, default=0.65)
    ap.add_argument("--none_edge_max", type=float, default=1.2)
    ap.add_argument("--none_score_max", type=float, default=0.85, help="score = snr * pf")

    # scan config
    ap.add_argument("--min_pre", type=int, default=2)
    ap.add_argument("--min_post", type=int, default=5)
    ap.add_argument("--max_gap_days", type=float, default=120)

    ap.add_argument("--dump_debug", action="store_true")
    return ap


def index_task_rows(dataset_jsonl: str, task_name: str) -> Dict[str, dict]:
    data: Dict[str, dict] = {}
    for row in read_jsonl(dataset_jsonl):
        if row.get("task_name") != task_name:
            continue
        _id = str(row.get("id"))
        data[_id] = row
    return data


def main() -> None:
    args = build_argparser().parse_args()

    ids = read_id_list(args.ids_jsonl)
    data = index_task_rows(args.dataset, TASK_NAME)

    out_recs: List[dict] = []
    n_use_alt = 0
    n_abstain = 0
    n_missing = 0

    for _id in ids:
        row = data.get(_id)
        if row is None:
            n_missing += 1
            continue

        t_days, y = extract_series_xy(row)
        best = scan_best_step(
            t_days,
            y,
            min_pre=args.min_pre,
            min_post=args.min_post,
            max_gap_days=args.max_gap_days,
        )

        if best is None:
            use_alt = False
            dbg = {"reason": "no_valid_candidate", "n": int(y.size)} if args.dump_debug else None
            n_abstain += 1
            out_recs.append(make_pred_record(_id, use_alt=use_alt, debug=dbg))
            continue

        # none-guard decision: weak evidence across ALL metrics
        use_alt = (
            (best.abs_mag <= args.none_mag_max)
            and (best.snr <= args.none_snr_max)
            and (best.pf <= args.none_pf_max)
            and (best.edge <= args.none_edge_max)
            and (best.score <= args.none_score_max)
        )

        if use_alt:
            n_use_alt += 1
        else:
            n_abstain += 1

        dbg = None
        if args.dump_debug:
            dbg = {
                "n": int(y.size),
                "best_k": best.k,
                "best_t_split_days": best.t_split_days,
                "best_mag": best.mag,
                "best_abs_mag": best.abs_mag,
                "best_noise": best.noise,
                "best_snr": best.snr,
                "best_pf": best.pf,
                "best_edge": best.edge,
                "best_score": best.score,
                "thr": {
                    "mag": args.none_mag_max,
                    "snr": args.none_snr_max,
                    "pf": args.none_pf_max,
                    "edge": args.none_edge_max,
                    "score": args.none_score_max,
                },
            }

        out_recs.append(make_pred_record(_id, use_alt=use_alt, debug=dbg))

    write_jsonl(args.out, out_recs)

    print(f"[OK] wrote: {args.out}")
    print(f"stats: total={len(ids)} use_alt_none={n_use_alt} abstain={n_abstain} missing={n_missing}")
    print(
        "none_thr: "
        f"mag<={args.none_mag_max} snr<={args.none_snr_max} pf<={args.none_pf_max} "
        f"edge<={args.none_edge_max} score<={args.none_score_max} (score=snr*pf)  "
        f"scan: min_pre={args.min_pre} min_post={args.min_post} max_gap_days={args.max_gap_days:g}"
    )


if __name__ == "__main__":
    main()
