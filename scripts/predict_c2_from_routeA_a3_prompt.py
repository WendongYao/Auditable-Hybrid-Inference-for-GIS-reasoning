#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict C2_step_mag from routeA a3 prompt by parsing:
- STEP_CANDIDATES (delta_mm, step_date_candidate, est_step_mag_abs_mm)
- WINDOW_AROUND_TOP1_JUMP for rank#1 (preferred)
- ANCHORS (fallback for non-top1 candidates, or if window missing)

Decision:
- Evaluate persistence using median pre/post split at step date.
- persist_frac = fraction of post points closer to post_med than pre_med.
- Candidate passes if:
    mag_used >= mag_thr
    snr >= snr_thr (snr = mag_used / rmse_mm)
    persist_frac >= persist_frac_thr
      + FIX5: window relax via anchor_shift_abs
      + FIX7 (now gated by args; defaults None): window soft persist / anchors support_gap gate
    n_pre >= 2 and n_post >= min_post_points

Selection (FIX6):
- Compute passers (still allowed to be window or anchors during evaluation)
- But final selection is WINDOW-ONLY:
  * If no window passer exists: return none (reason=no_window_passer)
  * Else: pick best among window passers by:
      persist desc, snr desc, anchor_shift_abs desc, gap_days asc
  * method label:
      - top1_pass_keep if best.rank == 1
      - else tiebreak_over_passers

Extra debug:
- support_gap = |mag_used - anchor_shift_abs|
- pf_override_ok (window only; FIX7, gated by args)
"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -------------------------
# IO
# -------------------------
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    # utf-8-sig handles BOM safely (your earlier crash)
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# Parsing helpers
# -------------------------
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_iso_date(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None


def _parse_yyyymmdd(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y%m%d")
    except Exception:
        return None


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def _extract_block_lines(text: str, header_regex: str) -> Optional[List[str]]:
    """
    Find a block that starts at a header line matching header_regex.
    Returns the remaining lines after that header, not including the header.
    """
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if re.search(header_regex, ln):
            return lines[i + 1 :]
    return None


def _parse_points_after_csv_header(lines: List[str]) -> List[Tuple[str, float]]:
    """
    Parse "date,disp_mm" style lines from a segment of lines.
    Stops when it hits a blank line or a new SECTION-ish line.
    """
    pts: List[Tuple[str, float]] = []
    in_csv = False
    for ln in lines:
        raw = ln.strip()
        if not raw:
            if in_csv:
                break
            continue

        if not in_csv:
            if raw.lower().startswith("date,disp_mm"):
                in_csv = True
            continue

        # stop if we hit another section header
        if re.match(r"^[A-Z0-9_ ]{6,}$", raw) and "," not in raw:
            break

        # parse data line
        if "," in raw:
            a, b = raw.split(",", 1)
            a = a.strip()
            b = b.strip()
            if _DATE_RE.match(a):
                v = _safe_float(b)
                if v is not None:
                    pts.append((a, float(v)))
                continue
    return pts


def parse_rmse_mm(prompt: str) -> Optional[float]:
    # e.g. "- rmse_mm: 0.4"
    m = re.search(r"rmse_mm:\s*([0-9.]+)", prompt)
    return _safe_float(m.group(1)) if m else None


@dataclass
class Candidate:
    rank: int
    left_iso: str
    right_iso: str
    step_yyyymmdd: str
    step_iso: str
    delta_mm: float
    est_step_mag_abs_mm: Optional[float]

    @property
    def gap_days(self) -> Optional[float]:
        dl = _parse_iso_date(self.left_iso)
        dr = _parse_iso_date(self.right_iso)
        if not dl or not dr:
            return None
        return float((dr - dl).days)


def parse_step_candidates(prompt: str) -> List[Candidate]:
    """
    Parse STEP_CANDIDATES lines like:
    - #1: 2019-10-22 -> 2020-06-06 | delta_mm=4.800 | step_date_candidate=20200606 | est_step_mag_abs_mm≈4.400
    """
    tail = _extract_block_lines(prompt, r"^\s*STEP_CANDIDATES\b")
    if not tail:
        return []

    candidates: List[Candidate] = []
    for ln in tail:
        raw = ln.strip()
        if not raw:
            continue
        if raw.startswith("WINDOW_AROUND_TOP1_JUMP") or raw.startswith("WINDOW_AROUND_TOP1") or raw.startswith("ANCHORS"):
            break

        if not raw.startswith("- #"):
            continue

        m_rank = re.match(r"-\s*#(\d+):\s*(.*)$", raw)
        if not m_rank:
            continue
        rank = int(m_rank.group(1))
        rest = m_rank.group(2)

        m_lr = re.search(r"(\d{4}-\d{2}-\d{2})\s*->\s*(\d{4}-\d{2}-\d{2})", rest)
        if not m_lr:
            continue
        left_iso, right_iso = m_lr.group(1), m_lr.group(2)

        m_delta = re.search(r"delta_mm\s*=\s*([-+]?\d+(?:\.\d+)?)", rest)
        if not m_delta:
            continue
        delta_mm = float(m_delta.group(1))

        m_step = re.search(r"step_date_candidate\s*=\s*(\d{8})", rest)
        if not m_step:
            continue
        step_yyyymmdd = m_step.group(1)
        dt = _parse_yyyymmdd(step_yyyymmdd)
        if not dt:
            continue
        step_iso = dt.strftime("%Y-%m-%d")

        est_val: Optional[float] = None
        m_est = re.search(r"est_step_mag_abs_mm[^0-9NA+-]*([NA]|[-+]?\d+(?:\.\d+)?)", rest)
        if m_est:
            token = m_est.group(1)
            if token != "NA":
                est_val = _safe_float(token)

        candidates.append(
            Candidate(
                rank=rank,
                left_iso=left_iso,
                right_iso=right_iso,
                step_yyyymmdd=step_yyyymmdd,
                step_iso=step_iso,
                delta_mm=delta_mm,
                est_step_mag_abs_mm=est_val,
            )
        )

    candidates.sort(key=lambda c: c.rank)
    return candidates


def parse_window_points(prompt: str) -> List[Tuple[str, float]]:
    tail = _extract_block_lines(prompt, r"^\s*WINDOW_AROUND_TOP1_JUMP\b")
    if not tail:
        tail = _extract_block_lines(prompt, r"^\s*WINDOW_AROUND_TOP1")
    if not tail:
        return []
    return _parse_points_after_csv_header(tail)


def parse_anchor_points(prompt: str) -> List[Tuple[str, float]]:
    tail = _extract_block_lines(prompt, r"^\s*ANCHORS\b")
    if not tail:
        return []
    return _parse_points_after_csv_header(tail)


# -------------------------
# Persistence / scoring
# -------------------------
@dataclass
class SplitEval:
    ok: bool
    reason: str
    split_idx: int
    n_pre: int
    n_post: int
    pre_med: Optional[float] = None
    post_med: Optional[float] = None
    persist_frac: Optional[float] = None
    anchor_shift_abs: Optional[float] = None


def eval_split(points: List[Tuple[str, float]], step_iso: str, min_post_points: int) -> SplitEval:
    if not points:
        return SplitEval(ok=False, reason="no_points", split_idx=0, n_pre=0, n_post=0)

    split_idx = None
    for i, (d, _) in enumerate(points):
        if d >= step_iso:
            split_idx = i
            break
    if split_idx is None:
        split_idx = len(points)

    pre = [v for (_, v) in points[:split_idx]]
    post = [v for (_, v) in points[split_idx:]]

    n_pre = len(pre)
    n_post = len(post)

    if n_pre < 2:
        return SplitEval(ok=False, reason="n_pre<2", split_idx=split_idx, n_pre=n_pre, n_post=n_post)
    if n_post < min_post_points:
        return SplitEval(ok=False, reason=f"n_post<{min_post_points}", split_idx=split_idx, n_pre=n_pre, n_post=n_post)

    pre_med = float(median(pre))
    post_med = float(median(post))
    anchor_shift_abs = float(abs(post_med - pre_med))

    good = 0
    for y in post:
        if abs(y - post_med) <= abs(y - pre_med):
            good += 1
    persist_frac = good / max(1, n_post)

    return SplitEval(
        ok=True,
        reason="ok",
        split_idx=split_idx,
        n_pre=n_pre,
        n_post=n_post,
        pre_med=pre_med,
        post_med=post_med,
        persist_frac=float(persist_frac),
        anchor_shift_abs=anchor_shift_abs,
    )


@dataclass
class CandEval:
    cand: Candidate
    mag_used: float
    snr: float
    used_points: str  # "window" or "anchors"
    split: SplitEval
    pass_all: bool
    support_gap: float
    pf_override_ok: bool  # window-only (FIX7, gated)


def choose_mag_used(c: Candidate, use_abs_delta: bool) -> float:
    if use_abs_delta:
        return float(abs(c.delta_mm))
    if c.est_step_mag_abs_mm is not None and not math.isnan(c.est_step_mag_abs_mm):
        return float(abs(c.est_step_mag_abs_mm))
    return float(abs(c.delta_mm))


def pick_best_candidate(
    candidates: List[Candidate],
    rmse_mm: float,
    window_pts: List[Tuple[str, float]],
    anchor_pts: List[Tuple[str, float]],
    use_abs_delta: bool,
    mag_thr: float,
    snr_thr: float,
    persist_frac_thr: float,
    min_post_points: int,
    # ---- FIX7 args (now optional; None => disabled) ----
    support_gap_max: Optional[float],
    persist_frac_soft_min: Optional[float],
    pf_override_snr_margin: Optional[float],
    pf_override_mag_margin: Optional[float],
) -> Tuple[Optional[CandEval], Dict[str, Any]]:
    """
    Evaluate all candidates and pick best.
    Returns (best_eval_or_None, debug_obj)
    """

    # -------------------------
    # FIX5 knobs (kept; always enabled)
    # -------------------------
    PERSIST_RELAX = 0.10
    MIN_WIN_ANCHOR_SHIFT = 1.50
    MIN_ANCHOR_SHIFT_ABS = 0.50

    dbg: Dict[str, Any] = {
        "sigma_rmse_mm": rmse_mm,
        "n_candidates": len(candidates),
        "has_window": bool(window_pts),
        "has_anchors": bool(anchor_pts),
        "fix5": {
            "PERSIST_RELAX": PERSIST_RELAX,
            "MIN_WIN_ANCHOR_SHIFT": MIN_WIN_ANCHOR_SHIFT,
            "MIN_ANCHOR_SHIFT_ABS": MIN_ANCHOR_SHIFT_ABS,
        },
        "fix7_gated": {
            "support_gap_max": support_gap_max,
            "persist_frac_soft_min": persist_frac_soft_min,
            "pf_override_snr_margin": pf_override_snr_margin,
            "pf_override_mag_margin": pf_override_mag_margin,
        },
    }

    evals: List[CandEval] = []

    for c in candidates:
        mag_used = float(choose_mag_used(c, use_abs_delta=use_abs_delta))
        snr = float(mag_used / max(1e-9, rmse_mm))

        if c.rank == 1 and window_pts:
            pts = window_pts
            used = "window"
        else:
            pts = anchor_pts
            used = "anchors"

        split = eval_split(pts, c.step_iso, min_post_points=min_post_points)

        mag_ok = (mag_used >= float(mag_thr))
        snr_ok = (snr >= float(snr_thr))

        persist_frac = split.persist_frac
        persist_frac = float(persist_frac) if persist_frac is not None else -1.0

        anchor_shift_abs = split.anchor_shift_abs
        anchor_shift_abs = float(anchor_shift_abs) if anchor_shift_abs is not None else None

        # support_gap always computed for debug; may be used by gated anchors check
        support_gap = abs(float(mag_used) - float(anchor_shift_abs)) if anchor_shift_abs is not None else 1e9

        # -------------------------
        # FIX7 (gated): support gap gate (anchors only)
        # default: not enabled
        # -------------------------
        support_gap_ok = True
        if support_gap_max is not None:
            support_gap_ok = (support_gap <= float(support_gap_max))

        # -------------------------
        # FIX7 (gated): persist frac strict + optional soft override (window only)
        # -------------------------
        pf_ok = (persist_frac >= float(persist_frac_thr))
        pf_override_ok = False

        # only if persist_frac_soft_min is provided
        if (not pf_ok) and (used == "window") and (persist_frac_soft_min is not None):
            soft_ok = (persist_frac >= float(persist_frac_soft_min))

            # margins only if BOTH provided
            if (pf_override_snr_margin is not None) and (pf_override_mag_margin is not None):
                strong_ok = (
                    (snr >= float(snr_thr) + float(pf_override_snr_margin))
                    and (mag_used >= float(mag_thr) + float(pf_override_mag_margin))
                )
                pf_ok = bool(soft_ok and strong_ok)
                pf_override_ok = bool(pf_ok)
            else:
                # if you want "no margins => don't relax", change this line to: pf_ok = False
                pf_ok = bool(soft_ok)

        persist_ok = bool(pf_ok)

        # -------------------------
        # FIX5: window relax via median gap (always enabled)
        # -------------------------
        if (not persist_ok) and (used == "window"):
            if (
                (persist_frac >= (float(persist_frac_thr) - float(PERSIST_RELAX)))
                and (anchor_shift_abs is not None)
                and (anchor_shift_abs >= float(MIN_WIN_ANCHOR_SHIFT))
            ):
                persist_ok = True

        # anchors fuse: small median gap => never allow
        if used == "anchors":
            if (anchor_shift_abs is None) or (anchor_shift_abs < float(MIN_ANCHOR_SHIFT_ABS)):
                persist_ok = False

        # FIX7 gated: anchors support_gap filter (only if support_gap_max provided)
        if used == "anchors":
            if not support_gap_ok:
                persist_ok = False

        passes = bool(mag_ok and snr_ok and persist_ok)
        pass_all = bool(split.ok and passes)

        evals.append(
            CandEval(
                cand=c,
                mag_used=float(mag_used),
                snr=float(snr),
                used_points=used,
                split=split,
                pass_all=pass_all,
                support_gap=float(support_gap),
                pf_override_ok=bool(pf_override_ok),
            )
        )

    def summarize(e: CandEval) -> Dict[str, Any]:
        return {
            "rank": e.cand.rank,
            "step_iso": e.cand.step_iso,
            "mag_used": e.mag_used,
            "snr": e.snr,
            "used_points": e.used_points,
            "persist_frac": e.split.persist_frac,
            "anchor_shift_abs": e.split.anchor_shift_abs,
            "support_gap": e.support_gap,
            "gap_days": e.cand.gap_days,
            "pass_all": e.pass_all,
            "pf_override_ok": e.pf_override_ok,
        }

    if evals:
        dbg["top1"] = summarize(evals[0])

    passers = [e for e in evals if e.pass_all]
    dbg["n_pass"] = len(passers)

    # -------------------------
    # FIX6 selection: WINDOW ONLY
    # -------------------------
    win_passers = [p for p in passers if p.used_points == "window"]

    if len(win_passers) == 0:
        dbg["pool_used"] = "none_no_window_passer"
        dbg["picked"] = {"method": "none", "reason": "no_window_passer"}
        dbg["passers_top3"] = []
        return None, dbg

    pool = win_passers
    dbg["pool_used"] = "window_only"

    def _k(p: CandEval):
        pf = p.split.persist_frac
        pf_v = float(pf) if pf is not None else -1.0
        snr_v = float(p.snr) if p.snr is not None else -1.0
        ash = p.split.anchor_shift_abs
        ash_v = float(ash) if ash is not None else -1.0
        gd = p.cand.gap_days
        gd_v = float(gd) if gd is not None else 1e9
        # persist desc, snr desc, anchor_shift_abs desc, gap_days asc
        return (-pf_v, -snr_v, -ash_v, gd_v)

    pool_sorted = sorted(pool, key=_k)
    best = pool_sorted[0]
    dbg["passers_top3"] = [summarize(x) for x in pool_sorted[:3]]

    dbg["picked"] = {
        "method": "top1_pass_keep" if best.cand.rank == 1 else "tiebreak_over_passers",
        "rank": best.cand.rank,
        "used_points": best.used_points,
        "answer_mag": best.mag_used,
        "persist_frac": best.split.persist_frac,
        "anchor_shift_abs": best.split.anchor_shift_abs,
        "gap_days": best.cand.gap_days,
        "support_gap": best.support_gap,
        "pf_override_ok": best.pf_override_ok,
    }

    return best, dbg


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_jsonl", required=True)
    ap.add_argument("--routeA_prompts_jsonl", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--mag_thr", type=float, default=2.5)
    ap.add_argument("--snr_thr", type=float, default=2.0)
    ap.add_argument("--persist_frac_thr", type=float, default=0.75)
    ap.add_argument("--min_post_points", type=int, default=5)

    ap.add_argument("--use_abs_delta", action="store_true")
    ap.add_argument("--dump_debug", action="store_true")

    # ---- FIX7 gated args: defaults None (disabled) ----
    ap.add_argument(
        "--support_gap_max",
        type=float,
        default=None,
        help=(
            "GATED (default disabled). Only for anchors-based candidates: require "
            "support_gap <= this. support_gap = abs(mag_used - anchor_shift_abs)."
        ),
    )
    ap.add_argument(
        "--persist_frac_soft_min",
        type=float,
        default=None,
        help=(
            "GATED (default disabled). For window candidates only: allow persist_frac < "
            "persist_frac_thr if still >= this (and optionally also meets strong margins)."
        ),
    )
    ap.add_argument(
        "--pf_override_snr_margin",
        type=float,
        default=None,
        help="GATED. Window-only soft persist: require snr >= snr_thr + margin (only if both margins are provided).",
    )
    ap.add_argument(
        "--pf_override_mag_margin",
        type=float,
        default=None,
        help="GATED. Window-only soft persist: require mag_used >= mag_thr + margin (only if both margins are provided).",
    )

    args = ap.parse_args()

    id_rows = list(read_jsonl(args.ids_jsonl))
    total = len(id_rows)

    prompts_by_id: Dict[str, str] = {}
    for r in read_jsonl(args.routeA_prompts_jsonl):
        rid = r.get("id")
        p = r.get("prompt")
        if isinstance(rid, str) and isinstance(p, str):
            prompts_by_id[rid] = p

    out_rows: List[Dict[str, Any]] = []

    stats = {
        "total": total,
        "c2": 0,
        "pred_step": 0,
        "pred_none": 0,
        "parse_fail": 0,
        "missing_prompt": 0,
    }

    model_name = "c2_from_routeA_a3_prompt_v2"  # keep name stable for your pipeline

    for r in id_rows:
        _id = r.get("id")
        task = r.get("task_name")

        base_out = {
            "id": _id,
            "task_name": task,
            "model": model_name,
            "answer_parsed": None,
            "answer_numeric": None,
            "parse_error": None,
        }

        if task != "C2_step_mag" or not isinstance(_id, str):
            out_rows.append(base_out)
            continue

        stats["c2"] += 1

        prompt = prompts_by_id.get(_id)
        if not prompt:
            stats["missing_prompt"] += 1
            base_out["parse_error"] = "missing_prompt"
            base_out["answer_parsed"] = "none"
            out_rows.append(base_out)
            continue

        rmse = parse_rmse_mm(prompt)
        if rmse is None or rmse <= 0:
            rmse = 1.0

        anchors = parse_anchor_points(prompt)
        window = parse_window_points(prompt)
        candidates = parse_step_candidates(prompt)

        if not candidates:
            stats["parse_fail"] += 1
            base_out["parse_error"] = "no_candidates_parsed"
            base_out["answer_parsed"] = "none"
            if args.dump_debug:
                base_out["debug"] = {
                    "sigma_rmse_mm": rmse,
                    "has_window": bool(window),
                    "has_anchors": bool(anchors),
                    "n_candidates": 0,
                    "picked": {"method": "none", "reason": "no_candidates_parsed"},
                }
            out_rows.append(base_out)
            stats["pred_none"] += 1
            continue

        best, dbg = pick_best_candidate(
            candidates=candidates,
            rmse_mm=rmse,
            window_pts=window,
            anchor_pts=anchors,
            use_abs_delta=args.use_abs_delta,
            mag_thr=args.mag_thr,
            snr_thr=args.snr_thr,
            persist_frac_thr=args.persist_frac_thr,
            min_post_points=args.min_post_points,
            support_gap_max=args.support_gap_max,
            persist_frac_soft_min=args.persist_frac_soft_min,
            pf_override_snr_margin=args.pf_override_snr_margin,
            pf_override_mag_margin=args.pf_override_mag_margin,
        )

        if best is None:
            base_out["answer_parsed"] = "none"
            stats["pred_none"] += 1
        else:
            base_out["answer_parsed"] = float(best.mag_used)
            base_out["answer_numeric"] = float(best.mag_used)
            stats["pred_step"] += 1

        if args.dump_debug:
            base_out["debug"] = dbg

        out_rows.append(base_out)

    write_jsonl(args.out, out_rows)

    print(f"[OK] wrote: {args.out}")
    print(
        "stats: "
        f"total={stats['total']}  c2={stats['c2']}  "
        f"pred_step={stats['pred_step']}  pred_none={stats['pred_none']}  "
        f"parse_fail={stats['parse_fail']}  missing_prompt={stats['missing_prompt']}"
    )
    print(
        "thr: "
        f"mag_thr={args.mag_thr}  snr_thr={args.snr_thr}  "
        f"persist_frac_thr={args.persist_frac_thr}  min_post_points={args.min_post_points}  "
        f"use_abs_delta={bool(args.use_abs_delta)}"
    )
    print(
        "fix7_gated: "
        f"support_gap_max={args.support_gap_max}  persist_frac_soft_min={args.persist_frac_soft_min}  "
        f"pf_override_snr_margin={args.pf_override_snr_margin}  pf_override_mag_margin={args.pf_override_mag_margin}"
    )


if __name__ == "__main__":
    main()
