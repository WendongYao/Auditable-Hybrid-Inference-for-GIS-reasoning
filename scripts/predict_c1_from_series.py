#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# scripts/predict_c1_from_series.py
import argparse, json, datetime
from dataclasses import dataclass
from statistics import median
from typing import Dict, Any, List, Optional, Tuple


TASK = "C1_step_time"


def read_ids(ids_jsonl: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    with open(ids_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            _id = o.get("id")
            tn = o.get("task_name")
            if _id and tn:
                m[str(_id)] = str(tn)
    return m


def load_dataset_rows(dataset_jsonl: str, want_ids: set) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(dataset_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            _id = o.get("id")
            if _id in want_ids:
                out[_id] = o
    return out


def _safe_date10(s: str) -> Optional[datetime.date]:
    try:
        return datetime.date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def _load_points(o: Dict[str, Any]) -> List[Tuple[datetime.date, float]]:
    s = (o.get("context") or {}).get("series") or {}
    t_iso = s.get("t_iso") or []
    yy = s.get("d_mm") or []
    mm = s.get("mask") or [1] * len(yy)

    pts: List[Tuple[datetime.date, float]] = []
    if not (t_iso and yy and len(t_iso) == len(yy)):
        return pts

    for ti, xi, mi in zip(t_iso, yy, mm):
        try:
            if int(mi) != 1:
                continue
        except Exception:
            continue
        d = _safe_date10(ti)
        if d is None:
            continue
        try:
            v = float(xi)
        except Exception:
            continue
        pts.append((d, v))

    pts.sort(key=lambda x: x[0])
    return pts


def _rmse_mm(o: Dict[str, Any], default: float = 1.0) -> float:
    try:
        v = float(((o.get("meta") or {}).get("rmse_mm")) or 0.0)
    except Exception:
        v = 0.0
    return v if v > 0 else float(default)


@dataclass
class Cand:
    idx: int
    step_date: datetime.date  # first post point date
    gap_days: int
    mag: float
    snr: float
    persist_frac: float
    edge_jump: float
    pre_med: float
    post_med: float
    n_pre: int
    n_post: int


def _persist_frac(pre_med: float, post_med: float, post_vals: List[float]) -> float:
    good = 0
    for y in post_vals:
        if abs(y - post_med) <= abs(y - pre_med):
            good += 1
    return good / max(1, len(post_vals))


def find_best_step(
    pts: List[Tuple[datetime.date, float]],
    rmse: float,
    min_pre_points: int,
    min_post_points: int,
    mag_thr: float,
    snr_thr: float,
    persist_frac_thr: float,
    edge_jump_thr: float,
    max_gap_days: Optional[int],
) -> Tuple[Optional[Cand], Dict[str, Any]]:
    dbg: Dict[str, Any] = {
        "n_pts": len(pts),
        "rmse_mm": rmse,
        "min_pre_points": min_pre_points,
        "min_post_points": min_post_points,
        "thr": {
            "mag_thr": mag_thr,
            "snr_thr": snr_thr,
            "persist_frac_thr": persist_frac_thr,
            "edge_jump_thr": edge_jump_thr,
            "max_gap_days": max_gap_days,
        },
    }

    if len(pts) < (min_pre_points + min_post_points):
        dbg["reason"] = "too_few_points"
        return None, dbg

    dates = [d for d, _ in pts]
    vals = [v for _, v in pts]

    cands: List[Cand] = []
    n = len(pts)

    for i in range(min_pre_points, n - min_post_points + 1):
        # boundary between i-1 and i
        dL, vL = dates[i - 1], vals[i - 1]
        dR, vR = dates[i], vals[i]
        gap = (dR - dL).days

        if (max_gap_days is not None) and (gap > max_gap_days):
            continue

        pre = vals[:i]
        post = vals[i:]
        n_pre = len(pre)
        n_post = len(post)

        pre_med = float(median(pre))
        post_med = float(median(post))
        mag = float(abs(post_med - pre_med))
        snr = float(mag / max(1e-9, rmse))
        pf = float(_persist_frac(pre_med, post_med, post))
        edge_jump = float(abs(vR - vL))

        if mag < mag_thr:
            continue
        if snr < snr_thr:
            continue
        if pf < persist_frac_thr:
            continue
        if edge_jump < edge_jump_thr:
            continue

        cands.append(Cand(
            idx=i,
            step_date=dR,
            gap_days=int(gap),
            mag=mag,
            snr=snr,
            persist_frac=pf,
            edge_jump=edge_jump,
            pre_med=pre_med,
            post_med=post_med,
            n_pre=n_pre,
            n_post=n_post,
        ))

    dbg["n_pass"] = len(cands)
    if not cands:
        dbg["reason"] = "no_candidate_pass"
        return None, dbg

    # pick best: persist desc, snr desc, mag desc, edge_jump desc, gap_days asc
    def key(c: Cand):
        return (-c.persist_frac, -c.snr, -c.mag, -c.edge_jump, c.gap_days)

    cands.sort(key=key)
    best = cands[0]

    dbg["picked"] = {
        "idx": best.idx,
        "step_date": best.step_date.isoformat(),
        "gap_days": best.gap_days,
        "mag": best.mag,
        "snr": best.snr,
        "persist_frac": best.persist_frac,
        "edge_jump": best.edge_jump,
        "pre_med": best.pre_med,
        "post_med": best.post_med,
        "n_pre": best.n_pre,
        "n_post": best.n_post,
    }
    dbg["top3"] = [
        {
            "idx": c.idx,
            "step_date": c.step_date.isoformat(),
            "gap_days": c.gap_days,
            "mag": c.mag,
            "snr": c.snr,
            "persist_frac": c.persist_frac,
            "edge_jump": c.edge_jump,
        }
        for c in cands[:3]
    ]
    return best, dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_jsonl", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--mag_thr", type=float, default=2.0)
    ap.add_argument("--snr_thr", type=float, default=1.5)
    ap.add_argument("--persist_frac_thr", type=float, default=0.7)
    ap.add_argument("--edge_jump_thr", type=float, default=0.0, help="0.0 means disabled")
    ap.add_argument("--min_pre_points", type=int, default=2)
    ap.add_argument("--min_post_points", type=int, default=5)
    ap.add_argument("--max_gap_days", type=int, default=120, help="set <=0 to disable")

    ap.add_argument("--dump_debug", action="store_true")
    args = ap.parse_args()

    max_gap_days = None if (args.max_gap_days is None or args.max_gap_days <= 0) else int(args.max_gap_days)

    id2task = read_ids(args.ids_jsonl)
    want_ids = {k for k, v in id2task.items() if v == TASK}
    rows = load_dataset_rows(args.dataset, want_ids)

    n_total = 0
    n_step = 0
    n_none = 0
    n_missing = 0

    with open(args.out, "w", encoding="utf-8") as fw:
        for _id in sorted(want_ids):
            n_total += 1
            o = rows.get(_id)
            if not o:
                n_missing += 1
                fw.write(json.dumps({
                    "id": _id,
                    "task_name": TASK,
                    "model": "c1_from_series_rule",
                    "answer_parsed": "none",
                    "answer_numeric": None,
                    "parse_error": "missing_dataset_row",
                }, ensure_ascii=False) + "\n")
                continue

            pts = _load_points(o)
            rmse = _rmse_mm(o, default=1.0)

            best, dbg = find_best_step(
                pts=pts,
                rmse=rmse,
                min_pre_points=int(args.min_pre_points),
                min_post_points=int(args.min_post_points),
                mag_thr=float(args.mag_thr),
                snr_thr=float(args.snr_thr),
                persist_frac_thr=float(args.persist_frac_thr),
                edge_jump_thr=float(args.edge_jump_thr),
                max_gap_days=max_gap_days,
            )

            if best is None:
                pred = "none"
                n_none += 1
            else:
                pred = best.step_date.isoformat()
                n_step += 1

            out_obj = {
                "id": o.get("id", _id),
                "task_name": TASK,
                "model": "c1_from_series_rule",
                "answer_parsed": pred,
                "answer_numeric": None,
                "parse_error": None,
            }
            if args.dump_debug:
                out_obj["debug"] = dbg

            fw.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"[OK] wrote: {args.out}")
    print(f"stats: total={n_total} pred_step={n_step} pred_none={n_none} missing={n_missing}")
    print(
        "thr: "
        f"mag_thr={args.mag_thr} snr_thr={args.snr_thr} persist_frac_thr={args.persist_frac_thr} "
        f"edge_jump_thr={args.edge_jump_thr} min_pre_points={args.min_pre_points} min_post_points={args.min_post_points} "
        f"max_gap_days={max_gap_days}"
    )


if __name__ == "__main__":
    main()
