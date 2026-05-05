#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# scripts/predict_b2_from_series.py
import argparse, json, math
import datetime
import statistics as st
from typing import Dict, Any, List, Optional


TASK = "B2_seasonality_amp"


def read_ids(ids_jsonl: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    with open(ids_jsonl, "r", encoding="utf-8") as f:
        for line in f:
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
            o = json.loads(line)
            _id = o.get("id")
            if _id in want_ids:
                out[_id] = o
    return out


def percentile(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(xs[f])
    return float(xs[f] * (c - k) + xs[c] * (k - f))


def _solve3(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    A = [row[:] for row in A]
    b = b[:]
    n = 3
    for i in range(n):
        piv = i
        for r in range(i, n):
            if abs(A[r][i]) > abs(A[piv][i]):
                piv = r
        A[i], A[piv] = A[piv], A[i]
        b[i], b[piv] = b[piv], b[i]
        if abs(A[i][i]) < 1e-12:
            return None
        div = A[i][i]
        for j in range(i, n):
            A[i][j] /= div
        b[i] /= div
        for r in range(n):
            if r == i:
                continue
            fac = A[r][i]
            for j in range(i, n):
                A[r][j] -= fac * A[i][j]
            b[r] -= fac * b[i]
    return b


def sinfit_amp(t_days: List[float], y: List[float], period_days: float = 365.25) -> float:
    w = 2.0 * math.pi / float(period_days)
    s = [math.sin(w * t) for t in t_days]
    c = [math.cos(w * t) for t in t_days]
    n = len(y)

    Sss = sum(si * si for si in s)
    Scc = sum(ci * ci for ci in c)
    Ssc = sum(si * ci for si, ci in zip(s, c))
    Ss = sum(s)
    Sc = sum(c)
    Sy = sum(y)
    Sys = sum(yi * si for yi, si in zip(y, s))
    Syc = sum(yi * ci for yi, ci in zip(y, c))

    A = [
        [Sss, Ssc, Ss],
        [Ssc, Scc, Sc],
        [Ss,  Sc,  n],
    ]
    sol = _solve3(A, [Sys, Syc, Sy])
    if sol is None:
        return float("nan")
    a, b, _ = sol
    return float(math.sqrt(a * a + b * b))


def _is_nan(x: float) -> bool:
    return not (x == x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_jsonl", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--use_abs_amp", action="store_true", help="(kept for symmetry; amp is non-negative anyway)")

    # decision thresholds
    ap.add_argument("--amp_thr", type=float, default=1.0, help="min peak amplitude to call seasonal")
    ap.add_argument("--snr_thr", type=float, default=2.0, help="amp_used / noise >= snr_thr")
    ap.add_argument("--amp_mode", type=str, default="sin", choices=["peak", "p2p", "p95", "sin"])
    ap.add_argument("--sin_period_days", type=float, default=365.25)

    # noise source
    ap.add_argument("--noise_mode", type=str, default="max", choices=["max", "rmse"])

    ap.add_argument("--round_to", type=float, default=0.1)
    ap.add_argument("--dump_debug", action="store_true")
    args = ap.parse_args()

    id2task = read_ids(args.ids_jsonl)
    want_ids = {k for k, v in id2task.items() if v == TASK}
    rows = load_dataset_rows(args.dataset, want_ids)

    n_total = 0
    n_pred_num = 0
    n_pred_none = 0
    n_missing = 0

    with open(args.out, "w", encoding="utf-8") as fw:
        for _id in sorted(want_ids):
            n_total += 1
            o = rows.get(_id)
            if not o:
                n_missing += 1
                out_obj = {
                    "id": _id,
                    "task_name": TASK,
                    "model": "b2_from_series_rule",
                    "answer_parsed": "none",
                    "answer_numeric": None,
                    "parse_error": "missing_dataset_row",
                }
                fw.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                continue

            # Expect: o["context"]["series"] has t_iso, d_mm, mask(optional)
            try:
                s = (o.get("context") or {}).get("series") or {}
                t_iso = s.get("t_iso") or []
                yy = s.get("d_mm") or []
                mm = s.get("mask") or [1] * len(yy)
            except Exception:
                t_iso, yy, mm = [], [], []

            tt: List[float] = []
            vv: List[float] = []

            if t_iso and yy and len(t_iso) == len(yy):
                try:
                    t0 = datetime.date.fromisoformat(str(t_iso[0])[:10])
                    for ti, xi, mi in zip(t_iso, yy, mm):
                        if int(mi) != 1:
                            continue
                        d = (datetime.date.fromisoformat(str(ti)[:10]) - t0).days
                        tt.append(float(d))
                        vv.append(float(xi))
                except Exception:
                    tt, vv = [], []

            # --- amplitude candidates on displacement ---
            med = st.median(vv) if vv else 0.0
            amp_peak = max(abs(x - med) for x in vv) if vv else float("nan")
            amp_p2p = (max(vv) - min(vv)) if vv else float("nan")
            amp_p95 = (percentile(vv, 95) - percentile(vv, 5)) / 2.0 if vv else float("nan")
            amp_sin = sinfit_amp(tt, vv, period_days=args.sin_period_days) if len(vv) >= 6 else float("nan")

            if args.amp_mode == "peak":
                amp_used = amp_peak
            elif args.amp_mode == "p2p":
                amp_used = amp_p2p / 2.0  # keep "peak amplitude" scale
            elif args.amp_mode == "p95":
                amp_used = amp_p95
            else:
                amp_used = amp_sin

            # --- noise estimate ---
            if args.noise_mode == "rmse":
                try:
                    noise = float(((o.get("meta") or {}).get("rmse_mm")) or 0.0)
                except Exception:
                    noise = 0.0
                noise = noise if noise > 0 else 1e-9
            else:
                diffs = [abs(vv[i + 1] - vv[i]) for i in range(len(vv) - 1)]
                noise = (max(diffs) if diffs else 0.0) or 1e-9

            # --- snr ---
            snr = (amp_used / noise) if (noise > 0 and not _is_nan(amp_used)) else 0.0

            pred = "none"
            if (not _is_nan(amp_used)) and amp_used >= args.amp_thr and snr >= args.snr_thr:
                rt = args.round_to if args.round_to > 0 else 0.1
                pred = round(amp_used / rt) * rt

            out_obj = {
                "id": o.get("id", _id),
                "task_name": TASK,
                "model": "b2_from_series_rule",
                "answer_parsed": pred,
                "answer_numeric": (None if pred == "none" else float(pred)),
                "parse_error": None,
            }

            if args.dump_debug:
                out_obj["debug"] = {
                    "n_used": len(vv),
                    "amp_peak": (None if _is_nan(amp_peak) else float(amp_peak)),
                    "amp_p2p": (None if _is_nan(amp_p2p) else float(amp_p2p)),
                    "amp_p95": (None if _is_nan(amp_p95) else float(amp_p95)),
                    "amp_sin": (None if _is_nan(amp_sin) else float(amp_sin)),
                    "amp_used": (None if _is_nan(amp_used) else float(amp_used)),
                    "amp_mode": args.amp_mode,
                    "sin_period_days": float(args.sin_period_days),
                    "noise_mode": args.noise_mode,
                    "noise": float(noise),
                    "snr": float(snr),
                }

            fw.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            if pred == "none":
                n_pred_none += 1
            else:
                n_pred_num += 1

    print(f"[OK] wrote: {args.out}")
    print(f"stats: total={n_total} pred_num={n_pred_num} pred_none={n_pred_none} missing={n_missing}")
    print(
        "thr: "
        f"amp_thr={args.amp_thr} snr_thr={args.snr_thr} "
        f"amp_mode={args.amp_mode} sin_period_days={args.sin_period_days} "
        f"noise_mode={args.noise_mode} round_to={args.round_to}"
    )


if __name__ == "__main__":
    main()
