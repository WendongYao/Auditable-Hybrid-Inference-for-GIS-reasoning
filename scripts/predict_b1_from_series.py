#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rule-based predictor for EGMS-QA task: B1_seasonality_present.

Produces a jsonl predictions file compatible with the EGMS-QA scoring/merging
pipeline used in this repo.

Key idea
- Fit (optional) linear trend
- Estimate annual sinus amplitude (or robust alternatives)
- Compute SNR using a noise estimate (rmse from meta or robust diff-MAD)
- Classify as seasonal/nonseasonal using amp_thr + snr_thr

Output schema (per line):
  {
    "id": ...,
    "task_name": "B1_seasonality_present",
    "model": "b1_rule_from_series_v1",
    "answer_parsed": "seasonal"|"nonseasonal",
    "answer_numeric": null,
    "parse_error": false,
    "debug": {...}   # optional when --dump_debug
  }

PowerShell example:
  python scripts/predict_b1_from_series.py `
    --ids_jsonl $IDS `
    --dataset $DATA `
    --out $P_B1 `
    --amp_mode sin `
    --sin_period_days 365.25 `
    --amp_thr 1.0 `
    --snr_thr 1.5 `
    --noise_mode rmse `
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

TASK_NAME = "B1_seasonality_present"


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
                # fallback: try common fields
                for k in ("qid", "sample_id", "uid"):
                    if isinstance(obj, dict) and k in obj:
                        ids.append(str(obj[k]))
                        break
    return ids


# -----------------------------
# Series helpers
# -----------------------------
def _parse_iso_date_to_ordinal_days(iso: str) -> Optional[int]:
    """Return integer ordinal days (datetime.toordinal) or None."""
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
    """Extract valid (t_days, y_mm) arrays from a dataset row."""
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
    """Robust sigma estimate using MAD scaled to std."""
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    # 0.6745 is median(|Z|) for Z~N(0,1)
    sigma = mad / 0.6744897501960817 if mad > 0 else 0.0
    return float(sigma)


def estimate_noise(row: dict, t_days: np.ndarray, y: np.ndarray, mode: str) -> float:
    """Noise estimate in mm."""
    mode = (mode or "").lower().strip()
    if mode == "rmse":
        meta = row.get("meta", {}) or {}
        v = meta.get("rmse_mm", None)
        if v is not None:
            try:
                vv = float(v)
                if math.isfinite(vv) and vv > 0:
                    return vv
            except Exception:
                pass
        # fallback if meta missing
        mode = "mad_d1"

    if y.size < 3:
        return 1e-6

    if mode == "mad":
        sig = robust_mad_sigma(y)
        return max(sig, 1e-6)

    # default: mad of first differences (better for trend + seasonal)
    dy = np.diff(y)
    sig = robust_mad_sigma(dy)
    return max(sig, 1e-6)


# -----------------------------
# Amplitude estimators
# -----------------------------
@dataclass
class SinFitResult:
    amp: float
    offset: float
    a_sin: float
    b_cos: float
    r2: float


def sinfit_amp(
    t_days: np.ndarray,
    y: np.ndarray,
    period_days: float,
) -> SinFitResult:
    """Least-squares fit y = c + a*sin(wt) + b*cos(wt).
    Returns amplitude = sqrt(a^2+b^2).
    """
    if y.size < 4 or t_days.size != y.size:
        return SinFitResult(amp=float("nan"), offset=float("nan"), a_sin=float("nan"), b_cos=float("nan"), r2=float("nan"))

    w = 2.0 * math.pi / float(period_days)
    s = np.sin(w * t_days)
    c = np.cos(w * t_days)
    X = np.stack([np.ones_like(t_days), s, c], axis=1)
    # Solve least squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    a_sin = float(beta[1])
    b_cos = float(beta[2])
    amp = float(math.sqrt(a_sin * a_sin + b_cos * b_cos))
    return SinFitResult(amp=amp, offset=float(beta[0]), a_sin=a_sin, b_cos=b_cos, r2=float(r2))


def amp_peak(y: np.ndarray) -> float:
    return float(np.max(y) - np.min(y)) / 2.0 if y.size else float("nan")


def amp_p2p(y: np.ndarray) -> float:
    return float(np.max(y) - np.min(y)) if y.size else float("nan")


def amp_p95(y: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    lo = float(np.percentile(y, 2.5))
    hi = float(np.percentile(y, 97.5))
    return (hi - lo) / 2.0


def choose_amp(
    t_days: np.ndarray,
    y: np.ndarray,
    mode: str,
    period_days: float,
) -> Tuple[float, Dict[str, Any]]:
    mode = (mode or "").lower().strip()
    dbg: Dict[str, Any] = {"amp_mode": mode}

    if y.size == 0:
        return float("nan"), dbg

    if mode == "sin":
        res = sinfit_amp(t_days, y, period_days=period_days)
        dbg.update({"sin_amp": res.amp, "sin_r2": res.r2, "a_sin": res.a_sin, "b_cos": res.b_cos, "offset": res.offset})
        return float(res.amp), dbg

    if mode == "peak":
        a = amp_peak(y)
        dbg["peak_amp"] = a
        return a, dbg

    if mode == "p2p":
        a = amp_p2p(y)
        dbg["p2p"] = a
        return a, dbg

    if mode == "p95":
        a = amp_p95(y)
        dbg["p95_amp"] = a
        return a, dbg

    # default fallback
    res = sinfit_amp(t_days, y, period_days=period_days)
    dbg.update({"sin_amp": res.amp, "sin_r2": res.r2})
    return float(res.amp), dbg


# -----------------------------
# Prediction record
# -----------------------------
def make_pred_record(_id: str, ans: str, debug: Optional[dict]) -> dict:
    return {
        "id": _id,
        "task_name": TASK_NAME,
        "model": "b1_rule_from_series_v1",
        "answer_parsed": ans,
        "answer_numeric": None,
        "parse_error": False,
        **({"debug": debug} if debug is not None else {}),
    }


# -----------------------------
# Main
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_jsonl", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--amp_mode", default="sin", choices=["sin", "peak", "p2p", "p95"])
    ap.add_argument("--sin_period_days", type=float, default=365.25)

    ap.add_argument("--amp_thr", type=float, default=1.0, help="Amplitude threshold in mm")
    ap.add_argument("--snr_thr", type=float, default=1.5, help="SNR threshold (amp/noise)")

    ap.add_argument("--noise_mode", default="rmse", choices=["rmse", "mad", "mad_d1"])

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
    n_seasonal = 0
    n_nonseasonal = 0
    n_missing = 0
    n_empty = 0

    for _id in ids:
        row = data.get(_id)
        if row is None:
            n_missing += 1
            continue

        t_days, y = extract_series_xy(row)
        if y.size < 6:
            # too short -> default nonseasonal
            n_empty += 1
            ans = "nonseasonal"
            out_recs.append(make_pred_record(_id, ans, debug={"reason": "too_few_points", "n": int(y.size)} if args.dump_debug else None))
            n_nonseasonal += 1
            continue

        amp, amp_dbg = choose_amp(t_days, y, mode=args.amp_mode, period_days=args.sin_period_days)
        noise = estimate_noise(row, t_days, y, mode=args.noise_mode)
        snr = float(amp / noise) if (math.isfinite(amp) and noise > 0) else float("nan")

        is_seasonal = (math.isfinite(amp) and math.isfinite(snr) and (amp >= args.amp_thr) and (snr >= args.snr_thr))
        ans = "seasonal" if is_seasonal else "nonseasonal"

        if ans == "seasonal":
            n_seasonal += 1
        else:
            n_nonseasonal += 1

        debug = None
        if args.dump_debug:
            debug = {
                "n": int(y.size),
                "amp": amp,
                "noise": noise,
                "snr": snr,
                "thr": {"amp_thr": args.amp_thr, "snr_thr": args.snr_thr},
                **amp_dbg,
            }

        out_recs.append(make_pred_record(_id, ans, debug=debug))

    write_jsonl(args.out, out_recs)

    print(f"[OK] wrote: {args.out}")
    print(f"stats: total={len(ids)} seasonal={n_seasonal} nonseasonal={n_nonseasonal} missing={n_missing} too_short={n_empty}")
    print(f"thr: amp_thr={args.amp_thr} snr_thr={args.snr_thr} amp_mode={args.amp_mode} sin_period_days={args.sin_period_days} noise_mode={args.noise_mode}")


if __name__ == "__main__":
    main()
