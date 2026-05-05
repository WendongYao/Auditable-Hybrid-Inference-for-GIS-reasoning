# egmsqa/baselines/step_residual_detector.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

DATE_YYYYMMDD = "%Y%m%d"

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
    if dt:
        return dt.strftime(DATE_YYYYMMDD)
    digits = "".join([c for c in str(s) if c.isdigit()])
    return digits[:8] if len(digits) >= 8 else str(s)

def _finite_idx(y: List[Any]) -> List[int]:
    idx=[]
    for i,v in enumerate(y):
        if v is None:
            continue
        try:
            fv=float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            idx.append(i)
    return idx

def _mad_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float(np.nanstd(x)) if x.size else 1.0
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    # 0.6745 makes MAD consistent with std for Gaussian
    s = mad / 0.6745 if mad > 0 else float(np.nanstd(x))
    return float(s if np.isfinite(s) and s > 1e-9 else 1.0)

def fit_trend_seasonal(t_days: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    y(t) = a + b*t + c*sin(wt) + d*cos(wt), w = 2*pi/365.25
    returns y_hat on given t_days
    """
    w = 2.0 * np.pi / 365.25
    X = np.stack([
        np.ones_like(t_days),
        t_days,
        np.sin(w * t_days),
        np.cos(w * t_days),
    ], axis=1)
    # least squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return X @ beta

def scan_step_on_residual(
    t_iso: List[str],
    y_mm: List[Any],
    rmse_mm: Optional[float] = None,
    k_side: int = 3,
    topk: int = 5,
) -> Dict[str, Any]:
    """
    在 residual 上找“持久 level shift”的最优边界。
    输出包含：present / best_i / step_date / mag_abs / snr / improve_frac / candidates_topk 等
    """
    valid = _finite_idx(y_mm)
    if len(valid) < (2 * k_side + 2):
        return {"present": False, "reason": "insufficient_valid"}

    # build aligned arrays
    t0 = parse_date_any(t_iso[valid[0]])
    if not t0:
        return {"present": False, "reason": "bad_time"}

    t_days = []
    y = []
    vidx = []
    for i in valid:
        dt = parse_date_any(t_iso[i])
        if not dt:
            continue
        t_days.append((dt - t0).days)
        y.append(float(y_mm[i]))
        vidx.append(i)
    t_days = np.asarray(t_days, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if y.size < (2 * k_side + 2):
        return {"present": False, "reason": "insufficient_time_parsed"}

    y_hat = fit_trend_seasonal(t_days, y)
    r = y - y_hat

    sigma = float(rmse_mm) if (rmse_mm is not None and rmse_mm > 0) else _mad_sigma(r)

    # 计算每个边界的 robust step_mag 与 improve_frac
    # 边界定义在“原始索引 i | i+1”（但我们用 valid 序列上的相邻点）
    candidates: List[Tuple[float,int,Dict[str,Any]]] = []
    for j in range(k_side, len(vidx) - k_side):
        # boundary between vidx[j-1] and vidx[j]
        left_end = j
        right_start = j

        pre = r[left_end - k_side:left_end]
        post = r[right_start:right_start + k_side]
        if pre.size < k_side or post.size < k_side:
            continue

        pre_med = float(np.median(pre))
        post_med = float(np.median(post))
        mag = post_med - pre_med
        mag_abs = abs(mag)

        # improve_frac：两段常数(用median)相对单段常数(全局median)的 SSE 改善比例
        glob = float(np.median(r))
        sse0 = float(np.sum((r - glob) ** 2))
        sse1 = float(np.sum((pre - pre_med) ** 2) + np.sum((post - post_med) ** 2))
        improve_frac = (sse0 - sse1) / sse0 if sse0 > 1e-9 else 0.0

        # 持久性：post 的 median 距离 pre_med 要明显，且 post 不“回拉”
        # （轻量规则：post中位数偏移占 mag_abs 的比例要大）
        persist = (abs(post_med - pre_med) >= 0.75 * mag_abs)

        snr = mag_abs / sigma if sigma > 1e-9 else 0.0

        # score：优先大的 mag_abs，同时要有 improve_frac 与 persist
        score = (snr * 1.0) + (improve_frac * 2.0) + (0.5 if persist else 0.0)

        info = {
            "j": j,
            "left_iso": t_iso[vidx[j-1]],
            "right_iso": t_iso[vidx[j]],
            "step_date_candidate": yyyymmdd_from_iso(t_iso[vidx[j]]),
            "mag_mm": float(mag),
            "mag_abs_mm": float(mag_abs),
            "snr": float(snr),
            "improve_frac": float(improve_frac),
            "persist_flag": bool(persist),
        }
        candidates.append((score, j, info))

    if not candidates:
        return {"present": False, "reason": "no_candidates"}

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][2]

    # baseline 的“是否 step”判定阈值（先给保守但有召回的默认值；后续可以 tune）
    # 你可以把这些阈值暴露成 config
    TH_MAG = 2.0
    TH_SNR = 1.0
    TH_IMP = 0.10

    present = (best["mag_abs_mm"] >= TH_MAG) and (best["snr"] >= TH_SNR) and (best["improve_frac"] >= TH_IMP) and best["persist_flag"]

    out = {
        "present": bool(present),
        "best": best,
        "sigma_mm": float(sigma),
        "thresholds": {"mag_abs_mm": TH_MAG, "snr": TH_SNR, "improve_frac": TH_IMP},
        "candidates_topk": [c[2] for c in candidates[:topk]],
        "reason": "ok" if present else "below_threshold",
    }
    return out
