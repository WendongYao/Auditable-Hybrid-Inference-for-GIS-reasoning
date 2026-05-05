# scripts/step_residual_detector.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DATE_YYYYMMDD = "%Y%m%d"


# ---------------- time utils ----------------
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


# ---------------- numeric utils ----------------
def _finite_idx(y: List[Any]) -> List[int]:
    idx: List[int] = []
    for i, v in enumerate(y):
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            idx.append(i)
    return idx


def _mad_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 5:
        s = float(np.nanstd(x)) if x.size else 1.0
        return float(s if np.isfinite(s) and s > 1e-9 else 1.0)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    s = mad / 0.6745 if mad > 0 else float(np.nanstd(x))
    return float(s if np.isfinite(s) and s > 1e-9 else 1.0)


def _ols_fit(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid.T @ resid)
    n, p = X.shape
    dof = max(1, n - p)
    sigma2 = rss / dof
    sigma = float(np.sqrt(sigma2))
    return {"beta": beta, "resid": resid, "rss": rss, "sigma": sigma, "dof": dof}


def _se_from_xtx(X: np.ndarray, sigma: float, idx: int) -> Optional[float]:
    try:
        xtx = X.T @ X
        inv = np.linalg.inv(xtx)
        v = float(inv[idx, idx]) * (sigma**2)
        if v <= 0 or not np.isfinite(v):
            return None
        return float(np.sqrt(v))
    except Exception:
        return None


def _bic(rss: float, n: int, p: int) -> float:
    # BIC = n*ln(rss/n) + p*ln(n)
    if rss <= 1e-12:
        rss = 1e-12
    return float(n * np.log(rss / max(1, n)) + p * np.log(max(2, n)))


def _median_gap_days(t_days: np.ndarray) -> float:
    if t_days.size < 3:
        return 12.0
    diffs = np.diff(np.sort(t_days))
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 12.0
    med = float(np.median(diffs))
    return med if np.isfinite(med) and med > 1e-6 else 12.0


def _gap_penalty(gap_days: float, med_gap: float) -> float:
    """
    Mild penalty for unusually large gaps (tie-break only).
    - If gap ~ median cadence => penalty ~ 1
    - If gap >> cadence => penalty decreases slowly (does NOT zero out)
    """
    if not np.isfinite(gap_days) or gap_days <= 0:
        return 1.0
    med = med_gap if (np.isfinite(med_gap) and med_gap > 1e-6) else 12.0
    r = gap_days / med
    if r <= 1.5:
        return 1.0
    # slow decay
    return float(1.0 / np.sqrt(1.0 + (r - 1.5)))


def _local_window_stats(
    t_days: np.ndarray,
    y: np.ndarray,
    resid0: np.ndarray,
    j: int,
    k_side: int,
) -> Dict[str, Any]:
    n = int(y.size)
    j0 = max(0, j - k_side)
    j1 = min(n, j + k_side)

    pre_y = y[j0:j]
    post_y = y[j:j1]
    pre_r = resid0[j0:j]
    post_r = resid0[j:j1]

    out: Dict[str, Any] = {
        "k_side": int(k_side),
        "j": int(j),
        "pre_n": int(pre_y.size),
        "post_n": int(post_y.size),
    }

    if pre_y.size >= 2 and post_y.size >= 2:
        pre_med_y = float(np.median(pre_y))
        post_med_y = float(np.median(post_y))
        mag_y = post_med_y - pre_med_y

        pre_med_r = float(np.median(pre_r))
        post_med_r = float(np.median(post_r))
        mag_r = post_med_r - pre_med_r

        # "closer to new level" ratio (raw + residual)
        closer_y = float(np.mean(np.abs(post_y - post_med_y) <= np.abs(post_y - pre_med_y)))
        closer_r = float(np.mean(np.abs(post_r - post_med_r) <= np.abs(post_r - pre_med_r)))

        out.update(
            {
                "pre_med_y_mm": pre_med_y,
                "post_med_y_mm": post_med_y,
                "mag_win_y_mm": float(mag_y),
                "mag_win_y_abs_mm": float(abs(mag_y)),
                "pre_med_resid_mm": pre_med_r,
                "post_med_resid_mm": post_med_r,
                "mag_win_resid_mm": float(mag_r),
                "mag_win_resid_abs_mm": float(abs(mag_r)),
                "persist_ratio_y": closer_y,
                "persist_ratio_resid": closer_r,
            }
        )
    else:
        out.update(
            {
                "pre_med_y_mm": None,
                "post_med_y_mm": None,
                "mag_win_y_mm": None,
                "mag_win_y_abs_mm": None,
                "pre_med_resid_mm": None,
                "post_med_resid_mm": None,
                "mag_win_resid_mm": None,
                "mag_win_resid_abs_mm": None,
                "persist_ratio_y": None,
                "persist_ratio_resid": None,
            }
        )

    # adjacent delta (exact boundary)
    if 0 < j < n:
        out["delta_adj_y_mm"] = float(y[j] - y[j - 1])
        out["delta_adj_resid_mm"] = float(resid0[j] - resid0[j - 1])
        out["gap_days"] = float(t_days[j] - t_days[j - 1])
    else:
        out["delta_adj_y_mm"] = None
        out["delta_adj_resid_mm"] = None
        out["gap_days"] = None

    return out


# ---------------- thresholds ----------------
@dataclass
class StepThresholds:
    # present=True 的阈值（先偏向 C1 对齐：允许更多候选进 best，但最终 present 仍需证据）
    mag_abs_mm: float = 1.5
    snr: float = 1.2
    improve_frac: float = 0.01
    delta_bic: float = -1.0
    persist_frac: float = 0.70  # raw/resid 至少一个达到


# ---------------- main detector ----------------
def scan_step_on_residual(
    t_iso: List[str],
    y_mm: List[Any],
    rmse_mm: Optional[float] = None,
    k_side: int = 3,
    topk: int = 5,
) -> Dict[str, Any]:
    """
    v3.1 (two-stage candidates + segmented regression scan + raw-window evidence + gap penalty)

    Stage-0: parse/sort/align
    Stage-1: cheap candidate proposal (based on boundary delta + window shift) -> keep M
    Stage-2: evaluate only proposed candidates with segmented regression:
        y = a + b*t + c*sin(wt) + d*cos(wt) + s*H(t>=break)

    Output (stable keys):
      - present: bool
      - best: dict
      - sigma_mm: float
      - thresholds: dict
      - candidates_topk: list[dict]
      - reason: str
      - extra: dict (debug/evidence; safe to stash into stats.step.extra.*)
    """
    valid = _finite_idx(y_mm)
    if len(valid) < (2 * k_side + 6):
        return {"present": False, "reason": "insufficient_valid", "extra": {"n_valid": int(len(valid))}}

    # build aligned arrays
    parsed: List[Tuple[datetime, float, str, int]] = []
    for i in valid:
        dt = parse_date_any(t_iso[i])
        if not dt:
            continue
        try:
            v = float(y_mm[i])
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        parsed.append((dt, v, str(t_iso[i]), int(i)))

    if len(parsed) < (2 * k_side + 6):
        return {"present": False, "reason": "insufficient_time_parsed", "extra": {"n_parsed": int(len(parsed))}}

    parsed.sort(key=lambda x: x[0])
    dates = [p[0] for p in parsed]
    y = np.asarray([p[1] for p in parsed], dtype=np.float64)
    t_iso_sorted = [p[2] for p in parsed]
    n = int(y.size)

    t0 = dates[0]
    t_days = np.asarray([(d - t0).days for d in dates], dtype=np.float64)
    med_gap = _median_gap_days(t_days)

    # base model (no step): 1 + t_years + sin + cos
    t_years = t_days / 365.25
    w = 2.0 * np.pi / 365.25
    sin1 = np.sin(w * t_days)
    cos1 = np.cos(w * t_days)
    X0 = np.stack([np.ones(n), t_years, sin1, cos1], axis=1)

    fit0 = _ols_fit(X0, y)
    rss0 = float(fit0["rss"])
    resid0 = fit0["resid"]
    bic0 = _bic(rss0, n=n, p=X0.shape[1])

    sigma_used = float(rmse_mm) if (rmse_mm is not None and rmse_mm > 0) else _mad_sigma(resid0)

    thr = StepThresholds()

    # ---------------- Stage-1: propose candidates (cheap) ----------------
    min_seg = max(int(k_side), 4)
    # boundaries are j in [min_seg, n-min_seg)
    stage1: List[Tuple[float, int, Dict[str, Any]]] = []
    for j in range(min_seg, n - min_seg):
        win = _local_window_stats(t_days, y, resid0, j, k_side)

        gap_days = float(win["gap_days"]) if win.get("gap_days") is not None else float("nan")
        gp = _gap_penalty(gap_days, med_gap)

        # cheap score emphasizes boundary alignment (C1):
        # - adjacent delta is strongly rewarded
        # - window shift supports persistence
        da_r = abs(float(win["delta_adj_resid_mm"])) if win.get("delta_adj_resid_mm") is not None else 0.0
        da_y = abs(float(win["delta_adj_y_mm"])) if win.get("delta_adj_y_mm") is not None else 0.0
        mw_r = float(win["mag_win_resid_abs_mm"]) if win.get("mag_win_resid_abs_mm") is not None else 0.0
        mw_y = float(win["mag_win_y_abs_mm"]) if win.get("mag_win_y_abs_mm") is not None else 0.0

        s1 = (da_r / max(1e-9, sigma_used)) * 1.2 + (mw_r / max(1e-9, sigma_used)) * 0.6
        s1 += (da_y / max(1e-9, sigma_used)) * 0.3 + (mw_y / max(1e-9, sigma_used)) * 0.2
        s1 *= gp

        info = {
            "j": int(j),
            "left_iso": str(t_iso_sorted[j - 1]),
            "right_iso": str(t_iso_sorted[j]),
            "step_date_candidate": yyyymmdd_from_iso(t_iso_sorted[j]),
            "gap_days": float(gap_days) if np.isfinite(gap_days) else None,
            "gap_penalty": float(gp),
            "sigma_used_mm": float(sigma_used),
            "stage1_score": float(s1),
            # raw-window evidence
            "delta_adj_y_mm": win.get("delta_adj_y_mm"),
            "delta_adj_resid_mm": win.get("delta_adj_resid_mm"),
            "mag_win_y_mm": win.get("mag_win_y_mm"),
            "mag_win_y_abs_mm": win.get("mag_win_y_abs_mm"),
            "mag_win_resid_mm": win.get("mag_win_resid_mm"),
            "mag_win_resid_abs_mm": win.get("mag_win_resid_abs_mm"),
            "persist_ratio_y": win.get("persist_ratio_y"),
            "persist_ratio_resid": win.get("persist_ratio_resid"),
        }
        stage1.append((float(s1), int(j), info))

    if not stage1:
        return {"present": False, "reason": "no_stage1_candidates", "extra": {"n": int(n)}}

    stage1.sort(key=lambda x: x[0], reverse=True)

    # keep M candidates
    M = min(len(stage1), max(20, int(topk) * 6))
    cand_js = {j for _, j, _ in stage1[:M]}

    # add a few deterministic backups (quantiles + max delta) to avoid missing
    qs = [0.2, 0.4, 0.6, 0.8]
    for q in qs:
        jj = int(round(min_seg + q * (n - 2 * min_seg)))
        jj = max(min_seg, min(n - min_seg - 1, jj))
        cand_js.add(jj)

    # strongest adjacent residual delta boundary
    best_adj = None
    best_adj_val = -1.0
    for j in range(min_seg, n - min_seg):
        v = abs(float(resid0[j] - resid0[j - 1]))
        if v > best_adj_val:
            best_adj_val = v
            best_adj = j
    if best_adj is not None:
        cand_js.add(int(best_adj))

    cand_js = sorted(cand_js)

    # ---------------- Stage-2: evaluate candidates with segmented regression ----------------
    candidates2: List[Tuple[float, int, Dict[str, Any]]] = []
    for j in cand_js:
        H = np.zeros(n, dtype=np.float64)
        H[j:] = 1.0
        X1 = np.concatenate([X0, H[:, None]], axis=1)

        fit1 = _ols_fit(X1, y)
        rss1 = float(fit1["rss"])
        sigma1 = float(fit1["sigma"])
        beta1 = fit1["beta"]
        step_coef = float(beta1[-1])  # signed step on y

        improve = (rss0 - rss1) / rss0 if rss0 > 1e-12 else 0.0
        bic1 = _bic(rss1, n=n, p=X1.shape[1])
        delta_bic = float(bic1 - bic0)

        se_step = _se_from_xtx(X1, sigma1, idx=X1.shape[1] - 1)
        snr_t = (abs(step_coef) / se_step) if (se_step is not None and se_step > 0) else None
        snr = float(snr_t) if snr_t is not None else float(abs(step_coef) / max(1e-9, sigma_used))

        win = _local_window_stats(t_days, y, resid0, j, k_side)
        gap_days = float(win["gap_days"]) if win.get("gap_days") is not None else float("nan")
        gp = _gap_penalty(gap_days, med_gap)

        # persistence flag: raw OR residual window should look like new level
        pr_y = win.get("persist_ratio_y")
        pr_r = win.get("persist_ratio_resid")
        pr_y_v = float(pr_y) if pr_y is not None else 0.0
        pr_r_v = float(pr_r) if pr_r is not None else 0.0
        persist_flag = bool((pr_y_v >= thr.persist_frac) or (pr_r_v >= thr.persist_frac))

        # stage2 score: emphasize boundary alignment (adj delta) to improve C1 date match,
        # while requiring global evidence (snr/improve/bic) for present decision.
        da_r = abs(float(win["delta_adj_resid_mm"])) if win.get("delta_adj_resid_mm") is not None else 0.0
        da_y = abs(float(win["delta_adj_y_mm"])) if win.get("delta_adj_y_mm") is not None else 0.0
        mw_r = float(win["mag_win_resid_abs_mm"]) if win.get("mag_win_resid_abs_mm") is not None else 0.0

        score = 0.0
        score += snr * 0.9
        score += float(improve) * 6.0
        score += max(0.0, -delta_bic) * 0.25
        score += (da_r / max(1e-9, sigma_used)) * 1.1  # align with "jump boundary"
        score += (da_y / max(1e-9, sigma_used)) * 0.25
        score += (mw_r / max(1e-9, sigma_used)) * 0.35
        score += 1.0 if persist_flag else 0.0
        score *= gp  # gap penalty as tie-break (mild)

        info: Dict[str, Any] = {
            "j": int(j),
            "left_iso": str(t_iso_sorted[j - 1]),
            "right_iso": str(t_iso_sorted[j]),
            "step_date_candidate": yyyymmdd_from_iso(t_iso_sorted[j]),
            # segmented regression evidence
            "mag_mm": float(step_coef),
            "mag_abs_mm": float(abs(step_coef)),
            "snr": float(snr),
            "snr_t": float(snr_t) if snr_t is not None else None,
            "improve_frac": float(improve),
            "delta_bic": float(delta_bic),
            "rss0": float(rss0),
            "rss1": float(rss1),
            "bic0": float(bic0),
            "bic1": float(bic1),
            "sigma_fit1_mm": float(sigma1),
            "sigma_used_mm": float(sigma_used),
            # raw-window evidence (for debugging + future gating consistency)
            "delta_adj_y_mm": win.get("delta_adj_y_mm"),
            "delta_adj_resid_mm": win.get("delta_adj_resid_mm"),
            "mag_win_y_mm": win.get("mag_win_y_mm"),
            "mag_win_y_abs_mm": win.get("mag_win_y_abs_mm"),
            "mag_win_resid_mm": win.get("mag_win_resid_mm"),
            "mag_win_resid_abs_mm": win.get("mag_win_resid_abs_mm"),
            "persist_ratio_y": win.get("persist_ratio_y"),
            "persist_ratio_resid": win.get("persist_ratio_resid"),
            "persist_flag": bool(persist_flag),
            # gap
            "gap_days": float(gap_days) if np.isfinite(gap_days) else None,
            "median_gap_days": float(med_gap),
            "gap_penalty": float(gp),
            # scoring
            "stage2_score": float(score),
        }
        candidates2.append((float(score), int(j), info))

    if not candidates2:
        return {
            "present": False,
            "reason": "no_stage2_candidates",
            "extra": {"stage1_kept": int(M), "stage1_total": int(len(stage1))},
        }

    candidates2.sort(key=lambda x: x[0], reverse=True)
    best = candidates2[0][2]

    present = (
        (best["mag_abs_mm"] >= thr.mag_abs_mm)
        and (best["snr"] >= thr.snr)
        and (best["improve_frac"] >= thr.improve_frac)
        and (best["delta_bic"] <= thr.delta_bic)
        and bool(best["persist_flag"])
    )

    # include rich evidence for stats.step.extra.*
    extra = {
        "version": "v3.1_two_stage_segmented_regression",
        "n": int(n),
        "k_side": int(k_side),
        "topk": int(topk),
        "rmse_mm_used": float(rmse_mm) if (rmse_mm is not None and rmse_mm > 0) else None,
        "sigma_used_mm": float(sigma_used),
        "base_fit": {
            "beta0": [float(x) for x in fit0["beta"].tolist()],
            "rss0": float(rss0),
            "bic0": float(bic0),
            "sigma0_mm": float(fit0["sigma"]),
        },
        "stage1": {
            "M": int(M),
            "n_total": int(len(stage1)),
            "candidates_topM": [c[2] for c in stage1[: min(M, max(1, int(topk) * 6))]],
        },
        "stage2": {
            "n_eval": int(len(candidates2)),
            "candidate_js": [int(j) for j in cand_js],
        },
    }

    return {
        "present": bool(present),
        "best": best,
        "sigma_mm": float(sigma_used),
        "thresholds": {
            "mag_abs_mm": float(thr.mag_abs_mm),
            "snr": float(thr.snr),
            "improve_frac": float(thr.improve_frac),
            "delta_bic": float(thr.delta_bic),
            "persist_frac": float(thr.persist_frac),
        },
        "candidates_topk": [c[2] for c in candidates2[: max(1, int(topk))]],
        "reason": "ok" if present else "below_threshold",
        "extra": extra,
    }
