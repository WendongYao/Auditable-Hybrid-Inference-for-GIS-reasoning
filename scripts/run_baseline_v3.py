# scripts/run_baseline_v3.py
import argparse, json, os
import importlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

import numpy as np
from tqdm import tqdm


# ---------------- IO ----------------
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

def write_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def parse_date(s: str) -> Optional[datetime]:
    s = str(s).strip()
    if not s:
        return None
    try:
        if len(s) == 8 and s.isdigit():
            return datetime.strptime(s, "%Y%m%d")
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            return datetime.strptime(s[:10], "%Y-%m-%d")
    except Exception:
        pass
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


# ---------------- utils ----------------
def ols_fit(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(np.dot(resid, resid))
    n, p = X.shape
    dof = max(1, n - p)
    sigma2 = rss / dof
    sigma = float(np.sqrt(sigma2))
    return {"beta": beta, "rss": rss, "sigma": sigma, "dof": dof}

def slope_se_from_xtx(X: np.ndarray, sigma: float) -> Optional[float]:
    try:
        xtx = X.T @ X
        inv = np.linalg.inv(xtx)
        se2 = (sigma ** 2) * float(inv[1, 1])
        if se2 <= 0:
            return None
        return float(np.sqrt(se2))
    except Exception:
        return None


TASK_SUFFIXES = {
    "A1_trend_dir", "A2_trend_v",
    "B1_seasonality_present", "B2_seasonality_amp",
    "C1_step_time", "C2_step_mag",
}

def series_cache_key(ex: Dict[str, Any]) -> str:
    ex_id = ex.get("id", "")
    task = ex.get("task_name") or ex.get("task") or ""
    if task and isinstance(ex_id, str) and ex_id.endswith("_" + task):
        return ex_id[: -(len(task) + 1)]
    sid = ((ex.get("context") or {}).get("series") or {}).get("series_id")
    if sid:
        return str(sid)
    if isinstance(ex_id, str):
        parts = ex_id.split("_")
        if len(parts) >= 2 and parts[-1] in TASK_SUFFIXES:
            return "_".join(parts[:-1])
    return str(ex_id)


# ---------------- baseline logic ----------------
@dataclass
class BaselineConfig:
    v_stable: float = 0.5  # mm/yr

    b_snr_thr: float = 1.5
    b_improve_thr: float = 0.05
    b_amp_min: float = 0.5  # mm


def compute_stats_from_series(
    series: Dict[str, Any],
    cfg: BaselineConfig,
    rmse_mm: Optional[float] = None,
    k_side: int = 3,
    topk: int = 5,
    step_detector: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if step_detector is None:
        raise ValueError("step_detector is required (pass from main via --step_detector_module).")

    t_iso = series.get("t_iso") or []
    d_mm = series.get("d_mm") or []
    if not t_iso or not d_mm or len(t_iso) != len(d_mm):
        return {"ok": False, "reason": "bad_series"}

    # parse + finite
    dates: List[datetime] = []
    y_list: List[float] = []
    for ti, di in zip(t_iso, d_mm):
        dt = parse_date(str(ti))
        if dt is None:
            continue
        if di is None:
            continue
        try:
            v = float(di)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        dates.append(dt)
        y_list.append(v)

    if len(y_list) < 6:
        return {"ok": False, "reason": "too_few_points", "n": int(len(y_list))}

    # sort by date
    order = np.argsort(np.array([d.toordinal() for d in dates]))
    dates = [dates[i] for i in order]
    y = np.array([y_list[i] for i in order], dtype=np.float64)
    t_iso_sorted = [d.strftime("%Y-%m-%d") for d in dates]

    t0 = dates[0]
    t_days = np.array([(d - t0).days for d in dates], dtype=np.float64)
    t_years = t_days / 365.25

    n = int(len(y))
    w = 2.0 * np.pi / 365.25
    sin1 = np.sin(w * t_days)
    cos1 = np.cos(w * t_days)

    # Linear
    X_lin = np.stack([np.ones(n), t_years], axis=1)
    fit_lin = ols_fit(X_lin, y)
    slope = float(fit_lin["beta"][1])
    rss_lin, sigma_lin = float(fit_lin["rss"]), float(fit_lin["sigma"])
    slope_se = slope_se_from_xtx(X_lin, sigma_lin)
    slope_snr = (abs(slope) / slope_se) if (slope_se is not None and slope_se > 0) else None

    # Seasonal: y ~ 1 + t + sin + cos
    X_sea = np.stack([np.ones(n), t_years, sin1, cos1], axis=1)
    fit_sea = ols_fit(X_sea, y)
    rss_sea, sigma_sea = float(fit_sea["rss"]), float(fit_sea["sigma"])
    c_sin, c_cos = float(fit_sea["beta"][2]), float(fit_sea["beta"][3])
    amp = float(np.sqrt(c_sin * c_sin + c_cos * c_cos))
    improve_sea = (rss_lin - rss_sea) / rss_lin if rss_lin > 0 else 0.0
    sea_snr = (amp / sigma_sea) if sigma_sea > 0 else None

    # ---- Step (detector injected) ----
    # ---- Step (detector injected) ----
    # ---- Step (detector injected) ----
    det = step_detector(
        t_iso=t_iso_sorted,
        y_mm=y.tolist(),
        rmse_mm=rmse_mm,
        k_side=k_side,
        topk=topk,
    )

    best = det.get("best", {}) or {}

    # local safe-float helper (this file doesn't have safe_float)
    def sf(x: Any, default: float = 0.0) -> float:
        try:
            if x is None:
                return float(default)
            v = float(x)
            return v if np.isfinite(v) else float(default)
        except Exception:
            return float(default)

    # ---- field aliases for gate/debug compatibility ----
    if isinstance(best, dict) and best:
        # persist_frac alias (gate expects persist_frac)
        if "persist_frac" not in best and "persist_closer_frac" in best:
            best["persist_frac"] = best.get("persist_closer_frac")

        # date aliases (some scripts prefer date/step_date)
        if "date" not in best and "step_date_candidate" in best:
            best["date"] = best.get("step_date_candidate")
        if "step_date" not in best and "step_date_candidate" in best:
            best["step_date"] = best.get("step_date_candidate")

    # detector 自己的 loose/strict（用于记录）
    present_loose = bool(det.get("present", False))
    present_strict = bool(best.get("present_strict", False)) if isinstance(best, dict) else False

    # keep original time choice (right-side epoch)
    step_time = best.get("step_date_candidate") if isinstance(best, dict) else None

    # IMPORTANT: you currently store abs(mag) into stats.step.mag_mm (legacy behavior)
    step_mag_abs = None
    if isinstance(best, dict) and best:
        if best.get("mag_abs_mm") is not None:
            step_mag_abs = sf(best.get("mag_abs_mm"), 0.0)
        elif best.get("mag_mm") is not None:
            # fallback if detector only provides signed mag_mm
            step_mag_abs = abs(sf(best.get("mag_mm"), 0.0))

    step_snr = sf(best.get("snr"), 0.0) if isinstance(best, dict) and best else 0.0
    step_improve = sf(best.get("improve_frac", 0.0), 0.0) if isinstance(best, dict) and best else 0.0
    # 对 present_gate 更保守：delta_bic 缺失就让它失败
    step_delta_bic = sf(best.get("delta_bic"), 1e9) if isinstance(best, dict) and best else 1e9
    step_sigma = sf(det.get("sigma_mm"), 0.0) if det.get("sigma_mm") is not None else None

    persist_frac = sf(best.get("persist_frac"), 0.0) if isinstance(best, dict) and best else 0.0
    persist_flag = bool(best.get("persist_flag", False)) if isinstance(best, dict) and best else False

    # ✅ gate-aligned present：用 gate 的阈值作为 baseline 的“是否报 step”（更保守）
    present_gate = (
            (step_mag_abs is not None and step_mag_abs >= 3.0) and
            (step_snr >= 2.0) and
            (step_improve >= 0.05) and
            (persist_flag or persist_frac >= 0.7) and
            (step_delta_bic <= 0.0)
    )

    # baseline 用 gate-aligned present（保守），而不是 detector loose
    present = bool(present_gate)

    # ---- keep stats.step.* keys stable (single definition; DO NOT duplicate later) ----
    step_stats: Dict[str, Any] = {
        "time": step_time,
        "mag_mm": step_mag_abs,  # legacy: this is abs magnitude
        "snr": step_snr,
        "sigma": step_sigma,
        "rss": None,  # detector doesn't output rss
        "improve_frac": step_improve,
        "delta_bic": step_delta_bic,  # surfaced for gate/debug convenience
        "margin": int(k_side),  # compatibility
        "method": "resid_trend_season_step_v3",
    }

    # ---- extra fields for compatibility / future-proof ----
    if isinstance(best, dict) and best:
        step_stats["mag_abs_mm"] = step_mag_abs
        step_stats["persist_frac"] = persist_frac
        step_stats["persist_flag"] = persist_flag
        # optional: keep signed magnitude for debug
        if best.get("mag_mm") is not None:
            step_stats["mag_signed_mm"] = sf(best.get("mag_mm"), 0.0)

    # ---- add new info into stats.step.extra.* ----
    extra = step_stats.get("extra", {}) if isinstance(step_stats.get("extra"), dict) else {}
    extra["present"] = present
    extra["present_loose"] = present_loose
    extra["present_strict"] = present_strict
    extra["present_gate"] = bool(present_gate)

    extra["best"] = best
    extra["residual_fit"] = {
        "sigma_mm": det.get("sigma_mm"),
        "thresholds": det.get("thresholds"),
        "reason": det.get("reason"),
        "rmse_mm_used": rmse_mm,
        "k_side": int(k_side),
        "topk": int(topk),
        "detector_module": getattr(step_detector, "__module__", None),
        "detector_name": getattr(step_detector, "__name__", None),
    }
    extra["candidates_topk"] = det.get("candidates_topk", [])
    step_stats["extra"] = extra

    return {
        "ok": True,
        "n": n,
        "start": dates[0].strftime("%Y%m%d"),
        "end": dates[-1].strftime("%Y%m%d"),
        "lin": {
            "slope_mm_per_yr": slope,
            "slope_se": slope_se,
            "slope_snr": slope_snr,
            "sigma": sigma_lin,
            "rss": rss_lin,
        },
        "season": {
            "amp_mm": amp,
            "snr": sea_snr,
            "sigma": sigma_sea,
            "rss": rss_sea,
            "improve_frac": float(improve_sea),
        },
        "step": step_stats,
    }


def baseline_answer(task: str, stats: Dict[str, Any], cfg: BaselineConfig) -> Any:
    if (not stats) or (not stats.get("ok")):
        if task in {"B2_seasonality_amp", "C2_step_mag", "C1_step_time"}:
            return "none"
        if task == "A1_trend_dir":
            return "stable"
        if task == "B1_seasonality_present":
            return "nonseasonal"
        if task == "A2_trend_v":
            return 0.0
        return "none"

    slope = stats["lin"]["slope_mm_per_yr"]

    sea_amp = stats["season"]["amp_mm"]
    sea_snr = stats["season"]["snr"]
    sea_improve = stats["season"]["improve_frac"]

    if task == "A2_trend_v":
        return float(slope)

    if task == "A1_trend_dir":
        if abs(slope) < cfg.v_stable:
            return "stable"
        return "subsiding" if slope < 0 else "uplifting"

    if task == "B1_seasonality_present":
        present = (
            (sea_snr is not None and sea_snr >= cfg.b_snr_thr) and
            (sea_improve >= cfg.b_improve_thr) and
            (sea_amp >= cfg.b_amp_min)
        )
        return "seasonal" if present else "nonseasonal"

    if task == "B2_seasonality_amp":
        present = (
            (sea_snr is not None and sea_snr >= cfg.b_snr_thr) and
            (sea_improve >= cfg.b_improve_thr) and
            (sea_amp >= cfg.b_amp_min)
        )
        return float(sea_amp) if present else "none"

    # ---- Step answers per your spec ----
    if task in {"C1_step_time", "C2_step_mag"}:
        step = stats.get("step", {}) if isinstance(stats.get("step"), dict) else {}
        extra = step.get("extra", {}) if isinstance(step.get("extra"), dict) else {}
        present = bool(extra.get("present", False))
        if not present:
            return "none"
        if task == "C1_step_time":
            return step.get("time") or "none"
        else:
            return float(step["mag_mm"]) if step.get("mag_mm") is not None else "none"

    return "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--restrict_ids", default="", help="Optional prompts.jsonl to restrict ids.")
    ap.add_argument("--in_prompts", default="", help=argparse.SUPPRESS)

    ap.add_argument("--v_stable", type=float, default=0.5)
    ap.add_argument("--b_snr_thr", type=float, default=1.5)
    ap.add_argument("--b_improve_thr", type=float, default=0.05)
    ap.add_argument("--b_amp_min", type=float, default=0.5)

    ap.add_argument(
        "--step_detector_module",
        type=str,
        default="step_residual_detector_v2",
        help="Python module name under scripts/, e.g. step_residual_detector_v2",
    )

    # ✅ make step scan configurable
    ap.add_argument("--step_k_side", type=int, default=3, help="k_side passed into step detector")
    ap.add_argument("--step_topk", type=int, default=5, help="topk candidates returned by step detector")

    args = ap.parse_args()

    det_mod = importlib.import_module(args.step_detector_module)
    step_detector = det_mod.scan_step_on_residual

    cfg = BaselineConfig(
        v_stable=args.v_stable,
        b_snr_thr=args.b_snr_thr,
        b_improve_thr=args.b_improve_thr,
        b_amp_min=args.b_amp_min,
    )

    restrict_path = args.restrict_ids.strip() or args.in_prompts.strip()
    keep_ids = None
    if restrict_path:
        keep_ids = set(r["id"] for r in read_jsonl(restrict_path))

    cache: Dict[str, Dict[str, Any]] = {}
    out_rows = []

    k_side = int(max(1, args.step_k_side))
    topk = int(max(1, args.step_topk))

    for ex in tqdm(read_jsonl(args.dataset), desc="Baseline v3 (resid-step-detector)"):
        ex_id = ex.get("id")
        if keep_ids is not None and ex_id not in keep_ids:
            continue

        task = ex.get("task_name") or ex.get("task") or "unknown"
        ctx = ex.get("context") or {}
        series = (ctx.get("series") or {})
        static = (ctx.get("static") or {})
        rmse = static.get("rmse_mm", None)

        ck = series_cache_key(ex)

        # cache per series; if k_side/topk changed, recompute
        if ck not in cache:
            cache[ck] = compute_stats_from_series(
                series,
                cfg,
                rmse_mm=rmse,
                k_side=k_side,
                topk=topk,
                step_detector=step_detector,
            )
        else:
            old = cache[ck]
            old_fit = (((old.get("step") or {}).get("extra") or {}).get("residual_fit") or {})
            old_rmse = old_fit.get("rmse_mm_used")
            old_k = old_fit.get("k_side")
            old_topk = old_fit.get("topk")

            need_recompute = False
            if old_rmse is None and rmse is not None:
                need_recompute = True
            if old_k != k_side or old_topk != topk:
                need_recompute = True

            if need_recompute:
                cache[ck] = compute_stats_from_series(
                    series,
                    cfg,
                    rmse_mm=rmse,
                    k_side=k_side,
                    topk=topk,
                    step_detector=step_detector,
                )

        stats = cache[ck]
        ans = baseline_answer(task, stats, cfg)

        out_rows.append({
            "id": ex_id,
            "task_name": task,
            "model": "baseline_v3_resid_step",
            "answer_parsed": ans,
            "parse_error": None,
            "stats": stats,
        })

    write_jsonl(args.out, out_rows)
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
