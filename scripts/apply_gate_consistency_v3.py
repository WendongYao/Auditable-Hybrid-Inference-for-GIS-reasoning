#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_gate_consistency_v3.py

v3 对齐结论：禁用 conflict-based override_diff 替换裁决，改为无害的 format_fix（只做等价归一化）。

变化要点（相对 v1）：
- ✅ 保留：force_none / override / conflict_force_none（可选）
- ❌ 删除语义：override_diff（不再用 disagreement 触发“用 baseline 覆盖”）
- ✅ 新增：format_fix（仅在双方可 parse 且差异在“等价范围”内，规范化 LLM 输出格式；不做裁决替换）

format_fix 规则（默认极保守）：
- C1_step_time：两边都能 parse 成日期，且 abs(diff_days) <= 1 cadence（默认 12 天）
  => 将 LLM 日期规范为 yyyymmdd（若原本已经是 yyyymmdd，通常不会触发）
- C2_step_mag：两边都能 parse 成 mm 浮点，且 abs_diff <= 0.1mm（默认）
  => 将 LLM 数值规范为 float(四舍五入到若干小数位)，不改变数值含义

D3 变化（最小 patch）：
- ✅ C2 base_none 情况新增联合证据 guard：当 baseline 最终预测为 none，但 baseline best step 满足
  (snr + improve + bic + mag) 联合阈值时，允许 keep_llm；否则仍然 force_none（保险丝）。

D4 变化（本次 patch）：
- ✅ C1 base_none 情况新增联合证据 guard：当 baseline 最终预测为 none，但 baseline best step 满足
  (snr + improve + bic + mag[可设很低]) 联合阈值时，允许 keep_llm；否则仍然 force_none（保险丝）。
  - 保留旧的 C1 “very-strong guard”（strong_step_for_base_none_guard），仅在未开启 c1_base_none_guard 时生效。
- ✅ 新增：--c1_guard_max_days，并在 C1 guard 内要求 LLM 日期与 baseline best date 的差值 <= max_days。

D5 变化（本次 patch）：
- ✅ 当 LLM=none 且 baseline_pred=none，但 baseline best evidence 很强时，允许用 baseline_best 做 impute（fill-missing）。
  - Patch 1：impute_best 仅在 base_pred==none 时触发（避免抢跑覆盖 override）。
  - Patch 2：C2 impute 不再用 mag_abs_mm；优先用带符号 mag_mm，否则不 impute（避免永远正号）。

Patch3（推荐）：
- ✅ persist_flag 显式 False 必须尊重（不要被 persist_frac 翻盘）
  - 新增 as_bool_strict()
  - extract_best_step() 中 pflag 推导逻辑按 diff 修改

兼容性：
- 保留旧参数 --override_on_diff / diff_tol_* / diff_requires_strong_base，但仅打印 warning，不再执行 replacement。

=========================
本次按你给的“修改方案”追加的改动（C2 override / impute_best 取值）：
- ✅ C2 override：不再直接用 baseline_pred（或 answer_parsed），优先使用 candidates_topk 中
  max |raw_adj_delta_mm|（且需高置信阈值通过）；否则回退到旧逻辑（base_pred）。
- ✅ C2 impute_best：同样优先 max |raw_adj_delta_mm|（高置信）；否则回退到旧逻辑（best.mag_abs_mm）。
- ✅ 不改命令行参数；阈值内部取 max(step_thr, guard_thr, 常量更严阈值)。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple


STEP_TASKS = {"C1_step_time", "C2_step_mag"}


# -------------------------
# IO helpers
# -------------------------
def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# -------------------------
# id / task / pred helpers
# -------------------------
_ID_KEYS = (
    "id",
    "qid",
    "_id",
    "example_id",
    "prompt_id",
    "uid",
    "sample_id",
    "query_id",
    "record_id",
)


def get_id(rec: Dict[str, Any]) -> str:
    for k in _ID_KEYS:
        v = rec.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in {"none", "null", "nan"}:
            return s

    meta = rec.get("meta") or rec.get("metadata") or {}
    if isinstance(meta, dict):
        for k in _ID_KEYS:
            v = meta.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s and s.lower() not in {"none", "null", "nan"}:
                return s

    for k, v in rec.items():
        if v is None:
            continue
        if "id" in str(k).lower():
            s = str(v).strip()
            if s and s.lower() not in {"none", "null", "nan"}:
                return s

    return ""


def get_task(rec: Dict[str, Any]) -> str:
    return str(rec.get("task") or rec.get("task_name") or rec.get("taskName") or "")


def get_pred_value(rec: Dict[str, Any]) -> Any:
    # answer_parsed > answer > pred
    if "answer_parsed" in rec:
        return rec.get("answer_parsed")
    if "answer" in rec:
        return rec.get("answer")
    return rec.get("pred")


def set_pred_fields(rec: Dict[str, Any], new_answer_parsed: Any, *, also_clear_cache: bool = True) -> None:
    """
    让 gate 的改动真正影响评测：
    - 强制写入 answer_parsed
    - 同步 pred / answer
    - None -> "none"
    """
    if new_answer_parsed is None:
        new_answer_parsed = "none"

    rec["answer_parsed"] = new_answer_parsed
    rec["pred"] = new_answer_parsed
    rec["answer"] = str(new_answer_parsed)

    if also_clear_cache:
        for k in ["answer_text", "pred_text", "parsed", "parse_error"]:
            rec.pop(k, None)


def is_none_pred(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        return s in {"none", "null", "nan", ""}
    return False


def as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            if isinstance(x, float) and not math.isfinite(x):
                return None
            return float(x)
        s = str(x).strip()
        if not s:
            return None
        if s.lower() in {"none", "null", "nan"}:
            return None
        v = float(s)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


# PATCH3: strict bool parser
def as_bool_strict(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


# -------------------------
# NEW (per your patch): C2 max-jump helper
# -------------------------
def pick_c2_maxjump_abs_delta_mm(
    cands: Any,
    *,
    mag_thr: float,
    snr_thr: float,
    imp_thr: float,
    persist_thr: float,
    bic_thr: float = -5.0,
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """
    For C2: pick max |raw_adj_delta_mm| among candidates_topk, but only if high-confidence.
    This matches gold≈abs(delta_mm) in the prompt route.

    Returns: (abs_delta_mm, cand_dict) or (None, None)
    """
    if not isinstance(cands, list) or not cands:
        return None, None

    best_abs: Optional[float] = None
    best_cand: Optional[Dict[str, Any]] = None

    for c in cands:
        if not isinstance(c, dict):
            continue

        d = as_float(c.get("raw_adj_delta_mm"))
        if d is None:
            continue
        absd = abs(float(d))

        snr = as_float(c.get("snr")) or 0.0
        imp = as_float(c.get("improve_frac")) or 0.0
        persist = as_float(c.get("persist_frac")) or 0.0
        dbic = as_float(c.get("delta_bic"))

        # High-confidence filter
        if absd < float(mag_thr):
            continue
        if float(snr) < float(snr_thr):
            continue
        if float(imp) < float(imp_thr):
            continue
        if float(persist) < float(persist_thr):
            continue

        # delta_bic: more negative is better; if missing, don't block
        if (dbic is not None) and (float(dbic) > float(bic_thr)):
            continue

        if (best_abs is None) or (absd > best_abs):
            best_abs = absd
            best_cand = c

    return best_abs, best_cand


# -------------------------
# robust parsers for format_fix
# -------------------------
_DATE_PATTERNS = ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d")


def parse_date_flexible(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    s = str(x).strip()
    if not s:
        return None
    # drop common wrappers/spaces
    s = s.replace(".", "-").replace("_", "-").strip()
    for pat in _DATE_PATTERNS:
        try:
            return datetime.strptime(s, pat)
        except Exception:
            pass
    return None


def canonical_yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def parse_mm_flexible(x: Any) -> Optional[float]:
    """
    允许输入含单位的字符串（极保守，只去掉常见尾缀）
    """
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        v = float(x)
        return v if math.isfinite(v) else None

    s = str(x).strip().lower()
    if not s or s in {"none", "null", "nan"}:
        return None

    # remove trailing units (very conservative)
    for suf in ["mm", "毫米"]:
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
            break

    # remove commas
    s = s.replace(",", "")
    return as_float(s)


def days_between(a: datetime, b: datetime) -> int:
    return abs((a - b).days)


# -------------------------
# baseline best extraction
# -------------------------
@dataclass
class BestStep:
    mag_abs_mm: float = 0.0
    mag_mm: Optional[float] = None  # NEW: signed
    snr: float = 0.0
    improve_frac: float = 0.0
    delta_bic: float = 0.0
    persist_frac: float = 0.0
    persist_flag: bool = False
    date: Optional[str] = None
    # NEW: carry candidates_topk for C2 max-jump
    candidates_topk: List[Dict[str, Any]] = field(default_factory=list)


def _to_float(x: Any, default: float = 0.0) -> float:
    v = as_float(x)
    return default if v is None else float(v)


def extract_best_step(baseline_rec: Dict[str, Any]) -> Optional[BestStep]:
    stats = baseline_rec.get("stats") or {}
    step = stats.get("step") or {}
    extra = step.get("extra") or {}
    best = extra.get("best") or {}
    if not isinstance(best, dict) or not best:
        return None

    # PATCH: collect candidates_topk (for C2 max-jump)
    cands = extra.get("candidates_topk") or best.get("candidates_topk") or []
    if not isinstance(cands, list):
        cands = []
    cands = [c for c in cands if isinstance(c, dict)]

    # Patch 2: avoid mixing signed into abs
    mag_abs = _to_float(best.get("mag_abs_mm"), 0.0)
    mag_signed = as_float(best.get("mag_mm"))
    # NOTE: if you later confirm best["mag"] is signed, you can enable:
    # if mag_signed is None:
    #     mag_signed = as_float(best.get("mag"))

    snr = _to_float(best.get("snr"), 0.0)
    imp = _to_float(best.get("improve_frac", best.get("improve", best.get("improve_ratio"))), 0.0)
    bic = _to_float(best.get("delta_bic", best.get("bic_delta", best.get("bic"))), 0.0)
    pfrac = _to_float(best.get("persist_frac", best.get("persist", best.get("persist_score"))), 0.0)

    # PATCH3: respect explicit persist_flag=False; only derive from pfrac when persist_flag missing/unparseable
    pf_raw = best.get("persist_flag", None)
    pf = as_bool_strict(pf_raw)
    if pf is None and ("persist_flag" not in best):
        pflag = (pfrac >= 0.5)
    elif pf is None:
        pflag = bool(pf_raw)  # last resort
    else:
        pflag = pf

    date = best.get("date") or best.get("step_date") or best.get("step_date_candidate") or best.get("time") or None
    if date is not None:
        date = str(date)

    return BestStep(
        mag_abs_mm=mag_abs,
        mag_mm=mag_signed,
        snr=snr,
        improve_frac=imp,
        delta_bic=bic,
        persist_frac=pfrac,
        persist_flag=pflag,
        date=date,
        candidates_topk=cands,
    )


# -------------------------
# gate logic
# -------------------------
def strong_no_step(
    b: BestStep,
    veto_mag_thr: float,
    veto_snr_thr: float,
    veto_imp_thr: float,
    veto_persist_frac: float,
    veto_use_bic: bool,
    veto_bic_thr: float,
) -> bool:
    """
    旧版的问题：要求 (not persist_flag) 才可能判定 no-step，导致 detector 一旦 persist_flag=True，
    regular veto 永不触发（你实验A changed=0 的根因）。

    新版思路（保守但能触发）：
    - 以 “mag_abs 很小” 作为 no-step 的主信号（更符合直觉）
    - mag 小时：再需要 1 个弱证据（snr/imp/persist/bic）即可 veto
    - mag 不小：需要更多弱证据（更保守，避免误杀）
    """
    weak_mag = (b.mag_abs_mm < veto_mag_thr)
    weak_snr = (b.snr < veto_snr_thr)
    weak_imp = (b.improve_frac < veto_imp_thr)

    # persist：只有当 persist_frac 低 且 persist_flag 也没站住，才算弱
    weak_persist = (b.persist_frac < veto_persist_frac) and (not b.persist_flag)

    weak_votes = int(weak_mag) + int(weak_snr) + int(weak_imp) + int(weak_persist)

    if veto_use_bic:
        weak_bic = (b.delta_bic > veto_bic_thr)
        weak_votes += int(weak_bic)

    # mag 已经很小 -> 只要再有 1 个弱信号就足够强 no-step
    if weak_mag:
        need = 2 if not veto_use_bic else 3
        return weak_votes >= need

    # mag 不小 -> 更保守：需要更多弱信号才 veto
    need = 3 if not veto_use_bic else 4
    return weak_votes >= need


def strong_step(
    b: BestStep,
    step_mag_thr: float,
    step_snr_thr: float,
    step_imp_thr: float,
    force_persist_frac: float,
    step_bic_thr: float,
) -> bool:
    ok = (
        (b.mag_abs_mm >= step_mag_thr)
        and (b.snr >= step_snr_thr)
        and (b.improve_frac >= step_imp_thr)
        and (b.persist_frac >= force_persist_frac)
        and b.persist_flag
    )
    if step_bic_thr == 0.0:
        ok = ok and (b.delta_bic <= 0.0)
    else:
        ok = ok and (b.delta_bic <= step_bic_thr)
    return ok


def strong_step_for_base_none_guard(
    b: BestStep,
    step_mag_thr: float,
    step_snr_thr: float,
    step_imp_thr: float,
    step_bic_thr: float,
) -> bool:
    ok = (
        (b.mag_abs_mm >= step_mag_thr)
        and (b.snr >= step_snr_thr)
        and (b.improve_frac >= step_imp_thr)
    )
    if step_bic_thr == 0.0:
        ok = ok and (b.delta_bic <= 0.0)
    else:
        ok = ok and (b.delta_bic <= step_bic_thr)
    return ok


def decide_gate(
    task: str,
    llm_pred: Any,
    base_pred: Any,
    base_best: Optional[BestStep],
    *,
    # veto_on_base_none
    veto_on_base_none: bool,
    # regular veto
    veto_mag_thr: float,
    veto_snr_thr: float,
    veto_imp_thr: float,
    veto_persist_frac: float,
    veto_use_bic: bool,
    veto_bic_thr: float,
    # override
    enable_override_c1: bool,
    enable_override_c2: bool,
    step_mag_thr: float,
    step_snr_thr: float,
    step_imp_thr: float,
    force_persist_frac: float,
    step_bic_thr: float,
    # conflict
    conflict_days: int,
    # format_fix (new)
    enable_format_fix: bool,
    format_fix_c1_cadence_days: int,
    format_fix_c2_tol_mm: float,
    format_fix_c2_decimals: int,
    # C2 base_none joint-evidence guard (D3)
    c2_base_none_guard: bool,
    c2_guard_mag_thr: float,
    c2_guard_snr_thr: float,
    c2_guard_imp_thr: float,
    c2_guard_bic_thr: float,
    # impute best when llm is none (only meaningful when base_pred is none)
    enable_impute_best_c1: bool,
    enable_impute_best_c2: bool,
    # C1 base-none joint-evidence guard (NEW)
    c1_base_none_guard: bool,
    c1_guard_mag_thr: float,
    c1_guard_snr_thr: float,
    c1_guard_imp_thr: float,
    c1_guard_bic_thr: float,
    c1_guard_max_days: int,
) -> Tuple[str, Any, Dict[str, Any]]:
    dbg: Dict[str, Any] = {}

    llm_none = is_none_pred(llm_pred)
    base_none = is_none_pred(base_pred)
    dbg["llm_none"] = llm_none
    dbg["base_none"] = base_none
    dbg["base_pred"] = base_pred

    if base_best is None:
        dbg["baseline_best"] = None
    else:
        dbg["baseline_best"] = {
            "mag_abs_mm": base_best.mag_abs_mm,
            "mag_mm": base_best.mag_mm,
            "snr": base_best.snr,
            "improve_frac": base_best.improve_frac,
            "delta_bic": base_best.delta_bic,
            "persist_frac": base_best.persist_frac,
            "persist_flag": base_best.persist_flag,
            "date": base_best.date,
            "candidates_topk_n": len(base_best.candidates_topk),
        }

    # ------------------------------------------------------------
    # NEW: impute baseline best when LLM predicts none
    # Patch 1: only allow when base_pred is none (avoid抢跑覆盖override)
    # Patch 2 (replaced by your patch): for C2, prefer max-jump(|raw_adj_delta_mm|) if high-conf,
    #                                  else fallback to best.mag_abs_mm
    # ------------------------------------------------------------
    if llm_none and base_none and (base_best is not None):
        is_strong = strong_step(
            base_best,
            step_mag_thr=step_mag_thr,
            step_snr_thr=step_snr_thr,
            step_imp_thr=step_imp_thr,
            force_persist_frac=force_persist_frac,
            step_bic_thr=step_bic_thr,
        )
        dbg["strong_step_for_impute_best"] = is_strong
        if is_strong:
            if task == "C1_step_time" and enable_impute_best_c1:
                dt = parse_date_flexible(base_best.date)
                if dt is not None:
                    return "impute_best", canonical_yyyymmdd(dt), dbg

            if task == "C2_step_mag" and enable_impute_best_c2:
                # High-confidence thresholds (stricter than defaults)
                mag_thr = max(float(step_mag_thr), float(c2_guard_mag_thr), 2.5)
                snr_thr = max(float(step_snr_thr), float(c2_guard_snr_thr), 2.5)
                imp_thr = max(float(step_imp_thr), float(c2_guard_imp_thr), 0.04)
                persist_thr = max(float(force_persist_frac), 0.8)

                v_abs, cand = pick_c2_maxjump_abs_delta_mm(
                    base_best.candidates_topk,
                    mag_thr=mag_thr,
                    snr_thr=snr_thr,
                    imp_thr=imp_thr,
                    persist_thr=persist_thr,
                    bic_thr=-5.0,
                )
                dbg["impute_c2_maxjump_thr"] = {
                    "mag_thr": mag_thr,
                    "snr_thr": snr_thr,
                    "imp_thr": imp_thr,
                    "persist_thr": persist_thr,
                    "bic_thr": -5.0,
                }
                dbg["impute_c2_maxjump_abs_delta_mm"] = v_abs
                if isinstance(cand, dict):
                    dbg["impute_c2_maxjump_cand"] = {
                        "raw_adj_delta_mm": cand.get("raw_adj_delta_mm"),
                        "snr": cand.get("snr"),
                        "improve_frac": cand.get("improve_frac"),
                        "persist_frac": cand.get("persist_frac"),
                        "delta_bic": cand.get("delta_bic"),
                    }

                if v_abs is None:
                    # fallback: old behavior (best.mag_abs_mm)
                    v_abs = base_best.mag_abs_mm
                    dbg["impute_c2_fallback"] = "best.mag_abs_mm"

                if v_abs is None:
                    dbg["impute_c2_skip_reason"] = "no_value_after_fallback"
                else:
                    try:
                        return "impute_best", round(float(v_abs), 1), dbg
                    except Exception:
                        return "impute_best", v_abs, dbg

    # conflict only for C1: both non-none date but disagree too much => force none (more conservative)
    if task == "C1_step_time" and (not llm_none) and (not base_none) and conflict_days > 0:
        d1 = parse_date_flexible(llm_pred)
        d2 = parse_date_flexible(base_pred)
        dd = days_between(d1, d2) if (d1 and d2) else None
        dbg["conflict_days"] = dd
        if dd is not None and dd > conflict_days:
            return "conflict_force_none", None, dbg

    # veto_on_base_none: baseline 最终预测 none，LLM 非 none => force_none（但 C2/C1 可用联合证据放行）
    if veto_on_base_none and (not llm_none) and base_none:
        if task == "C2_step_mag":
            # D3: joint-evidence guard (snr + imp + bic + mag) to allow keep_llm
            if c2_base_none_guard and (base_best is not None):
                ok = (
                    (base_best.snr >= float(c2_guard_snr_thr))
                    and (base_best.improve_frac >= float(c2_guard_imp_thr))
                    and (base_best.delta_bic <= float(c2_guard_bic_thr))
                    and (base_best.mag_abs_mm >= float(c2_guard_mag_thr))
                )
                dbg["c2_base_none_guard_enabled"] = True
                dbg["c2_guard_ok"] = ok
                dbg["c2_guard_thr"] = {
                    "mag_abs_mm": float(c2_guard_mag_thr),
                    "snr": float(c2_guard_snr_thr),
                    "improve_frac": float(c2_guard_imp_thr),
                    "delta_bic": float(c2_guard_bic_thr),
                }
                if ok:
                    dbg["veto_base_none_skipped_due_to_c2_guard"] = True
                    return "keep_llm", llm_pred, dbg

            # default behavior: conservative fuse (force none)
            dbg["force_none_base_none"] = True
            dbg["veto_base_none_no_guard_for_c2"] = True
            return "force_none", None, dbg

        # C1: joint-evidence guard + max_days constraint vs baseline best date
        if task == "C1_step_time" and c1_base_none_guard and (base_best is not None):
            ok = (
                (base_best.snr >= float(c1_guard_snr_thr))
                and (base_best.improve_frac >= float(c1_guard_imp_thr))
                and (base_best.delta_bic <= float(c1_guard_bic_thr))
                and (base_best.mag_abs_mm >= float(c1_guard_mag_thr))
            )
            dbg["c1_base_none_guard_enabled"] = True
            dbg["c1_guard_ok"] = ok
            dbg["c1_guard_thr"] = {
                "mag_abs_mm": float(c1_guard_mag_thr),
                "snr": float(c1_guard_snr_thr),
                "improve_frac": float(c1_guard_imp_thr),
                "delta_bic": float(c1_guard_bic_thr),
                "max_days": int(c1_guard_max_days),
            }
            if ok:
                llm_dt = parse_date_flexible(llm_pred)
                best_dt = parse_date_flexible(base_best.date)
                dd = days_between(llm_dt, best_dt) if (llm_dt and best_dt) else None
                dbg["c1_guard_date_diff_days"] = dd
                if (dd is not None) and (dd <= int(c1_guard_max_days)):
                    dbg["veto_base_none_skipped_due_to_c1_guard"] = True
                    return "keep_llm", llm_pred, dbg

        # (optional) keep old very-strong guard if you still want it when c1_guard not enabled
        if (
            (task == "C1_step_time")
            and (not c1_base_none_guard)
            and (base_best is not None)
            and strong_step_for_base_none_guard(
                base_best,
                step_mag_thr=step_mag_thr,
                step_snr_thr=step_snr_thr,
                step_imp_thr=step_imp_thr,
                step_bic_thr=step_bic_thr,
            )
        ):
            dbg["veto_base_none_skipped_due_to_strong_step"] = True
            dbg["veto_base_none_guard_used"] = "no_persist"
            return "keep_llm", llm_pred, dbg

        dbg["force_none_base_none"] = True
        return "force_none", None, dbg

    # regular veto: LLM 非 none 且 baseline best 强 no-step => force none
    if (not llm_none) and (base_best is not None):
        if strong_no_step(
            base_best,
            veto_mag_thr=veto_mag_thr,
            veto_snr_thr=veto_snr_thr,
            veto_imp_thr=veto_imp_thr,
            veto_persist_frac=veto_persist_frac,
            veto_use_bic=veto_use_bic,
            veto_bic_thr=veto_bic_thr,
        ):
            return "force_none", None, dbg

    # override: LLM 为 none，baseline 强 step 且 baseline 最终预测也非 none => 用 base_pred 覆盖
    # PATCH (per your request): C2 override prefers max-jump(|raw_adj_delta_mm|) if high-conf; else fallback to base_pred
    if llm_none and (not base_none) and (base_best is not None):
        is_strong = strong_step(
            base_best,
            step_mag_thr=step_mag_thr,
            step_snr_thr=step_snr_thr,
            step_imp_thr=step_imp_thr,
            force_persist_frac=force_persist_frac,
            step_bic_thr=step_bic_thr,
        )
        dbg["strong_step_for_override"] = is_strong
        if is_strong:
            if task == "C1_step_time" and enable_override_c1:
                return "override", base_pred, dbg

            if task == "C2_step_mag" and enable_override_c2:
                mag_thr = max(float(step_mag_thr), float(c2_guard_mag_thr), 2.5)
                snr_thr = max(float(step_snr_thr), float(c2_guard_snr_thr), 2.5)
                imp_thr = max(float(step_imp_thr), float(c2_guard_imp_thr), 0.04)
                persist_thr = max(float(force_persist_frac), 0.8)

                v_abs, cand = pick_c2_maxjump_abs_delta_mm(
                    base_best.candidates_topk,
                    mag_thr=mag_thr,
                    snr_thr=snr_thr,
                    imp_thr=imp_thr,
                    persist_thr=persist_thr,
                    bic_thr=-5.0,
                )
                dbg["override_c2_maxjump_thr"] = {
                    "mag_thr": mag_thr,
                    "snr_thr": snr_thr,
                    "imp_thr": imp_thr,
                    "persist_thr": persist_thr,
                    "bic_thr": -5.0,
                }
                dbg["override_c2_maxjump_abs_delta_mm"] = v_abs
                if isinstance(cand, dict):
                    dbg["override_c2_maxjump_cand"] = {
                        "raw_adj_delta_mm": cand.get("raw_adj_delta_mm"),
                        "snr": cand.get("snr"),
                        "improve_frac": cand.get("improve_frac"),
                        "persist_frac": cand.get("persist_frac"),
                        "delta_bic": cand.get("delta_bic"),
                    }

                if v_abs is None:
                    # fallback: old behavior (base_pred)
                    dbg["override_c2_fallback"] = "base_pred"
                    return "override", base_pred, dbg

                try:
                    return "override", round(float(v_abs), 1), dbg
                except Exception:
                    return "override", v_abs, dbg

    # -------------------------
    # format_fix (no arbitration, only normalization)
    # -------------------------
    if enable_format_fix and (not llm_none) and (not base_none):
        if task == "C1_step_time":
            d1 = parse_date_flexible(llm_pred)
            d2 = parse_date_flexible(base_pred)
            if d1 and d2:
                dd = days_between(d1, d2)
                dbg["format_fix_days"] = dd
                if dd <= int(format_fix_c1_cadence_days):
                    canon = canonical_yyyymmdd(d1)
                    if str(llm_pred).strip() != canon:
                        return "format_fix", canon, dbg

        elif task == "C2_step_mag":
            a = parse_mm_flexible(llm_pred)
            b = parse_mm_flexible(base_pred)
            if (a is not None) and (b is not None):
                abs_diff = abs(a - b)
                dbg["format_fix_abs_diff_mm"] = abs_diff
                if abs_diff <= float(format_fix_c2_tol_mm):
                    canon = round(float(a), int(format_fix_c2_decimals))
                    # 只有当“明显是格式差异/类型差异”时才改写，避免无意义 changed
                    if not (isinstance(llm_pred, (int, float)) and float(llm_pred) == float(canon)):
                        return "format_fix", canon, dbg

    return "keep_llm", llm_pred, dbg


# -------------------------
# restrict_ids loader
# -------------------------
def _load_ids_from_json_obj(obj: Any) -> List[str]:
    ids: List[str] = []
    if obj is None:
        return ids

    if isinstance(obj, list):
        for x in obj:
            if x is None:
                continue
            s = str(x).strip()
            if s and s.lower() not in {"none", "null", "nan"}:
                ids.append(s)
        return ids

    if isinstance(obj, dict):
        for k in ("ids", "restrict_ids", "id_list", "qid_list"):
            if k in obj:
                ids.extend(_load_ids_from_json_obj(obj.get(k)))
        rid = get_id(obj)
        if rid:
            ids.append(rid)
        return ids

    s = str(obj).strip()
    if s and s.lower() not in {"none", "null", "nan"}:
        ids.append(s)
    return ids


def load_restrict_set(arg: Optional[str]) -> Optional[set]:
    if not arg:
        return None
    s = str(arg).strip()
    if not s:
        return None

    if os.path.exists(s) and os.path.isfile(s):
        ext = os.path.splitext(s)[1].lower()

        if ext == ".jsonl":
            ids = set()
            for r in read_jsonl(s):
                rid = get_id(r)
                if rid:
                    ids.add(rid)
            return ids if ids else None

        if ext in {".txt", ".list"}:
            ids = set()
            with open(s, "r", encoding="utf-8") as f:
                for line in f:
                    t = line.strip()
                    if t:
                        ids.add(t)
            return ids if ids else None

        if ext == ".json":
            with open(s, "r", encoding="utf-8") as f:
                obj = json.load(f)
            ids = {x for x in _load_ids_from_json_obj(obj) if x}
            return ids if ids else None

        # unknown ext: try jsonl else lines
        try:
            ids = set()
            for r in read_jsonl(s):
                rid = get_id(r)
                if rid:
                    ids.add(rid)
            if ids:
                return ids
        except Exception:
            pass

        ids = set()
        with open(s, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    ids.add(t)
        return ids if ids else None

    ids = {x.strip() for x in s.split(",") if x.strip()}
    return ids if ids else None


# -------------------------
# args
# -------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--llm_path", required=True)
    p.add_argument("--baseline_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run_tag", default="gate_v3")

    # strong_step thresholds (override / base-none guard / impute_best)
    p.add_argument("--step_mag_thr", type=float, default=2.0)
    p.add_argument("--step_snr_thr", type=float, default=1.5)
    p.add_argument("--step_imp_thr", type=float, default=0.02)
    p.add_argument("--step_bic_thr", type=float, default=0.0)
    p.add_argument("--force_persist_frac", type=float, default=0.7)

    # veto thresholds (strong_no_step)
    p.add_argument("--none_margin_frac", type=float, default=0.7)
    p.add_argument("--veto_mag_thr", type=float, default=None)
    p.add_argument("--veto_snr_thr", type=float, default=None)
    p.add_argument("--veto_imp_thr", type=float, default=None)
    p.add_argument("--veto_persist_frac", type=float, default=0.7)
    p.add_argument("--veto_use_bic", action="store_true")
    p.add_argument("--veto_bic_thr", type=float, default=0.0)

    # base-none veto (with C1 guard)
    p.add_argument("--veto_on_base_none", action="store_true")

    # override (fill-missing)
    p.add_argument("--enable_override_c1", action="store_true")
    p.add_argument("--enable_override_c2", action="store_true")

    # impute best when llm is none (only meaningful when base_pred is none)
    p.add_argument("--enable_impute_best_c1", action="store_true")
    p.add_argument("--enable_impute_best_c2", action="store_true")

    # conflict (C1)
    p.add_argument("--conflict_days", type=int, default=0)

    # format_fix
    p.add_argument("--enable_format_fix", action="store_true")
    p.add_argument("--format_fix_c1_cadence_days", type=int, default=12)
    p.add_argument("--format_fix_c2_tol_mm", type=float, default=0.1)
    p.add_argument("--format_fix_c2_decimals", type=int, default=3)

    # D3: C2 base-none joint-evidence guard
    p.add_argument(
        "--c2_base_none_guard",
        action="store_true",
        help=(
            "C2 only: when baseline pred is none, allow keep_llm if baseline best satisfies "
            "joint evidence (snr+imp+bic+mag)."
        ),
    )
    p.add_argument("--c2_guard_mag_thr", type=float, default=0.0)
    p.add_argument("--c2_guard_snr_thr", type=float, default=0.0)
    p.add_argument("--c2_guard_imp_thr", type=float, default=0.0)
    p.add_argument("--c2_guard_bic_thr", type=float, default=0.0)

    # C1 base-none joint-evidence guard
    p.add_argument("--c1_base_none_guard", action="store_true")
    p.add_argument("--c1_guard_mag_thr", type=float, default=0.0)
    p.add_argument("--c1_guard_snr_thr", type=float, default=0.0)
    p.add_argument("--c1_guard_imp_thr", type=float, default=0.0)
    p.add_argument("--c1_guard_bic_thr", type=float, default=0.0)
    p.add_argument("--c1_guard_max_days", type=int, default=24)

    # deprecated (kept only to avoid breaking old command lines)
    p.add_argument("--override_on_diff", action="store_true", help="DEPRECATED in v3 (ignored)")
    p.add_argument("--diff_tol_days", type=int, default=0, help="DEPRECATED (ignored)")
    p.add_argument("--diff_tol_mm", type=float, default=0.0, help="DEPRECATED (ignored)")
    p.add_argument("--diff_tol_rel", type=float, default=0.0, help="DEPRECATED (ignored)")
    p.add_argument("--diff_requires_strong_base", action="store_true", help="DEPRECATED (ignored)")

    p.add_argument("--restrict_ids", type=str, default=None)
    return p


def _sample_ids(keys: List[str], k: int = 5) -> List[str]:
    out: List[str] = []
    for x in keys:
        if x and x not in out:
            out.append(x)
        if len(out) >= k:
            break
    return out


def main() -> None:
    args = build_argparser().parse_args()

    if args.override_on_diff or args.diff_tol_days or args.diff_tol_mm or args.diff_tol_rel or args.diff_requires_strong_base:
        print("[gate_v3][warning] override_diff is DEPRECATED and IGNORED in v3. Use --enable_format_fix if needed.")

    veto_mag_thr = args.veto_mag_thr if args.veto_mag_thr is not None else (args.step_mag_thr * args.none_margin_frac)
    veto_snr_thr = args.veto_snr_thr if args.veto_snr_thr is not None else (args.step_snr_thr * args.none_margin_frac)
    veto_imp_thr = args.veto_imp_thr if args.veto_imp_thr is not None else (args.step_imp_thr * args.none_margin_frac)

    restrict = load_restrict_set(args.restrict_ids)

    llm_map: Dict[str, Dict[str, Any]] = {}
    for r in read_jsonl(args.llm_path):
        rid = get_id(r)
        if rid:
            llm_map[rid] = r

    base_map: Dict[str, Dict[str, Any]] = {}
    for r in read_jsonl(args.baseline_path):
        rid = get_id(r)
        if rid:
            base_map[rid] = r

    common_ids = sorted(set(llm_map.keys()) & set(base_map.keys()))
    if restrict is not None:
        common_ids = [i for i in common_ids if i in restrict]

    print(
        f"[gate_v3] llm={len(llm_map)} base={len(base_map)} "
        f"restrict={len(restrict) if restrict else 'None'} common={len(common_ids)}"
    )

    if len(common_ids) == 0:
        print("[gate_v3][debug] sample llm ids:", _sample_ids(sorted(llm_map.keys()), 8))
        print("[gate_v3][debug] sample base ids:", _sample_ids(sorted(base_map.keys()), 8))
        print("[gate_v3][debug] sample restrict ids:", _sample_ids(sorted(list(restrict)) if restrict else [], 8))

    common = 0
    changed = 0
    force_none = 0
    force_none_base_none = 0
    override = 0
    conflict_force_none = 0
    format_fix = 0
    impute_best = 0
    skipped_nonstep = 0

    # --- diagnostics for why changed=0 ---
    veto_hits = 0
    veto_hits_by_task = {"C1_step_time": 0, "C2_step_mag": 0}
    veto_base_none_hits = 0
    override_hits = 0
    format_fix_hits = 0
    conflict_hits = 0
    impute_best_hits = 0

    with open(args.out, "w", encoding="utf-8") as wf:
        for rid in common_ids:
            common += 1
            llm = llm_map[rid]
            base = base_map[rid]
            task = get_task(llm)

            out = copy.deepcopy(llm)
            out.setdefault("stats", {})
            out["stats"].setdefault("step", {})
            out["stats"]["step"].setdefault("extra", {})
            out["stats"]["step"]["extra"].setdefault("gate_v1", {})  # keep key for compatibility

            gate_info: Dict[str, Any] = {"run_tag": args.run_tag}

            if task not in STEP_TASKS:
                skipped_nonstep += 1
                out["stats"]["step"]["extra"]["gate_v1"].update(gate_info)
                wf.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue

            llm_pred = get_pred_value(out)
            base_pred = get_pred_value(base)
            base_best = extract_best_step(base)

            action, new_value, dbg = decide_gate(
                task=task,
                llm_pred=llm_pred,
                base_pred=base_pred,
                base_best=base_best,
                veto_on_base_none=bool(args.veto_on_base_none),
                veto_mag_thr=float(veto_mag_thr),
                veto_snr_thr=float(veto_snr_thr),
                veto_imp_thr=float(veto_imp_thr),
                veto_persist_frac=float(args.veto_persist_frac),
                veto_use_bic=bool(args.veto_use_bic),
                veto_bic_thr=float(args.veto_bic_thr),
                enable_override_c1=bool(args.enable_override_c1),
                enable_override_c2=bool(args.enable_override_c2),
                step_mag_thr=float(args.step_mag_thr),
                step_snr_thr=float(args.step_snr_thr),
                step_imp_thr=float(args.step_imp_thr),
                force_persist_frac=float(args.force_persist_frac),
                step_bic_thr=float(args.step_bic_thr),
                conflict_days=int(args.conflict_days),
                enable_format_fix=bool(args.enable_format_fix),
                format_fix_c1_cadence_days=int(args.format_fix_c1_cadence_days),
                format_fix_c2_tol_mm=float(args.format_fix_c2_tol_mm),
                format_fix_c2_decimals=int(args.format_fix_c2_decimals),
                c2_base_none_guard=bool(args.c2_base_none_guard),
                c2_guard_mag_thr=float(args.c2_guard_mag_thr),
                c2_guard_snr_thr=float(args.c2_guard_snr_thr),
                c2_guard_imp_thr=float(args.c2_guard_imp_thr),
                c2_guard_bic_thr=float(args.c2_guard_bic_thr),
                enable_impute_best_c1=bool(args.enable_impute_best_c1),
                enable_impute_best_c2=bool(args.enable_impute_best_c2),
                c1_base_none_guard=bool(args.c1_base_none_guard),
                c1_guard_mag_thr=float(args.c1_guard_mag_thr),
                c1_guard_snr_thr=float(args.c1_guard_snr_thr),
                c1_guard_imp_thr=float(args.c1_guard_imp_thr),
                c1_guard_bic_thr=float(args.c1_guard_bic_thr),
                c1_guard_max_days=int(args.c1_guard_max_days),
            )

            if action == "force_none":
                veto_hits += 1
                if task in veto_hits_by_task:
                    veto_hits_by_task[task] += 1
                if isinstance(dbg, dict) and dbg.get("force_none_base_none"):
                    veto_base_none_hits += 1
            elif action == "override":
                override_hits += 1
            elif action == "format_fix":
                format_fix_hits += 1
            elif action == "conflict_force_none":
                conflict_hits += 1
            elif action == "impute_best":
                impute_best_hits += 1

            gate_info["action"] = action
            gate_info["debug"] = dbg
            gate_info["thr"] = {
                "veto_on_base_none": bool(args.veto_on_base_none),
                "veto_mag_thr": float(veto_mag_thr),
                "veto_snr_thr": float(veto_snr_thr),
                "veto_imp_thr": float(veto_imp_thr),
                "veto_persist_frac": float(args.veto_persist_frac),
                "veto_use_bic": bool(args.veto_use_bic),
                "veto_bic_thr": float(args.veto_bic_thr),
                "step_mag_thr": float(args.step_mag_thr),
                "step_snr_thr": float(args.step_snr_thr),
                "step_imp_thr": float(args.step_imp_thr),
                "step_bic_thr": float(args.step_bic_thr),
                "force_persist_frac": float(args.force_persist_frac),
                "enable_override_c1": bool(args.enable_override_c1),
                "enable_override_c2": bool(args.enable_override_c2),
                "enable_impute_best_c1": bool(args.enable_impute_best_c1),
                "enable_impute_best_c2": bool(args.enable_impute_best_c2),
                "conflict_days": int(args.conflict_days),
                "enable_format_fix": bool(args.enable_format_fix),
                "format_fix_c1_cadence_days": int(args.format_fix_c1_cadence_days),
                "format_fix_c2_tol_mm": float(args.format_fix_c2_tol_mm),
                "format_fix_c2_decimals": int(args.format_fix_c2_decimals),
                "c2_base_none_guard": bool(args.c2_base_none_guard),
                "c2_guard_mag_thr": float(args.c2_guard_mag_thr),
                "c2_guard_snr_thr": float(args.c2_guard_snr_thr),
                "c2_guard_imp_thr": float(args.c2_guard_imp_thr),
                "c2_guard_bic_thr": float(args.c2_guard_bic_thr),
                "c1_base_none_guard": bool(args.c1_base_none_guard),
                "c1_guard_mag_thr": float(args.c1_guard_mag_thr),
                "c1_guard_snr_thr": float(args.c1_guard_snr_thr),
                "c1_guard_imp_thr": float(args.c1_guard_imp_thr),
                "c1_guard_bic_thr": float(args.c1_guard_bic_thr),
                "c1_guard_max_days": int(args.c1_guard_max_days),
            }

            if action != "keep_llm":
                old_effective = get_pred_value(out)
                set_pred_fields(out, new_value, also_clear_cache=True)
                new_effective = get_pred_value(out)
                if old_effective != new_effective:
                    changed += 1

                if action == "force_none":
                    force_none += 1
                    if isinstance(dbg, dict) and dbg.get("force_none_base_none"):
                        force_none_base_none += 1
                elif action == "override":
                    override += 1
                elif action == "conflict_force_none":
                    conflict_force_none += 1
                elif action == "format_fix":
                    format_fix += 1
                elif action == "impute_best":
                    impute_best += 1

            out["stats"]["step"]["extra"]["gate_v1"].update(gate_info)
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(
        f"common: {common}\n"
        f"changed_answers: {changed}\n"
        f"force_none: {force_none}\n"
        f"  force_none_base_none: {force_none_base_none}\n"
        f"override: {override}\n"
        f"impute_best: {impute_best}\n"
        f"format_fix: {format_fix}\n"
        f"conflict_force_none: {conflict_force_none}\n"
        f"skipped_nonstep: {skipped_nonstep}\n"
        f"veto_thr: mag<{veto_mag_thr} snr<{veto_snr_thr} imp<{veto_imp_thr} "
        f"persist<{args.veto_persist_frac} bic={'on' if args.veto_use_bic else 'off'}\n"
        + f"diag: veto_hits={veto_hits} (base_none={veto_base_none_hits}) "
          f"override_hits={override_hits} impute_best_hits={impute_best_hits} "
          f"format_fix_hits={format_fix_hits} conflict_hits={conflict_hits}\n"
        + f"diag_by_task: veto_hits_C1={veto_hits_by_task.get('C1_step_time',0)} "
          f"veto_hits_C2={veto_hits_by_task.get('C2_step_mag',0)}\n"
    )


if __name__ == "__main__":
    main()
