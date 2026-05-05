import argparse
import csv
import json
import os
import re
import math
import random
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple

TASKS = ["A1_trend_dir", "A2_trend_v", "B1_seasonality_present", "B2_seasonality_amp", "C1_step_time", "C2_step_mag"]

DISPLAY_MODEL = {
    "gemini25flash": "Gemini-2.5 Flash",
    "gemini_batch": "Gemini-2.5 Flash",
    "gemini3pro": "Gemini-3 Pro",
    "gemini3flash": "Gemini-3 Flash",
    "deepseekv32a3v1": "DeepSeek-V3.2",
    "gpt52": "ChatGPT-5.2",
    "llama4maverick": "Llama4-Maverick",
    "baseline_v3": "Baseline-v3",
}

ABL_VARIANTS_CORE = [
    "ALL_base",
    "ABL_plus_B1",
    "ABL_plus_B2",
    "ABL_plus_C1",
    "ABL_plus_C2",
    "ABL_plus_B1B2",
    "ABL_plus_C1C2",
    "ABL_plus_B1B2C1C2",
    "GATEv3_consistency_only",
    "GATEv3_after_C1C2predict",
    # 如果你之后补了 full gate，请用这个标准名
    "ABL_plus_B1B2C1C2_plus_gatev3",
]

# 论文方法名 -> 你当前 scores 目录名（严格对齐“新策略”）
METHODS = {
    # Main 3
    "LLM-only (routeA)": ["ALL_base"],
    "+Rules (B1B2C1C2; no gate)": ["ABL_plus_B1B2C1C2"],
    "+Rules+Gate (Full)": ["ABL_plus_B1B2C1C2_plus_gatev3"],  # 不再用旧目录 proxy

    # Component ablations (6)
    "+B1 rule only": ["ABL_plus_B1"],
    "+B2 rule only": ["ABL_plus_B2"],
    "+B1+B2 rules": ["ABL_plus_B1B2"],
    "+C1 rule only": ["ABL_plus_C1"],
    "+C2 rule only": ["ABL_plus_C2"],
    "+C1+C2 rules": ["ABL_plus_C1C2"],

    # Gate ablations (2)
    "LLM-only + gate_v3": ["GATEv3_consistency_only"],
    "+C1C2 rules + gate_v3": ["GATEv3_after_C1C2predict"],
}

PIPELINE = [
    ("LLM-only (routeA)", "ALL_base"),
    ("+ B1 merge", "ABL_plus_B1"),
    ("+ B2 merge", "ABL_plus_B1B2"),
    ("+ C1 merge", "ABL_plus_C1"),          # 近似展示；你若有 tmp-stage 可改
    ("+ C2 merge", "ABL_plus_B1B2C1C2"),
    ("+ gate_v3", "ABL_plus_B1B2C1C2_plus_gatev3"),
]

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_csv_dict(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def latex_escape(s: str) -> str:
    if s is None:
        return ""
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(s))

def fmt_pct(x: Optional[float], nd: int = 1) -> str:
    if x is None:
        return "--"
    return f"{x*100:.{nd}f}"

def fmt_delta(a: Optional[float], b: Optional[float], nd: int = 1) -> str:
    if a is None or b is None:
        return "--"
    return f"{(a-b)*100:+.{nd}f}"

def mean(xs: List[Optional[float]]) -> Optional[float]:
    ys = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(ys)/len(ys) if ys else None

def model_display(m: str) -> str:
    return DISPLAY_MODEL.get(m, m)

def discover_models(scores_root: str) -> List[str]:
    if not os.path.isdir(scores_root):
        return []
    out = []
    for d in os.listdir(scores_root):
        p = os.path.join(scores_root, d)
        if os.path.isdir(p):
            out.append(d)
    return sorted(out)

def has_summary(scores_root: str, model: str, variant: str) -> bool:
    return os.path.exists(os.path.join(scores_root, model, variant, "summary.json"))

def filter_only_ablation_models(scores_root: str, models: List[str]) -> List[str]:
    keep = []
    for m in models:
        if not has_summary(scores_root, m, "ALL_base"):
            continue
        ok = False
        for v in ["ABL_plus_B1", "ABL_plus_B2", "ABL_plus_C1", "ABL_plus_C2", "ABL_plus_B1B2", "ABL_plus_C1C2", "ABL_plus_B1B2C1C2"]:
            if has_summary(scores_root, m, v):
                ok = True
                break
        if ok:
            keep.append(m)
    return keep

def load_variant(scores_root: str, model: str, variant: str) -> Optional[Dict[str, Any]]:
    sd = os.path.join(scores_root, model, variant)
    sj = os.path.join(sd, "summary.json")
    if not os.path.exists(sj):
        return None
    s = read_json(sj)

    bt = {}
    by_task_path = os.path.join(sd, "by_task.csv")
    if os.path.exists(by_task_path):
        for row in read_csv_dict(by_task_path):
            tn = row.get("task_name")
            if tn and row.get("acc") not in (None, ""):
                bt[tn] = float(row["acc"])

    bb = defaultdict(dict)
    by_bucket_path = os.path.join(sd, "by_task_bucket.csv")
    if os.path.exists(by_bucket_path):
        for row in read_csv_dict(by_bucket_path):
            tn = row.get("task_name")
            bl = row.get("bucket_label")
            if tn and bl and row.get("acc") not in (None, ""):
                bb[tn][bl] = float(row["acc"])

    return {
        "model": model,
        "variant": variant,
        "dir": sd,
        "micro": float(s.get("micro_acc")) if s.get("micro_acc") is not None else None,
        "macro": float(s.get("macro_acc_over_tasks")) if s.get("macro_acc_over_tasks") is not None else None,
        "by_task": bt,
        "by_bucket": dict(bb),
    }

def pick_variant(scores_root: str, model: str, cands: List[str]) -> Optional[str]:
    for v in cands:
        if has_summary(scores_root, model, v):
            return v
    return None

def latex_table_begin(colspec: str) -> str:
    return "\\begin{tabular}{" + colspec + "}\n\\toprule\n"

def latex_table_end() -> str:
    return "\\bottomrule\n\\end{tabular}\n"

def write_table(tex: List[str], caption: str, label: str, tabular: str, note: Optional[str] = None):
    if note:
        tex.append(f"\\paragraph{{}} {note}\n\n")
    tex.append("\\begin{table}[t]\n\\centering\n\\small\n")
    tex.append(tabular)
    tex.append(f"\\caption{{{caption}}}\n")
    tex.append(f"\\label{{{label}}}\n")
    tex.append("\\end{table}\n\n")

def build_task_spec_table() -> str:
    colspec = "lllp{5.2cm}"
    lines = [latex_table_begin(colspec)]
    lines.append("Task & Output & Type & Scoring / Notes \\\\\n\\midrule\n")
    lines.append("A1\\_trend\\_dir & \\{\\texttt{subsiding,uplifting,stable}\\} & Cat. & Exact match. \\\\\n")
    lines.append("A2\\_trend\\_v & number (mm/yr) & Num. & $|\\hat{v}-v|\\le \\tau_v$ (abs tol). \\\\\n")
    lines.append("B1\\_seasonality\\_present & \\{\\texttt{seasonal,nonseasonal}\\} & Cat. & Exact match. \\\\\n")
    lines.append("B2\\_seasonality\\_amp & number (mm) or \\texttt{none} & Num. & If seasonal: $|\\hat{a}-a|\\le \\tau_a$; else \\texttt{none}. \\\\\n")
    lines.append("C1\\_step\\_time & \\texttt{yyyymmdd} or \\texttt{none} & Struct. & Exact match; \\texttt{none} allowed. \\\\\n")
    lines.append("C2\\_step\\_mag & number (mm) or \\texttt{none} & Num. & If step: $|\\hat{m}-m|\\le \\tau_m$; else \\texttt{none}. \\\\\n")
    lines.append(latex_table_end())
    return "".join(lines)

def build_dataset_stats_stub() -> str:
    colspec = "lrrrr"
    lines = [latex_table_begin(colspec)]
    lines.append("Tile & \\#Points & \\#Epochs & Time span & Balanced eval (per trend class) \\\\\n\\midrule\n")
    for tile in ["E32N34", "E32N35", "E40N30", "E44N43", "E48N24"]:
        lines.append(f"{tile} & -- & -- & 2019--2023 & --/--/-- \\\\\n")
    lines.append(latex_table_end())
    return "".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_root", required=True)
    ap.add_argument("--out_tex", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--models", default="", help="comma-separated allowlist; empty=auto-discover")
    ap.add_argument("--only_abl_models", action="store_true", help="keep only models with ALL_base and some ABL_plus_*")
    args = ap.parse_args()

    scores_root = args.scores_root
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models.strip() else discover_models(scores_root)
    if args.only_abl_models:
        models = filter_only_ablation_models(scores_root, models)

    if not models:
        raise SystemExit("No models selected. Check --scores_root / --models / --only_abl_models")

    # Build pick + load
    pick = {}  # (model, method) -> variant
    data = {}  # (model, variant) -> record

    for m in models:
        for method, cands in METHODS.items():
            v = pick_variant(scores_root, m, cands)
            if v:
                pick[(m, method)] = v
                rec = load_variant(scores_root, m, v)
                if rec:
                    data[(m, v)] = rec

    # Optional flat CSV (only selected models, only known variants)
    if args.out_csv:
        rows = []
        for (m, v), rec in data.items():
            bt = rec.get("by_task", {})
            rows.append({
                "model": m,
                "variant": v,
                "micro_acc": rec.get("micro"),
                "macro_acc": rec.get("macro"),
                "A1": bt.get("A1_trend_dir"),
                "A2": bt.get("A2_trend_v"),
                "B1": bt.get("B1_seasonality_present"),
                "B2": bt.get("B2_seasonality_amp"),
                "C1": bt.get("C1_step_time"),
                "C2": bt.get("C2_step_mag"),
            })
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            w.writeheader()
            w.writerows(rows)

    tex = []
    tex.append("% Auto-generated by scripts/make_acl_tables_from_scores_v3.py\n\n")
    tex.append("\\documentclass{article}\n")
    tex.append("\\usepackage{booktabs}\n")
    tex.append("\\usepackage[margin=1in]{geometry}\n")
    tex.append("\\begin{document}\n\n")
    tex.append("\\newcommand{\\egmsqa}{EGMS-QA}\n\n")

    tex.append("\\section*{ACL-friendly evaluation suite (ablation-only models)}\n\n")
    tex.append("This file is generated from \\texttt{scores\\_root} and includes only models that have the ablation suite "
               "(\\texttt{ALL\\_base} + at least one \\texttt{ABL\\_plus\\_*}).\n\n")

    write_table(
        tex,
        caption="Task specification of \\egmsqa{}. Tolerances follow dataset-defined evaluation rules.",
        label="tab:task_spec",
        tabular=build_task_spec_table(),
        note="Table 1 documents output types and scoring rules for reproducibility."
    )
    write_table(
        tex,
        caption="Dataset statistics template (fill tile-level counts as needed).",
        label="tab:data_stats",
        tabular=build_dataset_stats_stub(),
        note="Table 2 is a stub you can fill if you decide to report tile-level stats."
    )

    # Table 3: LLM-only per-task
    colspec = "lcccccc|cc"
    lines = [latex_table_begin(colspec)]
    lines.append("Model & A1 & A2 & B1 & B2 & C1 & C2 & Macro & Micro \\\\\n\\midrule\n")
    for m in models:
        v = pick.get((m, "LLM-only (routeA)"))
        rec = data.get((m, v)) if v else None
        bt = rec["by_task"] if rec else {}
        lines.append(
            f"{latex_escape(model_display(m))} & "
            f"{fmt_pct(bt.get('A1_trend_dir'))} & {fmt_pct(bt.get('A2_trend_v'))} & "
            f"{fmt_pct(bt.get('B1_seasonality_present'))} & {fmt_pct(bt.get('B2_seasonality_amp'))} & "
            f"{fmt_pct(bt.get('C1_step_time'))} & {fmt_pct(bt.get('C2_step_mag'))} & "
            f"{fmt_pct(rec.get('macro') if rec else None)} & {fmt_pct(rec.get('micro') if rec else None)} \\\\\n"
        )
    lines.append(latex_table_end())
    write_table(
        tex,
        caption="LLM-only results with routeA prompting. Per-task accuracy highlights systematic brittleness.",
        label="tab:llm_only_routeA",
        tabular="".join(lines),
        note="Table 3 is your failure diagnosis table."
    )

    # Table 4: Main (LLM vs +Rules vs +Rules+Gate)
    colspec = "lccc|ccc"
    lines = [latex_table_begin(colspec)]
    lines.append("& \\multicolumn{3}{c|}{Macro} & \\multicolumn{3}{c}{Micro} \\\\\n")
    lines.append("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n")
    lines.append("Model & LLM & +Rules & +Rules+Gate & LLM & +Rules & +Rules+Gate \\\\\n\\midrule\n")
    for m in models:
        def get(method):
            v = pick.get((m, method))
            return data.get((m, v)) if v else None
        r_llm = get("LLM-only (routeA)")
        r_rules = get("+Rules (B1B2C1C2; no gate)")
        r_full = get("+Rules+Gate (Full)")
        lines.append(
            f"{latex_escape(model_display(m))} & "
            f"{fmt_pct(r_llm.get('macro') if r_llm else None)} & {fmt_pct(r_rules.get('macro') if r_rules else None)} & {fmt_pct(r_full.get('macro') if r_full else None)} & "
            f"{fmt_pct(r_llm.get('micro') if r_llm else None)} & {fmt_pct(r_rules.get('micro') if r_rules else None)} & {fmt_pct(r_full.get('micro') if r_full else None)} \\\\\n"
        )
    lines.append(latex_table_end())
    write_table(
        tex,
        caption="Main results across backbones. +Rules applies sequential merges for B1/B2/C1/C2. +Rules+Gate is shown only if you scored the dedicated Full-gate variant.",
        label="tab:main_across_models",
        tabular="".join(lines),
        note="Table 4 is the headline cross-model result table."
    )

    # Table 6: Component ablation (avg over selected models)
    rows = [
        ("LLM-only (routeA)", "LLM-only (routeA)"),
        ("+ B1 rule", "+B1 rule only"),
        ("+ B2 rule", "+B2 rule only"),
        ("+ B1 + B2 rules", "+B1+B2 rules"),
        ("+ C1 rule", "+C1 rule only"),
        ("+ C2 rule", "+C2 rule only"),
        ("+ C1 + C2 rules", "+C1+C2 rules"),
        ("+ Rules (B1B2C1C2; no gate)", "+Rules (B1B2C1C2; no gate)"),
        ("LLM-only + gate_v3", "LLM-only + gate_v3"),
        ("+ C1C2 rules + gate_v3", "+C1C2 rules + gate_v3"),
        ("+ Rules + gate_v3 (Full)", "+Rules+Gate (Full)"),
    ]
    colspec = "lcc|cccc"
    lines = [latex_table_begin(colspec)]
    lines.append("Variant & Macro & Micro & B1 & B2 & C1 & C2 \\\\\n\\midrule\n")
    for label, method in rows:
        macs=[]; mics=[]; b1s=[]; b2s=[]; c1s=[]; c2s=[]
        for m in models:
            v = pick.get((m, method))
            if not v:
                continue
            rec = data.get((m, v))
            if not rec:
                continue
            macs.append(rec.get("macro")); mics.append(rec.get("micro"))
            bt = rec["by_task"]
            b1s.append(bt.get("B1_seasonality_present"))
            b2s.append(bt.get("B2_seasonality_amp"))
            c1s.append(bt.get("C1_step_time"))
            c2s.append(bt.get("C2_step_mag"))
        lines.append(
            f"{latex_escape(label)} & {fmt_pct(mean(macs))} & {fmt_pct(mean(mics))} & "
            f"{fmt_pct(mean(b1s))} & {fmt_pct(mean(b2s))} & {fmt_pct(mean(c1s))} & {fmt_pct(mean(c2s))} \\\\\n"
        )
    lines.append(latex_table_end())
    write_table(
        tex,
        caption="Component ablations averaged over selected models. Each module only updates its target tasks. Gate variants are kept for completeness.",
        label="tab:abl_components",
        tabular="".join(lines),
        note="Table 6 is the attribution table reviewers care about."
    )

    # Table 7: Sequential pipeline (representative model = first in list)
    rep = models[0]
    colspec = "lcc"
    lines = [latex_table_begin(colspec)]
    lines.append("Pipeline stage & Macro & Micro \\\\\n\\midrule\n")
    for stage_name, variant in PIPELINE:
        rec = load_variant(scores_root, rep, variant)
        lines.append(f"{latex_escape(stage_name)} & {fmt_pct(rec.get('macro') if rec else None)} & {fmt_pct(rec.get('micro') if rec else None)} \\\\\n")
    lines.append(latex_table_end())
    write_table(
        tex,
        caption=f"Sequential pipeline (representative: {latex_escape(model_display(rep))}). Cells are '--' if a stage was not scored.",
        label="tab:sequential_pipeline",
        tabular="".join(lines),
        note="Table 7 mirrors your real serial merge pipeline."
    )

    tex.append("\\end{document}\n")

    os.makedirs(os.path.dirname(args.out_tex) or ".", exist_ok=True)
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write("".join(tex))

    print(f"[OK] wrote TeX: {args.out_tex}")
    if args.out_csv:
        print(f"[OK] wrote CSV: {args.out_csv}")
    print("Selected models:", ", ".join(models))

if __name__ == "__main__":
    main()
