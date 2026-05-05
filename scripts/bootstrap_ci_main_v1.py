import argparse, csv, os, random
from collections import defaultdict
from typing import Dict, List, Tuple

TASKS = [
    "A1_trend_dir",
    "A2_trend_v",
    "B1_seasonality_present",
    "B2_seasonality_amp",
    "C1_step_time",
    "C2_step_mag",
]

def read_item_scores(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def index_by_task_and_id(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    # task -> id -> correct(0/1)
    out: Dict[str, Dict[str, int]] = defaultdict(dict)
    for r in rows:
        tid = r["id"]
        t = r["task_name"]
        c = int(float(r["correct"]))
        out[t][tid] = c
    return out

def intersect_ids(a: Dict[str, Dict[str, int]], b: Dict[str, Dict[str, int]]) -> Dict[str, List[str]]:
    common: Dict[str, List[str]] = {}
    for t in TASKS:
        ia = set(a.get(t, {}).keys())
        ib = set(b.get(t, {}).keys())
        common[t] = sorted(list(ia & ib))
    return common

def score_on_sample(
    corr: Dict[str, Dict[str, int]],
    sample_ids: Dict[str, List[str]],
) -> Tuple[float, float, Dict[str, float]]:
    # returns (macro, micro, per_task_acc)
    per_task = {}
    tot_correct = 0
    tot_n = 0
    macro_sum = 0.0
    macro_k = 0
    for t in TASKS:
        ids = sample_ids.get(t, [])
        if not ids:
            continue
        s = 0
        for _id in ids:
            s += corr[t].get(_id, 0)
        acc = s / len(ids)
        per_task[t] = acc
        tot_correct += s
        tot_n += len(ids)
        macro_sum += acc
        macro_k += 1
    macro = macro_sum / max(macro_k, 1)
    micro = tot_correct / max(tot_n, 1)
    return macro, micro, per_task

def stratified_bootstrap_ids(
    ids_by_task: Dict[str, List[str]],
    rng: random.Random,
) -> Dict[str, List[str]]:
    out = {}
    for t in TASKS:
        ids = ids_by_task.get(t, [])
        n = len(ids)
        if n == 0:
            out[t] = []
            continue
        out[t] = [ids[rng.randrange(n)] for _ in range(n)]
    return out

def pct(x: float) -> float:
    return 100.0 * x

def ci_from_samples(samples: List[float]) -> Tuple[float, float, float]:
    s = sorted(samples)
    n = len(s)
    mean = sum(s) / max(n, 1)
    lo = s[int(0.025 * (n - 1))]
    hi = s[int(0.975 * (n - 1))]
    return mean, lo, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_root", required=True)
    ap.add_argument("--tags", required=True, help="comma-separated")
    ap.add_argument("--llm_variant", default="ALL_base")
    ap.add_argument("--rules_variant", default="ABL_plus_B1B2C1C2")
    ap.add_argument("--full_variant", default="ABL_plus_B1B2C1C2_plus_gatev3")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    rng = random.Random(args.seed)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_tex) or ".", exist_ok=True)

    csv_rows = []
    tex_lines = []

    # LaTeX header (per-model deltas: Rules-LLM and Full-Rules)
    tex_lines.append(r"\begin{table}[t]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\small")
    tex_lines.append(r"\begin{tabular}{lcc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"Model & $\Delta$Macro (+Rules--LLM) [95\% CI] & $\Delta$Macro (Full--+Rules) [95\% CI] \\")
    tex_lines.append(r"\midrule")

    for tag in tags:
        p_llm = os.path.join(args.scores_root, tag, args.llm_variant, "item_scores.csv")
        p_rules = os.path.join(args.scores_root, tag, args.rules_variant, "item_scores.csv")
        p_full = os.path.join(args.scores_root, tag, args.full_variant, "item_scores.csv")

        if not (os.path.exists(p_llm) and os.path.exists(p_rules) and os.path.exists(p_full)):
            raise FileNotFoundError(f"missing item_scores.csv for tag={tag}. Check:\n{p_llm}\n{p_rules}\n{p_full}")

        llm = index_by_task_and_id(read_item_scores(p_llm))
        rules = index_by_task_and_id(read_item_scores(p_rules))
        full = index_by_task_and_id(read_item_scores(p_full))

        # ensure paired: intersect ids for each task
        common_lr = intersect_ids(llm, rules)
        common_rf = intersect_ids(rules, full)

        # use the strict intersection across all three for clean paired bootstrap
        common_all = {}
        for t in TASKS:
            s = set(common_lr[t]) & set(common_rf[t])
            common_all[t] = sorted(list(s))

        # point estimates on common set
        macro_llm, micro_llm, _ = score_on_sample(llm, common_all)
        macro_rules, micro_rules, _ = score_on_sample(rules, common_all)
        macro_full, micro_full, _ = score_on_sample(full, common_all)

        # bootstrap distributions (paired, stratified by task)
        dm_rules_macro = []
        dm_full_macro = []
        dm_rules_micro = []
        dm_full_micro = []

        for _ in range(args.B):
            samp = stratified_bootstrap_ids(common_all, rng)
            m1, u1, _ = score_on_sample(llm, samp)
            m2, u2, _ = score_on_sample(rules, samp)
            m3, u3, _ = score_on_sample(full, samp)

            dm_rules_macro.append(m2 - m1)
            dm_full_macro.append(m3 - m2)
            dm_rules_micro.append(u2 - u1)
            dm_full_micro.append(u3 - u2)

        # CI for deltas
        dmac_rules_mean, dmac_rules_lo, dmac_rules_hi = ci_from_samples(dm_rules_macro)
        dmac_full_mean, dmac_full_lo, dmac_full_hi = ci_from_samples(dm_full_macro)
        dmic_rules_mean, dmic_rules_lo, dmic_rules_hi = ci_from_samples(dm_rules_micro)
        dmic_full_mean, dmic_full_lo, dmic_full_hi = ci_from_samples(dm_full_micro)

        csv_rows.append({
            "tag": tag,
            "macro_llm": macro_llm,
            "macro_rules": macro_rules,
            "macro_full": macro_full,
            "micro_llm": micro_llm,
            "micro_rules": micro_rules,
            "micro_full": micro_full,
            "dmacro_rules_mean": dmac_rules_mean,
            "dmacro_rules_lo": dmac_rules_lo,
            "dmacro_rules_hi": dmac_rules_hi,
            "dmacro_full_mean": dmac_full_mean,
            "dmacro_full_lo": dmac_full_lo,
            "dmacro_full_hi": dmac_full_hi,
            "dmicro_rules_mean": dmic_rules_mean,
            "dmicro_rules_lo": dmic_rules_lo,
            "dmicro_rules_hi": dmic_rules_hi,
            "dmicro_full_mean": dmic_full_mean,
            "dmicro_full_lo": dmic_full_lo,
            "dmicro_full_hi": dmic_full_hi,
            "B": args.B,
            "seed": args.seed,
        })

        # LaTeX: show delta macro only (most readable)
        tex_lines.append(
            f"{tag} & "
            f"{pct(dmac_rules_mean):.2f} [{pct(dmac_rules_lo):.2f}, {pct(dmac_rules_hi):.2f}] & "
            f"{pct(dmac_full_mean):.2f} [{pct(dmac_full_lo):.2f}, {pct(dmac_full_hi):.2f}] \\\\"
        )

    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"\caption{Paired stratified bootstrap (by task) confidence intervals on dev\_balanced. Deltas are in percentage points.}")
    tex_lines.append(r"\label{tab:bootstrap_ci}")
    tex_lines.append(r"\end{table}")

    # write csv
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        w.writeheader()
        w.writerows(csv_rows)

    # write tex
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")

    print("[OK] wrote:", args.out_csv, "rows=", len(csv_rows))
    print("[OK] wrote:", args.out_tex)

if __name__ == "__main__":
    main()
