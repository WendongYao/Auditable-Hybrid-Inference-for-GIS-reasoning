#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# scripts/merge_preds_taskwise_conditional.py
import argparse, json
from typing import Dict, Any, Tuple, Optional, Set


def _task(o: Dict[str, Any]) -> Optional[str]:
    return o.get("task_name") or o.get("task")


def _key(o: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    _id = o.get("id")
    tn = _task(o)
    if _id is None or tn is None:
        return None
    return (str(_id), str(tn))


def load_pred_map(path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    m: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            k = _key(o)
            if k is not None:
                m[k] = o
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_main", required=True)
    ap.add_argument("--pred_alt", required=True)
    ap.add_argument("--use_alt_tasks", required=True, help='comma separated task names')
    ap.add_argument("--out", required=True)
    ap.add_argument("--alt_flag_field", default="use_alt", help="bool field in alt to decide override")
    args = ap.parse_args()

    use_tasks: Set[str] = set([t.strip() for t in args.use_alt_tasks.split(",") if t.strip()])
    alt_map = load_pred_map(args.pred_alt)

    picked = 0
    total = 0

    with open(args.pred_main, "r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            total += 1
            o = json.loads(line)
            tn = _task(o)
            k = _key(o)

            if tn in use_tasks and k in alt_map:
                alt = alt_map[k]
                use_alt = bool(alt.get(args.alt_flag_field, False))
                if use_alt:
                    # overwrite standard answer fields; keep rest from main for stability
                    for fld in ["answer_parsed", "answer_numeric", "parse_error", "model"]:
                        if fld in alt:
                            o[fld] = alt[fld]
                    # also carry the flag for traceability
                    o["merged_from_alt"] = True
                    picked += 1

            w.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"[OK] wrote: {args.out}")
    print(f"lines={total}, picked_alt={picked}, use_alt_tasks={sorted(list(use_tasks))}, alt_flag_field={args.alt_flag_field}")


if __name__ == "__main__":
    main()
