# scripts/merge_preds_taskwise.py
import argparse, json
from typing import Dict, Any, Iterable

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def get_id(r: Dict[str, Any]) -> str:
    return str(r.get("id") or r.get("rid") or "").strip()

def get_task(r: Dict[str, Any]) -> str:
    return str(r.get("task_name") or r.get("task") or "").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_main", required=True, help="non-routeA preds (main)")
    ap.add_argument("--pred_alt", required=True, help="routeA preds (alt)")
    ap.add_argument("--use_alt_tasks", default="C1_step_time", help="comma separated tasks to take from alt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    use_alt = {t.strip() for t in args.use_alt_tasks.split(",") if t.strip()}

    main_map: Dict[str, Dict[str, Any]] = {}
    for r in read_jsonl(args.pred_main):
        rid = get_id(r)
        if rid:
            main_map[rid] = r

    alt_map: Dict[str, Dict[str, Any]] = {}
    for r in read_jsonl(args.pred_alt):
        rid = get_id(r)
        if rid:
            alt_map[rid] = r

    # union ids, but keep output order deterministic: follow main first, then remaining from alt
    ordered_ids = list(main_map.keys()) + [rid for rid in alt_map.keys() if rid not in main_map]

    merged = 0
    picked_alt = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for rid in ordered_ids:
            r_main = main_map.get(rid)
            r_alt = alt_map.get(rid)

            chosen = r_main if r_main is not None else r_alt
            src = "main"

            if r_alt is not None:
                task = get_task(r_alt)
                if task in use_alt:
                    # if alt exists and task matches, prefer alt
                    chosen = r_alt
                    src = "alt"
                    picked_alt += 1

            if chosen is None:
                continue

            # add a light trace (safe)
            chosen = dict(chosen)
            chosen.setdefault("stats", {}).setdefault("step", {}).setdefault("extra", {})
            chosen["stats"]["step"]["extra"]["merge_taskwise"] = {"src": src}

            f.write(json.dumps(chosen, ensure_ascii=False) + "\n")
            merged += 1

    print(f"[OK] wrote: {args.out}")
    print(f"lines={merged}, picked_alt={picked_alt}, use_alt_tasks={sorted(use_alt)}")

if __name__ == "__main__":
    main()
