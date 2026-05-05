import argparse, json, os
from typing import Any, Dict, List

TASK_B1 = "B1_seasonality_present"
TASK_B2 = "B2_seasonality_amp"
TASK_C1 = "C1_step_time"
TASK_C2 = "C2_step_mag"

def jread(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def jwrite(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def get_ap(row: Dict[str, Any]) -> Any:
    if "answer_parsed" in row:
        return row["answer_parsed"]
    return row.get("pred")

def set_none(row: Dict[str, Any]) -> None:
    if "answer_parsed" in row:
        row["answer_parsed"] = "none"
    if "pred" in row:
        row["pred"] = "none"
    if "answer_numeric" in row:
        row["answer_numeric"] = None

def base_key(_id: str, task: str) -> str:
    suf = "_" + task
    return _id[:-len(suf)] if _id.endswith(suf) else _id

def norm(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pred", required=True)
    ap.add_argument("--out_pred", required=True)
    ap.add_argument("--gate_b1b2", action="store_true")
    ap.add_argument("--gate_c1c2", action="store_true")
    args = ap.parse_args()

    rows = jread(args.in_pred)

    b1_map: Dict[str, str] = {}
    c1_map: Dict[str, str] = {}

    for r in rows:
        t = r.get("task_name") or r.get("task")
        _id = r.get("id")
        if not t or not _id:
            continue
        k = base_key(_id, t)
        v = norm(get_ap(r))
        if t == TASK_B1 and v:
            b1_map[k] = v
        if t == TASK_C1 and v:
            c1_map[k] = v

    n_b2 = 0
    n_c2 = 0
    for r in rows:
        t = r.get("task_name") or r.get("task")
        _id = r.get("id")
        if not t or not _id:
            continue
        k = base_key(_id, t)

        if args.gate_b1b2 and t == TASK_B2:
            b1 = b1_map.get(k, "")
            if b1 in ("nonseasonal", "none", "no", "false", "0"):
                set_none(r)
                r["gate_applied"] = "B1->B2"
                n_b2 += 1

        if args.gate_c1c2 and t == TASK_C2:
            c1 = c1_map.get(k, "")
            if c1 in ("none", "no_step", "nostep", "no", "false", "0"):
                set_none(r)
                r["gate_applied"] = "C1->C2"
                n_c2 += 1

    jwrite(args.out_pred, rows)
    print(f"[OK] wrote: {args.out_pred}")
    print(f"gate counts: B2_overridden={n_b2} C2_overridden={n_c2}")

if __name__ == "__main__":
    main()
