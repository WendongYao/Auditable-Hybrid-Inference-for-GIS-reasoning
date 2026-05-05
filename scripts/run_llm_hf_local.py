#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_llm import normalize_answer, parse_answer_json, validate_answer


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    out = set()
    for row in read_jsonl(path):
        ex_id = row.get("id")
        if ex_id is not None:
            out.add(str(ex_id))
    return out


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_model_and_tokenizer(
    model_id: str,
    load_in_4bit: bool,
    attn_impl: str,
    trust_remote_code: bool,
) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    if load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_cfg
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return tokenizer, model


def format_input_prompt(tokenizer: Any, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    return prompt


def decode_new_text(tokenizer: Any, outputs: torch.Tensor, input_lengths: List[int]) -> List[str]:
    texts: List[str] = []
    for i in range(outputs.shape[0]):
        gen_ids = outputs[i, input_lengths[i]:]
        texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return texts


def run_batch(
    tokenizer: Any,
    model: Any,
    rows: List[Dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_chat_template: bool,
) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [format_input_prompt(tokenizer, str(r["prompt"]), use_chat_template=use_chat_template) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_lengths = enc["attention_mask"].sum(dim=1).tolist()
        enc = {k: v.to(model.device) for k, v in enc.items()}
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(temperature > 0),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)
        with torch.no_grad():
            outputs = model.generate(**enc, **gen_kwargs)
        decoded = decode_new_text(tokenizer, outputs, input_lengths)
        for ex, text in zip(batch, decoded):
            ans, parse_err = parse_answer_json(text)
            valid, valid_reason = validate_answer(str(ex["task_name"]), ans)
            if ans is not None:
                ans = normalize_answer(ans)
                if isinstance(ans, str):
                    try:
                        if str(ex["task_name"]) in {"A2_trend_v", "B2_seasonality_amp", "C2_step_mag"} and ans != "none":
                            ans = float(ans)
                    except Exception:
                        pass
            if not valid:
                parse_err = parse_err or valid_reason
                ans = None

            out_rows.append(
                {
                    "id": ex["id"],
                    "task_name": ex["task_name"],
                    "model": "hf_local",
                    "answer_parsed": ans,
                    "parse_error": parse_err,
                    "valid_reason": valid_reason,
                    "raw_text": text,
                }
            )
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--attn_impl", default="sdpa")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rows = list(read_jsonl(args.in_prompts))
    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]

    done_ids: set[str] = set()
    if args.resume:
        done_ids = load_done_ids(args.out)
        if done_ids:
            rows = [r for r in rows if str(r["id"]) not in done_ids]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tokenizer, model = build_model_and_tokenizer(
        model_id=args.model_id,
        load_in_4bit=bool(args.load_in_4bit),
        attn_impl=args.attn_impl,
        trust_remote_code=bool(args.trust_remote_code),
    )

    total = len(rows)
    if total == 0:
        print("No rows to run.")
        return

    t0 = time.time()
    written = 0
    pbar = tqdm(total=total, desc=f"HF local {args.model_id}")
    for start in range(0, total, max(1, args.batch_size * 16)):
        block = rows[start:start + max(1, args.batch_size * 16)]
        out_rows = run_batch(
            tokenizer=tokenizer,
            model=model,
            rows=block,
            batch_size=int(args.batch_size),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            use_chat_template=bool(args.use_chat_template),
        )
        append_jsonl(args.out, out_rows)
        written += len(out_rows)
        pbar.update(len(out_rows))
    pbar.close()

    dt = time.time() - t0
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "n_written": written,
                "elapsed_sec": round(dt, 2),
                "examples_per_sec": round(written / max(dt, 1e-9), 4),
                "out": args.out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
