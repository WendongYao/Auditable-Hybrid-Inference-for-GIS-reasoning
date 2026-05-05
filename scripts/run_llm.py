# scripts/run_llm.py
import argparse, json, os, time, re, threading
from typing import Any, Dict, Optional, Tuple, List
import requests
from tqdm import tqdm

try:
    import apikey  # type: ignore
except Exception:
    apikey = None

DEFAULT_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Vertex OpenAI-compatible endpoint:
# https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{LOCATION}/endpoints/openapi/chat/completions
# (docs show the OpenAI-compatible endpoint path and method) :contentReference[oaicite:4]{index=4}
DEFAULT_VERTEX_LOCATION = "us-east5"
DEFAULT_VERTEX_MODEL = "meta/llama-4-scout-17b-16e-instruct-maas"  # commonly used; other docs list without "meta/" too.

JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
DATE_YYYYMMDD_RE = re.compile(r"^\d{8}$")


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# Common parsing / validation
# -------------------------
def parse_answer_json(text: str) -> Tuple[Optional[Any], Optional[str]]:
    if not text:
        return None, "empty_response"
    s = text.strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "answer" in obj:
            return obj["answer"], None
        return None, "json_missing_answer_key"
    except Exception:
        pass

    first = s.find("{")
    if first != -1:
        try:
            dec = json.JSONDecoder()
            obj, _ = dec.raw_decode(s[first:])
            if isinstance(obj, dict) and "answer" in obj:
                return obj["answer"], None
            return None, "salvaged_json_missing_answer_key"
        except Exception:
            pass

    m = JSON_OBJ_RE.search(s)
    if not m:
        return None, "no_json_object_found"
    snippet = m.group(0)
    try:
        obj = json.loads(snippet)
        if isinstance(obj, dict) and "answer" in obj:
            return obj["answer"], None
        return None, "salvaged_json_missing_answer_key"
    except Exception as e:
        return None, f"json_parse_error:{type(e).__name__}"


def normalize_answer(ans: Any) -> Any:
    if isinstance(ans, str):
        return ans.strip()
    return ans


def validate_answer(task_name: str, ans: Any) -> Tuple[bool, str]:
    ans = normalize_answer(ans)

    if task_name == "A1_trend_dir":
        if isinstance(ans, str) and ans in {"subsiding", "uplifting", "stable"}:
            return True, "ok"
        return False, "invalid_label_expected_subsiding_uplifting_stable"

    if task_name == "B1_seasonality_present":
        if isinstance(ans, str) and ans in {"seasonal", "nonseasonal"}:
            return True, "ok"
        return False, "invalid_label_expected_seasonal_nonseasonal"

    if task_name == "C1_step_time":
        if isinstance(ans, str):
            if ans == "none":
                return True, "ok"
            if DATE_YYYYMMDD_RE.match(ans):
                return True, "ok"
        return False, "invalid_time_expected_yyyymmdd_or_none"

    if task_name in {"A2_trend_v"}:
        if isinstance(ans, (int, float)):
            return True, "ok"
        if isinstance(ans, str):
            try:
                float(ans)
                return True, "ok"
            except Exception:
                pass
        return False, "invalid_number_expected"

    if task_name in {"B2_seasonality_amp", "C2_step_mag"}:
        if isinstance(ans, str) and ans == "none":
            return True, "ok"
        if isinstance(ans, (int, float)):
            return True, "ok"
        if isinstance(ans, str):
            try:
                float(ans)
                return True, "ok"
            except Exception:
                pass
        return False, "invalid_number_or_none_expected"

    return True, "ok"


def build_repair_suffix(task_name: str) -> str:
    if task_name == "A1_trend_dir":
        allowed = 'One of: "subsiding", "uplifting", "stable".'
    elif task_name == "B1_seasonality_present":
        allowed = 'One of: "seasonal", "nonseasonal".'
    elif task_name == "C1_step_time":
        allowed = 'Either "none" or an 8-digit date "yyyymmdd".'
    elif task_name == "A2_trend_v":
        allowed = "A single JSON number (mm/year)."
    elif task_name in {"B2_seasonality_amp", "C2_step_mag"}:
        allowed = 'Either "none" or a JSON number (mm).'
    else:
        allowed = "Return JSON only with key 'answer'."

    return (
        "\n\nIMPORTANT (format): Output MUST be exactly one JSON object and nothing else.\n"
        f"Allowed answer format for this task: {allowed}\n"
        "Output NOW, no explanation.\n"
    )


# -------------------------
# Gemini backend (unchanged)
# -------------------------
def _gemini_finish_reason(resp_json: Dict[str, Any]) -> Optional[str]:
    cands = resp_json.get("candidates") or []
    if not cands:
        return None
    return cands[0].get("finishReason")


def _gemini_extract_text(resp_json: Dict[str, Any]) -> str:
    cands = resp_json.get("candidates") or []
    if not cands:
        return ""
    content = (cands[0].get("content") or {})
    parts = content.get("parts") or []
    texts = []
    for p in parts:
        if isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
            texts.append(p["text"])
    return "".join(texts).strip()


def call_gemini_generate(
    api_key: str,
    model: str,
    prompt: str,
    endpoint: str,
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    thinking_budget: Optional[int] = None,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    url = f"{endpoint}/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    generation_cfg: Dict[str, Any] = {
        "temperature": temperature,
        "topP": top_p,
        "maxOutputTokens": int(max_output_tokens),
        "responseMimeType": "application/json",
    }
    # Thinking config (Gemini 2.5)
    if thinking_budget is not None:
        tb = int(thinking_budget)
        # Gemini 2.5 Pro 不能关thinking；最小128
        if "gemini-2.5-pro" in model and tb == 0:
            tb = 128
        generation_cfg["thinkingConfig"] = {"thinkingBudget": tb}
    else:
        # 没显式指定时：默认沿用你之前的策略——flash禁用thinking以提速
        if model.startswith("gemini-2.5-flash"):
            generation_cfg["thinkingConfig"] = {"thinkingBudget": 256}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_cfg,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text}
    return {"status_code": r.status_code, "body": body}


# -------------------------
# Vertex (OpenAI-compatible) backend for Llama 4 MaaS
# -------------------------
class VertexTokenManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._credentials = None
        self._expiry_ts = 0.0

    def _refresh_locked(self) -> str:
        from google.auth import default
        from google.auth.transport.requests import Request

        # cloud-platform scope is standard for Vertex calls
        creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(Request())
        token = getattr(creds, "token", None)
        expiry = getattr(creds, "expiry", None)
        if not token:
            raise RuntimeError("Failed to obtain Google access token from ADC.")
        self._credentials = creds
        self._expiry_ts = time.time() + 3300  # ~55min fallback if expiry not available
        if expiry is not None:
            try:
                self._expiry_ts = float(expiry.timestamp())
            except Exception:
                pass
        return token

    def get_token(self) -> str:
        with self._lock:
            # refresh if expiring in < 5 minutes
            if time.time() >= (self._expiry_ts - 300):
                return self._refresh_locked()
            token = getattr(self._credentials, "token", None) if self._credentials else None
            if not token:
                return self._refresh_locked()
            return token


def call_vertex_openai_chat(
    token_mgr: VertexTokenManager,
    project_id: str,
    location: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
) -> Dict[str, Any]:
    # OpenAI-compatible endpoint path in Vertex docs :contentReference[oaicite:5]{index=5}
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions"
    )

    token = token_mgr.get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        # response_format is supported by the OpenAI-compatible API (model support may vary) :contentReference[oaicite:6]{index=6}
        "response_format": {"type": "json_object"},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text}
    return {"status_code": r.status_code, "body": body}


def _openai_finish_reason(resp_json: Dict[str, Any]) -> Optional[str]:
    choices = resp_json.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason") or choices[0].get("finishReason")


def _openai_extract_text(resp_json: Dict[str, Any]) -> str:
    choices = resp_json.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0].get("message") or {})
    content = msg.get("content")
    return (content or "").strip() if isinstance(content, str) else ""


# -------------------------
# Main runner
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_prompts", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--backend", choices=["gemini", "vertex_llama4"], default="vertex_llama4")

    # gemini
    ap.add_argument("--gemini_model", default=DEFAULT_GEMINI_MODEL)
    ap.add_argument("--gemini_endpoint", default=DEFAULT_GEMINI_ENDPOINT)

    # vertex llama4
    ap.add_argument("--project_id", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip())
    ap.add_argument("--location", default=DEFAULT_VERTEX_LOCATION)
    ap.add_argument("--vertex_model", default=DEFAULT_VERTEX_MODEL)

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--timeout_s", type=int, default=60)

    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--retry_backoff", type=float, default=1.7)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_output_tokens", type=int, default=128)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--auto_expand_tokens", action="store_true")
    ap.add_argument("--max_output_tokens_cap", type=int, default=512)
    ap.add_argument("--expand_factor", type=float, default=2.0)

    ap.add_argument("--repair_on_failure", action="store_true")
    ap.add_argument("--debug_dir", default="", help="If set, dumps per-item response JSONs for debugging.")
    ap.add_argument("--thinking_budget", type=int, default=None,
                    help="Gemini thinkingBudget. flash可设0关闭；2.5 pro最小128。")

    args = ap.parse_args()

    # Keys / creds
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key and apikey is not None and getattr(apikey, "gemini_api_key", "").strip():
        gemini_key = apikey.gemini_api_key.strip()

    if args.backend == "vertex_llama4":
        if not args.project_id:
            raise RuntimeError("Vertex backend requires --project_id (or env GOOGLE_CLOUD_PROJECT).")
        # Llama 4 MaaS is available in us-east5 per model docs :contentReference[oaicite:7]{index=7}
        token_mgr = VertexTokenManager()
    else:
        token_mgr = None

    rows = list(read_jsonl(args.in_prompts))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    out_rows: List[Dict[str, Any]] = []

    for ex in tqdm(rows, desc=f"Running ({args.backend})"):
        ex_id = ex.get("id")
        task_name = ex.get("task_name") or ex.get("task") or "unknown"
        base_prompt = ex.get("prompt", "")

        model_name = args.gemini_model if args.backend == "gemini" else args.vertex_model
        rec: Dict[str, Any] = {"id": ex_id, "task_name": task_name, "model": model_name}

        attempt = 0
        cur_max_tokens = int(args.max_output_tokens)
        used_repair = False

        while True:
            attempt += 1
            prompt = base_prompt + (build_repair_suffix(task_name) if (used_repair and args.repair_on_failure) else "")

            t0 = time.time()

            if args.backend == "gemini":
                if not gemini_key:
                    raise RuntimeError("Gemini backend requires GEMINI_API_KEY env var (or apikey.gemini_api_key).")
                result = call_gemini_generate(
                    api_key=gemini_key,
                    model=args.model,
                    prompt=prompt,
                    endpoint=args.endpoint,
                    temperature=args.temperature,
                    max_output_tokens=cur_max_tokens,
                    top_p=args.top_p,
                    thinking_budget=args.thinking_budget,
                )

                dt = time.time() - t0
                status = result["status_code"]
                body = result["body"]
                finish = _gemini_finish_reason(body) if isinstance(body, dict) else None
                text = _gemini_extract_text(body) if isinstance(body, dict) else ""

            else:
                result = call_vertex_openai_chat(
                    token_mgr=token_mgr,  # type: ignore
                    project_id=args.project_id,
                    location=args.location,
                    model=args.vertex_model,
                    prompt=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=cur_max_tokens,
                    timeout_s=args.timeout_s,
                )
                dt = time.time() - t0
                status = result["status_code"]
                body = result["body"]
                finish = _openai_finish_reason(body) if isinstance(body, dict) else None
                text = _openai_extract_text(body) if isinstance(body, dict) else ""

            rec.update({
                "attempt": attempt,
                "latency_s": round(dt, 4),
                "status_code": status,
                "finish_reason": finish,
                "max_output_tokens_used": cur_max_tokens,
                "usage": body.get("usage") if isinstance(body, dict) else None,
            })

            if args.debug_dir and isinstance(body, dict):
                dbg_path = os.path.join(args.debug_dir, f"{ex_id}.response.json")
                with open(dbg_path, "w", encoding="utf-8") as f:
                    json.dump(body, f, ensure_ascii=False, indent=2)

            # HTTP error handling
            if status != 200:
                rec["raw_text"] = ""
                rec["answer_parsed"] = None
                rec["parse_error"] = f"http_{status}"
                rec["error_body"] = body
                if status in (429, 500, 502, 503, 504) and attempt <= args.max_retries:
                    time.sleep(args.retry_backoff ** (attempt - 1))
                    continue
                break

            # parse + validate
            ans, parse_err = parse_answer_json(text)
            rec["raw_text"] = text
            rec["answer_parsed"] = ans
            rec["parse_error"] = parse_err

            is_valid, why = validate_answer(task_name, ans)
            rec["valid"] = bool(is_valid)
            rec["valid_reason"] = why

            need_retry = False
            if parse_err is not None or not is_valid:
                need_retry = True
            if (finish in {"length", "MAX_TOKENS"}) and (not (text or "").strip()):
                need_retry = True

            if (not need_retry) or (attempt >= args.max_retries):
                break

            # retry strategies
            if args.auto_expand_tokens and (finish in {"length", "MAX_TOKENS"} or parse_err == "empty_response"):
                next_tok = int(min(args.max_output_tokens_cap, max(cur_max_tokens + 16, int(cur_max_tokens * args.expand_factor))))
                if next_tok > cur_max_tokens:
                    cur_max_tokens = next_tok
                    time.sleep(args.retry_backoff ** (attempt - 1))
                    continue

            if args.repair_on_failure and not used_repair:
                used_repair = True
                time.sleep(args.retry_backoff ** (attempt - 1))
                continue

            time.sleep(args.retry_backoff ** (attempt - 1))

        out_rows.append(rec)
        if args.sleep > 0:
            time.sleep(args.sleep)

    write_jsonl(args.out, out_rows)
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
