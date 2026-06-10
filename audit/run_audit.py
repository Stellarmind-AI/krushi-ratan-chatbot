"""
Krushi Chatbot Audit Harness
============================

Purpose
-------
Drive the production chatbot exactly the way the Streamlit frontend does — over
the WebSocket at /ws/chat — for a list of questions, capture each answer plus the
relevant server-side internals (generated SQL, selected tools, intent, flow,
row counts, F1 clarification round-trip), and write a QA spreadsheet (CSV) that
imports cleanly into Google Sheets / Excel.

This is an OBSERVABILITY tool. It does not judge answer correctness. It only:
  1. (optionally) launches the backend and pipes its console logs to a file,
  2. acts as a WebSocket client — one fresh session per question (the bot has
     no memory, so every question is independent),
  3. auto-answers the F1 clarification buttons (configurable default policy),
  4. slices the server log per question to extract SQL / intent / flow,
  5. emits results.csv (the QA sheet) + results.json (raw backup) + per-Q logs.

Dynamic input
-------------
Point --questions at WHATEVER question list you have. Two formats are accepted
and auto-detected:

  • Markdown / text (recommended — your existing format):
        ## Module 1: Greeting Test Cases
        1. **Gujarati:** રામ રામ ભાઈ! ...
        2. **Hindi:**    ...
        3. **English:**  ...
    Module headers become the "Module" column; the **Language:** prefix (optional)
    becomes the "Language" column. No other tagging is required.

  • JSON (advanced — lets you pre-set expected_flow / clarification_choice):
        [{"question": "...", "expected_flow": "SQL", "clarification_choice": "crop_price"}]

Why it launches the backend
---------------------------
The app logger (app/core/logger.py) writes ONLY to the console — there is no log
file. The generated SQL lives only in that stream. So to capture it reliably and
correlate it per question, the harness owns the backend process and redirects its
console to a per-run server.log. Use --attach to connect to a backend you run
yourself (pass --server-log <path> if you redirected its console to a file).

Usage
-----
    python audit/run_audit.py                         # launch backend, run questions.md
    python audit/run_audit.py --questions my_list.md  # any markdown/text list
    python audit/run_audit.py --questions my.json     # advanced JSON list
    python audit/run_audit.py --attach --server-log server.log

No new dependencies — uses `websockets` and `requests` (already in requirements.txt).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import requests
import websockets

SCRIPT_DIR   = Path(__file__).resolve().parent          # .../audit
PROJECT_ROOT = SCRIPT_DIR.parent                        # repo root

VALID_FLOWS = {"GREETING", "GENERAL", "NAVIGATION", "SQL"}
LANG_WORDS  = {"gujarati", "hindi", "english", "gujrati", "guj", "hin", "eng"}

# ── F1 clarification: pick the option that matches the question's DOMAIN ──────
# The wire only carries {label, intent_key}; the label is translated to the
# user's language, so we match on intent_key (stable English) instead. We infer
# the intended domain from (1) the module/section header in the question file
# [strongest — it is the tester's own grouping], then (2) multilingual keywords
# in the question text, and only then fall back to a safe default. The first F1
# button is usually 'navigation', so we never blindly pick the first option.
NAV_KEY = "navigation"

# intent_key (from confirmation_layer.INTENT_TO_TABLES) -> coarse domain bucket
INTENT_DOMAIN = {
    "navigation": "navigation",
    "crop_price": "crop",
    "kshop_product": "kshop", "equipment_kshop": "kshop",
    "buy_sell_product": "buysell", "equipment_used": "buysell",
    "seed_info": "seed",
    "local_news": "news",
    "video_search": "video",
}

# Substring (lowercased) in a MODULE / SECTION header -> domain.
_HEADER_DOMAIN = [
    ("navigation", "navigation"),
    ("mandi", "crop"), ("bhav", "crop"), ("flow a", "crop"), ("crop", "crop"),
    ("k-shop", "kshop"), ("kshop", "kshop"), ("flow b", "kshop"), ("new equipment", "kshop"),
    ("buy/sell", "buysell"), ("buy sell", "buysell"), ("flow c", "buysell"),
    ("used", "buysell"), ("livestock", "buysell"), ("second-hand", "buysell"),
    ("seed", "seed"),
    ("news", "news"), ("weather", "news"),
    ("video", "video"),
]

# Multilingual keyword buckets for the question TEXT (fallback when the header
# is not domain-specific). Extend freely — Gujarati / Hindi / English / romanized.
_DOMAIN_KEYWORDS = {
    "crop":    ["ભાવ", "બજાર", "મંડી", "યાર્ડ", "રોકડિયા", "અનાજ", "કઠોળ", "તેલીબીયા", "મણ",
                "मंडी", "भाव", "दाम", "कीमत", "अनाज",
                "mandi", "bhav", "yard", "price", "rate", "crop", "maund"],
    "kshop":   ["કંપની", "નવું", "પંપ", "મોટર", "સ્પ્રેયર", "સીડર", "ઓગર", "બ્રશ કટર", "ઝટકા", "સબમર્સિબલ",
                "कंपनी", "नया", "पंप", "मोटर",
                "k-shop", "kshop", "company", "new", "pump", "motor", "sprayer", "seeder", "starter"],
    "buysell": ["વેચવા", "જૂનો", "જૂની", "ભેંસ", "ગાય", "ઘેટા", "બકરા", "ઘોડો", "બળદ", "સાંઢ",
                "ટ્રેક્ટર", "થ્રેશર", "ટ્રોલી", "પશુ", "લિસ્ટ",
                "बेचना", "पुराना", "भैंस", "गाय", "ट्रैक्टर", "पशु",
                "sell", "used", "old", "buffalo", "cow", "goat", "bull", "tractor", "thresher", "livestock"],
    "seed":    ["બીજ", "બિયારણ", "વાવણી", "વાવવા", "seed", "variety", "sowing"],
    "news":    ["સમાચાર", "ન્યૂઝ", "હવામાન", "ચેતવણી", "યોજના", "સબસિડી",
                "समाचार", "मौसम", "योजना", "सब्सिडी",
                "news", "weather", "alert", "scheme", "subsidy", "forecast"],
    "video":   ["વીડિયો", "વિડીયો", "વિડિઓ", "वीडियो", "video", "videos"],
    "navigation": ["કેવી રીતે", "કેમનું", "ક્યાં જઈને", "ક્યાં ક્લિક", "કિયા ખાનામાં", "પાનું",
                   "ફોટો પાડીને", "ઓર્ડર થાય", "મુકવો", "ભરવાનું",
                   "कैसे", "कहां जाकर", "कहां क्लिक", "सेक्शन", "पेज", "फॉर्म",
                   "how to", "how do i", "where do i", "where can i", "which section",
                   "which page", "upload", "place the order", "fill out", "steps to"],
}


def detect_target_domain(q: dict) -> tuple[str | None, str]:
    """Infer the question's intended domain. Returns (domain|None, via)."""
    header = (q.get("category") or "").lower()
    for needle, dom in _HEADER_DOMAIN:
        if needle in header:
            return dom, "module"
    text = (q.get("question") or "").lower()
    scores = {dom: sum(1 for w in words if w.lower() in text)
              for dom, words in _DOMAIN_KEYWORDS.items()}
    scores = {d: s for d, s in scores.items() if s}
    if scores:
        return max(scores, key=scores.get), "keyword"
    return None, ""


# ─────────────────────────────────────────────────────────────────────────────
# Config / CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Krushi chatbot audit harness")
    p.add_argument("--questions", default=str(SCRIPT_DIR / "questions.md"),
                   help="Path to questions file (markdown/text list OR JSON). Auto-detected.")
    p.add_argument("--out", default=str(SCRIPT_DIR / "runs"),
                   help="Output directory for run folders.")
    p.add_argument("--host", default="127.0.0.1", help="Backend host.")
    p.add_argument("--port", type=int, default=int(os.getenv("APP_PORT", "8000")),
                   help="Backend port.")
    p.add_argument("--ws-path", default="/ws/chat", help="WebSocket path.")
    p.add_argument("--delay", type=float, default=6.0,
                   help="Minimum seconds between any two inputs (rate-limit safety).")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Max seconds to wait for one question to fully answer.")
    p.add_argument("--recv-timeout", type=float, default=30.0,
                   help="Max seconds to wait for a single WS message before re-checking.")
    p.add_argument("--clarify-default", default="first",
                   help="Which F1 button to click when a question doesn't specify one: "
                        "'first', 'last', or 'index:N' (0-based).")
    p.add_argument("--attach", action="store_true",
                   help="Do NOT launch the backend; connect to an already-running one.")
    p.add_argument("--server-log", default=None,
                   help="(attach mode) Path to the backend's console log, if you redirected it.")
    p.add_argument("--startup-timeout", type=float, default=90.0,
                   help="Max seconds to wait for a launched backend to become ready.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Question loading — dynamic: markdown/text list OR JSON, auto-detected
# ─────────────────────────────────────────────────────────────────────────────
_RE_HEADER  = re.compile(r"^\s*#{1,6}\s*(?P<title>.+?)\s*$")
_RE_MODULE  = re.compile(r"(?i)^\s*module\s+\d+\s*[:\-].*$")
_RE_ITEM    = re.compile(r"^\s*(?:(?P<num>\d+)[.)]|[-*•])\s*(?P<body>.+)$")
_RE_LANGPFX = re.compile(r"^\*{0,2}\s*([A-Za-z]+)\s*:\s*\*{0,2}\s*(?P<rest>.*)$", re.S)


def _strip_lang_prefix(text: str):
    """Return (language|None, cleaned_text). Recognises **Gujarati:** / Hindi: etc."""
    m = _RE_LANGPFX.match(text.strip())
    if m and m.group(1).lower() in LANG_WORDS:
        return m.group(1).capitalize(), m.group("rest").strip()
    return None, text.strip()


def parse_text_questions(text: str) -> list[dict]:
    """Parse a simple markdown/text list (the format you already keep)."""
    items: list[dict] = []
    module = None
    current = None
    seq = 0
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        # Section header (## Module ... or a bare 'Module N: ...' line)
        is_header = (_RE_HEADER.match(line) and not _RE_ITEM.match(line)) or _RE_MODULE.match(line)
        if is_header:
            h = _RE_HEADER.match(line)
            module = (h.group("title") if h else line).strip().strip("*").strip()
            current = None
            continue

        m = _RE_ITEM.match(line)
        if m:
            lang, body = _strip_lang_prefix(m.group("body"))
            seq += 1
            qid = f"q{int(m.group('num')):03d}" if m.group("num") else f"q{seq:03d}"
            current = {"id": qid, "question": body, "language": lang,
                       "category": module, "group": module}
            items.append(current)
            continue

        # Not a header, not a list item: a bare line. If it carries a language
        # prefix treat it as a new question; else append to the previous one
        # (wrapped line); else start a fresh question.
        lang, body = _strip_lang_prefix(line)
        if lang:
            seq += 1
            current = {"id": f"q{seq:03d}", "question": body, "language": lang,
                       "category": module, "group": module}
            items.append(current)
        elif current is not None:
            current["question"] = (current["question"] + " " + body).strip()
        else:
            seq += 1
            current = {"id": f"q{seq:03d}", "question": body, "language": None,
                       "category": module, "group": module}
            items.append(current)
    return items


def _validate(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i, q in enumerate(items, 1):
        if not isinstance(q, dict) or not str(q.get("question", "")).strip():
            print(f"  ! skipping item #{i}: missing 'question'")
            continue
        ef = (q.get("expected_flow") or "").strip().upper()
        if ef and ef not in VALID_FLOWS:
            print(f"  ! item '{q.get('id', i)}': unknown expected_flow '{ef}'")
        out.append(q)
    return out


def load_questions(path: str) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8")
    ext = Path(path).suffix.lower()

    if ext == ".json":
        data = json.loads(raw)
    elif ext in (".md", ".txt", ".markdown"):
        return _validate(parse_text_questions(raw))
    else:
        try:
            data = json.loads(raw)            # unknown extension: try JSON first
        except json.JSONDecodeError:
            return _validate(parse_text_questions(raw))

    items = data.get("questions", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("JSON questions file must be a list or {'questions': [...]}.")
    return _validate(items)


# ─────────────────────────────────────────────────────────────────────────────
# Backend lifecycle (launch mode)
# ─────────────────────────────────────────────────────────────────────────────
def launch_backend(host: str, port: int, log_path: Path):
    """Start uvicorn (reload OFF, unbuffered) with console -> log_path."""
    logf = open(log_path, "w", encoding="utf-8", errors="replace")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        sys.executable, "-u", "-m", "uvicorn", "app.main:app",
        "--host", host, "--port", str(port), "--log-level", "info",
    ]
    print(f"  launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=logf,
                            stderr=subprocess.STDOUT, env=env)
    return proc, logf


def stop_backend(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def wait_until_ready(host: str, port: int, timeout: float, proc=None) -> bool:
    """Poll the root health ping until the server answers (or the process dies)."""
    url = f"http://{host}:{port}/"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            if requests.get(url, timeout=3).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def get_stats(host: str, port: int) -> dict:
    try:
        r = requests.get(f"http://{host}:{port}/api/stats", timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Log reading + per-question parsing
# ─────────────────────────────────────────────────────────────────────────────
def read_log(path: Path | None) -> str:
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return ""


# Markers emitted by app/core/logger.py + orchestrator. Parsed leniently.
_RE_SQL_GEN   = re.compile(r"📊 SQL GENERATED \| table=(?P<table>.*?) \| sql=(?P<sql>.*)")
_RE_SQL_EXEC  = re.compile(r"🔍 SQL EXECUTING \| table=(?P<table>.*?) \| sql=(?P<sql>.*)")
_RE_SQL_ROWS  = re.compile(r"SQL RESULT \| table=(?P<table>.*?) \| rows=(?P<rows>\d+)")
_RE_TOOLS     = re.compile(r"🔧 TOOLS SELECTED: (?P<tools>[^|]*)")
_RE_INTENT_RT = re.compile(r"⚡ INTENT ROUTED \(no LLM\) → (?P<rule>[^|]*)")
_RE_F1_INTENT = re.compile(r"🎯 F1 confirmed intent='(?P<intent>[^']*)'")
_RE_ROUTE     = re.compile(r"🔀 ROUTE → \[(?P<flow>[^\]]+)\]")
_RE_STEP1_FLOW= re.compile(r"DONE \[STEP 1 / ROUTE AGENT\][^\n]*?flow=(?P<flow>\w+)")
_RE_PIPE_END  = re.compile(r"✅ PIPELINE END(?P<cache> \[CACHE HIT\])? — (?P<flow>\w+) flow — total=(?P<ms>\d+)ms")
_RE_LANG      = re.compile(r"LANGUAGE DETECTED \| lang_type=(?P<lang>\w+)")
_RE_NO_DATA   = re.compile(r"🔍 NO DATA FOUND")
_RE_ERROR     = re.compile(r"\[ERROR\][^\n]*\| [^|]+ \| (?P<msg>.*)")


def _norm_sql(s: str) -> str:
    """Normalise SQL for de-duplication: collapse whitespace, drop a trailing
    safety `LIMIT <n>` and any trailing semicolon. The orchestrator logs the
    LLM's SQL (no LIMIT) and the executor re-logs the same query with the
    `LIMIT 1000` safety cap appended — both under the same marker. These are the
    SAME query, so we keep only the first (the agent's generated SQL)."""
    s = " ".join(s.split())
    s = re.sub(r"\s+limit\s+\d+\s*$", "", s, flags=re.IGNORECASE)
    return s.rstrip(";").strip()


def parse_log(slice_text: str) -> dict:
    """Extract the relevant internals from one question's server-log slice."""
    raw_sql = [{"table": m.group("table").strip(), "sql": m.group("sql").strip()}
               for m in _RE_SQL_GEN.finditer(slice_text)]
    if not raw_sql:
        raw_sql = [{"table": m.group("table").strip(), "sql": m.group("sql").strip()}
                   for m in _RE_SQL_EXEC.finditer(slice_text)]
    sql, _seen = [], set()
    for item in raw_sql:                       # de-dup the generated/executed double-log
        key = (item["table"], _norm_sql(item["sql"]))
        if key in _seen:
            continue
        _seen.add(key)
        sql.append(item)

    rows = {m.group("table").strip(): int(m.group("rows"))
            for m in _RE_SQL_ROWS.finditer(slice_text)}

    tools = [t.strip() for m in _RE_TOOLS.finditer(slice_text)
             for t in m.group("tools").split(",") if t.strip()]

    def first(rx, grp):
        m = rx.search(slice_text)
        return m.group(grp) if m else None

    pipe_flow, server_ms, cache = None, None, False
    m = _RE_PIPE_END.search(slice_text)
    if m:
        pipe_flow = m.group("flow")
        server_ms = int(m.group("ms"))
        cache = bool(m.group("cache"))

    route = _RE_STEP1_FLOW.search(slice_text) or _RE_ROUTE.search(slice_text)

    return {
        "sql": sql, "sql_rows": rows, "tools": tools,
        "intent": first(_RE_F1_INTENT, "intent"),
        "intent_rule": (first(_RE_INTENT_RT, "rule") or "").strip() or None,
        "route_flow": route.group("flow") if route else None,
        "pipeline_flow": pipe_flow, "server_total_ms": server_ms,
        "cache_hit_log": cache, "lang_detected": first(_RE_LANG, "lang"),
        "errors": [m.group("msg").strip() for m in _RE_ERROR.finditer(slice_text)],
        "flags": (["NO_DATA_FOUND"] if _RE_NO_DATA.search(slice_text) else []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Clarification handling
# ─────────────────────────────────────────────────────────────────────────────
def choose_clarification(q: dict, options: list[dict], default_policy: str
                         ) -> tuple[str, str, bool, str]:
    """
    Decide which F1 button to 'click'. Returns (intent_key, label, auto_picked, via).

    Priority:
      1. explicit per-question `clarification_choice` (JSON power users) — wins.
         May be an intent_key, a 0-based index, or a label/intent substring.
      2. domain match: infer the question's domain (module header, then keywords)
         and pick the option whose intent_key belongs to that domain. Matching is
         on intent_key (stable English) — NOT the label, which is translated.
      3. safe default: the first NON-navigation (data) option — navigation has its
         own module, and an F1 prompt on a data question almost always wants data.
      4. only-navigation options: apply --clarify-default policy.
    """
    if not options:
        return "", "", True, "none"

    # 1) explicit per-question override
    choice = q.get("clarification_choice")
    if choice is not None:
        s = str(choice).strip()
        if s.isdigit() and int(s) < len(options):
            o = options[int(s)]; return o["intent_key"], o.get("label", ""), False, "explicit"
        for o in options:
            if o["intent_key"] == s:
                return o["intent_key"], o.get("label", ""), False, "explicit"
        low = s.lower()
        for o in options:
            if low in o.get("label", "").lower() or low in o["intent_key"].lower():
                return o["intent_key"], o.get("label", ""), False, "explicit"

    # 2) match the question's intended domain (module header, then keywords)
    domain, via = detect_target_domain(q)
    if domain:
        for o in options:
            if INTENT_DOMAIN.get(o["intent_key"]) == domain:
                return o["intent_key"], o.get("label", ""), True, via

    # 3) safe default: first non-navigation (data) option
    for o in options:
        if o["intent_key"] != NAV_KEY:
            return o["intent_key"], o.get("label", ""), True, "default-data"

    # 4) only navigation options remain -> apply policy
    idx = 0
    if default_policy == "last":
        idx = len(options) - 1
    elif default_policy.startswith("index:"):
        try:
            idx = max(0, min(int(default_policy.split(":", 1)[1]), len(options) - 1))
        except ValueError:
            idx = 0
    o = options[idx]
    return o["intent_key"], o.get("label", ""), True, "default"


# ─────────────────────────────────────────────────────────────────────────────
# Pacing — guarantee >= delay seconds between every input
# ─────────────────────────────────────────────────────────────────────────────
class Pacer:
    def __init__(self, delay: float):
        self.delay = delay
        self.last = 0.0

    async def wait(self):
        if self.last:
            remaining = self.delay - (time.monotonic() - self.last)
            if remaining > 0:
                await asyncio.sleep(remaining)

    def mark(self):
        self.last = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# Run a single question over a fresh WebSocket session
# ─────────────────────────────────────────────────────────────────────────────
async def run_question(q: dict, args, server_log_path: Path | None,
                       pacer: Pacer) -> dict:
    sid  = str(q.get("id") or uuid4())
    text = q["question"]
    uri  = f"ws://{args.host}:{args.port}{args.ws_path}"

    result: dict = {
        "id": q.get("id"), "group": q.get("group"), "language": q.get("language"),
        "category": q.get("category"), "expected_flow": (q.get("expected_flow") or "").upper(),
        "notes": q.get("notes"), "question": text,
        "answer": "", "flow": None, "lang_type": None, "cache_hit": None,
        "query_data": {}, "clarification": {"asked": False},
        "latency_ms": None, "error": None, "other_messages": [],
    }

    log_start = len(read_log(server_log_path))
    t0 = time.perf_counter()
    final: dict | None = None
    try:
        async with websockets.connect(uri, max_size=None, open_timeout=15) as ws:
            await pacer.wait()
            await ws.send(json.dumps(
                {"type": "text_input", "text": text, "session_id": sid}, ensure_ascii=False))
            pacer.mark()

            deadline = time.perf_counter() + args.timeout
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    result["error"] = "timeout waiting for response"
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(),
                                                 timeout=min(remaining, args.recv_timeout))
                except asyncio.TimeoutError:
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    result["other_messages"].append({"raw": str(raw)[:200]})
                    continue

                mtype = msg.get("type")
                if mtype == "clarification_request":
                    opts = msg.get("options", [])
                    ik, label, auto, via = choose_clarification(q, opts, args.clarify_default)
                    result["clarification"] = {
                        "asked": True, "question": msg.get("question"),
                        "scenario": msg.get("scenario"),
                        "matched_keyword": msg.get("matched_keyword"),
                        "options": opts, "chosen_intent_key": ik,
                        "chosen_label": label, "auto_picked": auto, "matched_by": via,
                    }
                    await pacer.wait()
                    await ws.send(json.dumps(
                        {"type": "clarification_response", "session_id": sid,
                         "intent_key": ik}, ensure_ascii=False))
                    pacer.mark()
                    continue
                elif mtype == "text_output":
                    final = msg
                    if msg.get("is_complete", True):
                        break
                elif mtype == "error":
                    result["error"] = msg.get("error", "unknown error")
                    final = msg
                    break
                elif mtype == "control_ack":
                    continue
                else:
                    result["other_messages"].append(msg)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000)

    await asyncio.sleep(0.4)  # let the server flush its final log lines
    log_slice = read_log(server_log_path)[log_start:]
    result["log"] = parse_log(log_slice)
    result["_log_slice"] = log_slice

    if final and final.get("type") == "text_output":
        result["answer"]     = final.get("text", "")
        result["flow"]       = final.get("flow")
        result["lang_type"]  = final.get("lang_type")
        result["cache_hit"]  = final.get("cache_hit")
        result["query_data"] = final.get("query_data", {})
        result["sources"]    = final.get("sources", [])

    exp = result["expected_flow"]
    act = (result["flow"] or result["log"].get("pipeline_flow") or "")
    result["routing_ok"] = (exp == str(act).upper()) if exp and act else None
    result["errored"] = bool(result["error"])
    result["empty_answer"] = (not result["answer"].strip()) and not result["errored"]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
async def run_all(questions: list[dict], args, server_log_path: Path | None,
                  logs_dir: Path) -> list[dict]:
    pacer = Pacer(args.delay)
    results: list[dict] = []
    total = len(questions)
    for i, q in enumerate(questions, 1):
        qid = q.get("id") or f"q{i:03d}"
        print(f"  [{i}/{total}] {qid:<10} {q['question'][:60]}")
        res = await run_question(q, args, server_log_path, pacer)

        slice_text = res.pop("_log_slice", "")
        log_file = logs_dir / f"{i:03d}_{qid}.log"
        log_file.write_text(slice_text, encoding="utf-8")
        res["raw_log"] = str(log_file.relative_to(logs_dir.parent))

        status = ("ERR" if res["errored"] else
                  "EMPTY" if res["empty_answer"] else
                  f"flow={res['flow']}")
        clar = " (F1)" if res["clarification"].get("asked") else ""
        print(f"        -> {status}{clar}  {res['latency_ms']}ms")
        results.append(res)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# QA spreadsheet (CSV — imports into Google Sheets / Excel)
# ─────────────────────────────────────────────────────────────────────────────
SHEET_COLUMNS = [
    "ID", "Module", "Language", "Detected Lang", "Question", "Flow",
    "Confirmation Layer", "SQL Generated by agent", "DB Result (rows)",
    "Answer", "Latency (s)", "Issues",
    "Expected Answer", "Pass/Fail", "Remark",   # <- you fill these manually
]


def _confirmation_cell(c: dict) -> str:
    if not c.get("asked"):
        return "NO"
    opts = " | ".join(o.get("label", "") for o in c.get("options", []))
    via = c.get("matched_by") or ""
    tag = f"auto via {via}" if c.get("auto_picked") else (via or "explicit")
    picked = c.get("chosen_label") or c.get("chosen_intent_key") or ""
    return (f"picked: {picked}  [{tag}]\n"
            f"intent_key: {c.get('chosen_intent_key') or ''}\n"
            f"asked: {c.get('question') or ''}\n"
            f"options: {opts}")


def _db_rows_cell(r: dict) -> str:
    rows = r["log"].get("sql_rows") or {}
    if not rows and r.get("query_data"):
        rows = {t: (len(v) if isinstance(v, list) else 1)
                for t, v in r["query_data"].items()}
    return ", ".join(f"{t}={n}" for t, n in rows.items())


def write_sheet(results: list[dict], out_dir: Path) -> Path:
    path = out_dir / "results.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(SHEET_COLUMNS)
        for r in results:
            log = r["log"]
            sql = "\n".join(f"[{s['table']}] {s['sql']}" for s in log.get("sql", []))
            issues = []
            if r["errored"]:
                issues.append(f"ERROR: {r['error']}")
            if r["empty_answer"]:
                issues.append("empty answer")
            if r["routing_ok"] is False:
                issues.append(f"flow expected {r['expected_flow']}, got {r['flow']}")
            issues.extend(log.get("flags", []))
            detected = r.get("lang_type") or log.get("lang_detected") or ""
            lat = f"{r['latency_ms'] / 1000:.1f}" if r.get("latency_ms") is not None else ""
            w.writerow([
                r.get("id") or "", r.get("category") or "", r.get("language") or "",
                detected, r.get("question") or "", r.get("flow") or "",
                _confirmation_cell(r["clarification"]), sql, _db_rows_cell(r),
                r.get("answer") or "", lat, "; ".join(issues),
                "", "", "",                       # Expected Answer / Pass-Fail / Remark
            ])
    return path


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()

    questions = load_questions(args.questions)
    if not questions:
        print("No valid questions found. Add questions to the file and retry.")
        return 1
    print(f"Loaded {len(questions)} question(s) from {args.questions}")

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out) / run_id
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    proc = logf = None
    server_log_path: Path | None = None
    mode = "attach" if args.attach else "launch"
    try:
        if args.attach:
            server_log_path = Path(args.server_log) if args.server_log else None
            print(f"Attach mode — expecting backend at {args.host}:{args.port}")
            if not wait_until_ready(args.host, args.port, 10):
                print("  ! backend not reachable. Start it, or drop --attach to auto-launch.")
                return 1
            if not server_log_path:
                print("  ! no --server-log given: SQL/intent columns will be empty "
                      "(answer + flow still captured from the wire).")
        else:
            server_log_path = out_dir / "server.log"
            print("Launch mode — starting backend...")
            proc, logf = launch_backend(args.host, args.port, server_log_path)
            if not wait_until_ready(args.host, args.port, args.startup_timeout, proc):
                print(f"  ! backend failed to become ready. See {server_log_path}")
                return 1
            print("  backend ready.")

        stats = get_stats(args.host, args.port)
        meta = {
            "run_id": run_id, "mode": mode, "host": args.host, "port": args.port,
            "delay": args.delay, "clarify_default": args.clarify_default,
            "questions_file": args.questions,
            "sql_enabled": stats.get("features", {}).get("sql_flow"),
            "model": stats.get("llm", {}).get("primary_model"),
        }

        print(f"\nRunning {len(questions)} question(s) (>= {args.delay}s between inputs)...\n")
        results = asyncio.run(run_all(questions, args, server_log_path, logs_dir))

        sheet_path = write_sheet(results, out_dir)
        (out_dir / "results.json").write_text(   # raw, lossless backup (ignore if unused)
            json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8")

        # Console summary
        n = len(results)
        err = sum(r["errored"] for r in results)
        emp = sum(r["empty_answer"] for r in results)
        clr = sum(r["clarification"].get("asked", False) for r in results)
        lat = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
        print("\nDone.")
        print(f"  questions={n}  errored={err}  empty={emp}  clarified={clr}  "
              f"avg_latency={round(sum(lat) / len(lat)) if lat else 0}ms")
        print(f"\n  QA SHEET (import to Google Sheets) : {sheet_path}")
        print(f"  raw json                            : {out_dir / 'results.json'}")
        print(f"  per-question logs                   : {logs_dir}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        if not args.attach:
            stop_backend(proc)
        if logf:
            logf.close()


if __name__ == "__main__":
    raise SystemExit(main())
