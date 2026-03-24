"""
Enhanced Structured Logger — Full pipeline observability.

Every stage of the pipeline logs with:
  - Timing (how long each step took)
  - Token counts (LLM cost tracking)
  - Flow type (SQL / NAVIGATION / GENERAL / GREETING)
  - Cache hits/misses
  - Translation events
  - Step-by-step progress markers
"""

import logging
import os
import sys
import threading
import time
from typing import Any, Dict, Optional
from datetime import datetime
from app.core.config import settings


_TERMINAL_LOCK = threading.Lock()
_CONSOLE_READY = False


def _supports_color() -> bool:
    """Enable ANSI colors only when the current terminal is likely to support them."""
    stream = getattr(sys, "stdout", None)
    if not stream or not hasattr(stream, "isatty") or not stream.isatty():
        return False

    if os.name != "nt":
        return True

    return bool(
        os.getenv("WT_SESSION")
        or os.getenv("ANSICON")
        or os.getenv("TERM_PROGRAM")
        or os.getenv("ConEmuANSI") == "ON"
    )


def _configure_console() -> None:
    """Best-effort UTF-8 console setup for Windows terminals."""
    global _CONSOLE_READY
    if _CONSOLE_READY:
        return

    _CONSOLE_READY = True

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    if os.name != "nt":
        return

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except Exception:
        pass


def _write_terminal(text: str) -> None:
    """Write Unicode text directly to the terminal when possible."""
    _configure_console()

    with _TERMINAL_LOCK:
        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes

                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
                invalid_handle = ctypes.c_void_p(-1).value
                mode = wintypes.DWORD()

                if handle not in (0, invalid_handle) and kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                    written = wintypes.DWORD()
                    kernel32.WriteConsoleW(handle, text, len(text), ctypes.byref(written), None)
                    return
            except Exception:
                pass

        stream = sys.stdout
        try:
            stream.write(text)
        except UnicodeEncodeError:
            if hasattr(stream, "buffer"):
                stream.buffer.write(text.encode("utf-8", errors="replace"))
            else:
                stream.write(text.encode("ascii", errors="backslashreplace").decode("ascii"))
        stream.flush()


class UnicodeConsoleHandler(logging.Handler):
    """Logging handler that writes through the Unicode-safe terminal writer."""

    terminator = "\n"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) + self.terminator
            _write_terminal(msg)
        except Exception:
            self.handleError(record)


class ColoredFormatter(logging.Formatter):
    USE_COLOR = _supports_color()
    COLORS = {
        'DEBUG':    '\033[36m',   # Cyan
        'INFO':     '\033[32m',   # Green
        'WARNING':  '\033[33m',   # Yellow
        'ERROR':    '\033[31m',   # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET':    '\033[0m'
    }
    EMOJIS = {
        'DEBUG': '🔍', 'INFO': 'ℹ️', 'WARNING': '⚠️',
        'ERROR': '❌', 'CRITICAL': '🚨'
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET']) if self.USE_COLOR else ""
        emoji = self.EMOJIS.get(record.levelname, '')
        reset = self.COLORS['RESET'] if self.USE_COLOR else ""
        ts = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        msg = f"{color}{emoji} [{record.levelname}]{reset} {ts} | {record.name} | {record.getMessage()}"
        if hasattr(record, 'extra_fields') and record.extra_fields:
            extra = " | ".join(f"{k}={v}" for k, v in record.extra_fields.items())
            msg += f" | {extra}"
        return msg


class StructuredLogger:
    """
    Structured logger with step tracking and pipeline timing.

    Usage:
        logger = get_logger("my_module")
        logger.info("message", key=value, ...)
        logger.step("ROUTE AGENT", "Classifying question", flow="SQL")
    """

    def __init__(self, name: str):
        _configure_console()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, settings.LOG_LEVEL))
        self.logger.handlers.clear()
        handler = UnicodeConsoleHandler()
        handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.name = name

    def _log(self, level: int, msg: str, **kwargs):
        extra = {k: v for k, v in kwargs.items() if v is not None}
        self.logger.log(level, msg, extra={'extra_fields': extra})

    def debug(self, msg: str, **kwargs):    self._log(logging.DEBUG,    msg, **kwargs)
    def info(self, msg: str, **kwargs):     self._log(logging.INFO,     msg, **kwargs)
    def warning(self, msg: str, **kwargs):  self._log(logging.WARNING,  msg, **kwargs)
    def error(self, msg: str, **kwargs):    self._log(logging.ERROR,    msg, **kwargs)
    def critical(self, msg: str, **kwargs): self._log(logging.CRITICAL, msg, **kwargs)

    # ── Pipeline step markers ─────────────────────────────────────────────────

    def step(self, step_name: str, detail: str = "", **kwargs):
        """
        Log a named pipeline step.
        Example: logger.step("ROUTE AGENT", "Detected SQL flow", flow="SQL")
        """
        msg = f"┌─ STEP [{step_name}]"
        if detail:
            msg += f" — {detail}"
        self._log(logging.INFO, msg, **kwargs)

    def step_done(self, step_name: str, elapsed_ms: float, **kwargs):
        """Log step completion with timing."""
        self._log(logging.INFO,
                  f"└─ DONE [{step_name}] ({elapsed_ms:.0f}ms)", **kwargs)

    def pipeline_start(self, query: str, client_id: str = ""):
        """Log start of full pipeline for a query."""
        _write_terminal(f"🚀 PIPELINE START | query={query[:100]} | client={client_id}\n")

    def pipeline_end(self, flow: str, total_ms: float, cached: bool = False):
        """Log end of full pipeline."""
        cache_tag = " [CACHE HIT]" if cached else ""
        self._log(logging.INFO,
                  f"✅ PIPELINE END{cache_tag} — {flow} flow — total={total_ms:.0f}ms")

    # ── Specific event loggers ────────────────────────────────────────────────

    def route_decision(self, flow: str, reason: str, confidence: str = "high"):
        """Log route agent decision."""
        self._log(logging.INFO,
                  f"🔀 ROUTE → [{flow}]",
                  reason=reason, confidence=confidence)

    def translation_start(self, direction: str, lang_from: str, lang_to: str, text_len: int):
        """direction = 'input' or 'output'"""
        self._log(logging.INFO,
                  f"🌐 TRANSLATE [{direction.upper()}] {lang_from} → {lang_to}",
                  chars=text_len)

    def translation_done(self, direction: str, elapsed_ms: float, text_len: int):
        self._log(logging.INFO,
                  f"🌐 TRANSLATE [{direction.upper()}] DONE",
                  elapsed_ms=f"{elapsed_ms:.0f}ms", output_chars=text_len)

    # ── Content-level logging — see actual text flowing through the pipeline ──

    def translation_io(self, provider: str, direction: str,
                       input_text: str, output_text: str, elapsed_ms: float = 0.0):
        """
        Log the FULL input and output of a translation call.
        This is the key log for catching hallucinations in translation.
        """
        separator = "─" * 60
        block = (
            f"\n{separator}\n"
            f"🌐 TRANSLATION [{provider.upper()}] {direction} ({elapsed_ms:.0f}ms)\n"
            f"  INPUT  ({len(input_text)} chars):\n"
            f"    {input_text}\n"
            f"  OUTPUT ({len(output_text)} chars):\n"
            f"    {output_text}\n"
            f"{separator}"
        )
        self._log(logging.INFO, block)

    def llm_io(self, purpose: str, provider: str, model: str,
               system_prompt: str, user_prompt: str, response_text: str,
               tokens_used: Optional[int] = None, elapsed_ms: float = 0.0):
        """
        Log the FULL prompt and response of an LLM call.
        Truncates prompts at 500 chars but shows full response.
        """
        separator = "═" * 60
        sys_display = system_prompt[:500] + ("…" if len(system_prompt) > 500 else "")
        usr_display = user_prompt[:500] + ("…" if len(user_prompt) > 500 else "")
        block = (
            f"\n{separator}\n"
            f"🤖 LLM CALL [{purpose}] — {provider}/{model} ({elapsed_ms:.0f}ms, tokens={tokens_used})\n"
            f"  SYSTEM:\n    {sys_display}\n"
            f"  USER:\n    {usr_display}\n"
            f"  RESPONSE ({len(response_text)} chars):\n"
            f"    {response_text}\n"
            f"{separator}"
        )
        self._log(logging.INFO, block)

    def content_log(self, label: str, text: str, max_len: int = 0):
        """Log a labeled piece of content. max_len=0 means no truncation."""
        display = text if (max_len == 0 or len(text) <= max_len) else text[:max_len] + "…"
        self._log(logging.INFO, f"📝 {label}: {display}")

    def cache_hit(self, query: str, tools: list):
        self._log(logging.INFO,
                  f"⚡ CACHE HIT — skipping tool selection + SQL generation",
                  query=query[:60], tools=str(tools))

    def cache_miss(self, query: str):
        self._log(logging.DEBUG, f"💨 CACHE MISS", query=query[:60])

    def intent_routed(self, rule_name: str, tables: list):
        self._log(logging.INFO,
                  f"⚡ INTENT ROUTED (no LLM) → {rule_name}",
                  tables=str(tables))

    def llm_call_start(self, call_num: int, purpose: str, provider: str, est_tokens: int = 0):
        """Log start of an LLM call."""
        self._log(logging.INFO,
                  f"🤖 LLM CALL #{call_num} START — {purpose}",
                  provider=provider, est_tokens=est_tokens)

    def llm_call_done(self, call_num: int, purpose: str, elapsed_ms: float, tokens_used: int = 0):
        """Log completion of an LLM call."""
        self._log(logging.INFO,
                  f"🤖 LLM CALL #{call_num} DONE — {purpose}",
                  elapsed_ms=f"{elapsed_ms:.0f}ms", tokens=tokens_used)

    def tool_selection(self, tools: list, query: str):
        self._log(logging.INFO,
                  f"🔧 TOOLS SELECTED: {', '.join(tools)}", query=query[:80])

    def sql_generation(self, query: str, table: str):
        self._log(logging.INFO, f"📊 SQL GENERATED", table=table, sql=query[:200])

    def sql_execution_start(self, table: str, sql: str):
        self._log(logging.INFO, f"🔍 SQL EXECUTING", table=table, sql=sql[:150])

    def sql_execution_done(self, table: str, rows: int, elapsed_ms: float):
        self._log(logging.INFO, f"🗄️  SQL RESULT",
                  table=table, rows=rows, elapsed_ms=f"{elapsed_ms:.0f}ms")

    def query_execution(self, query: str, rows: int, execution_time: float):
        self.sql_execution_done("", rows, execution_time * 1000)

    def json_lookup(self, flow: str, matched_id: str, score: float):
        """Log a knowledge base JSON lookup result."""
        self._log(logging.INFO,
                  f"📖 JSON LOOKUP [{flow}]",
                  matched=matched_id, score=f"{score:.2f}")

    def final_answer(self, answer: str, tokens: Optional[int] = None, lang: str = "en"):
        self._log(logging.INFO,
                  f"💬 ANSWER GENERATED (english)",
                  chars=len(answer), tokens=tokens, lang=lang)

    def answer_translated(self, target_lang: str, elapsed_ms: float, chars: int):
        self._log(logging.INFO,
                  f"🌐 ANSWER TRANSLATED → {target_lang}",
                  elapsed_ms=f"{elapsed_ms:.0f}ms", chars=chars)

    def no_data_found(self, flow: str, query: str):
        self._log(logging.WARNING,
                  f"🔍 NO DATA FOUND [{flow}]", query=query[:80])

    def llm_call(self, provider: str, model: str, tokens: Optional[int] = None):
        self._log(logging.INFO, f"🤖 LLM CALL", provider=provider, model=model, tokens=tokens)

    def rate_limit_hit(self, provider: str, retry_after: float):
        self._log(logging.WARNING, f"⏱️ RATE LIMIT HIT",
                  provider=provider, retry_after=f"{retry_after:.2f}s")

    def fallback_trigger(self, from_provider: str, to_provider: str, reason: str):
        self._log(logging.WARNING, f"🔄 FALLBACK TRIGGERED",
                  from_provider=from_provider, to_provider=to_provider, reason=reason)

    def websocket_connect(self, client_id: str):
        self._log(logging.INFO, f"🔌 WEBSOCKET CONNECTED", client_id=client_id)

    def websocket_disconnect(self, client_id: str):
        self._log(logging.INFO, f"🔌 WEBSOCKET DISCONNECTED", client_id=client_id)

    def audio_processing(self, action: str, duration: Optional[float] = None,
                         format: Optional[str] = None):
        self._log(logging.INFO, f"🎤 AUDIO {action.upper()}", duration=duration, format=format)

    def error_with_context(self, error: Exception, context: Dict[str, Any]):
        self._log(logging.ERROR, f"ERROR: {str(error)}",
                  error_type=type(error).__name__, **context)


# ── Timing helper ─────────────────────────────────────────────────────────────

class Timer:
    """Simple context manager for measuring elapsed time in ms."""
    def __init__(self):
        self.elapsed_ms = 0.0
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


# ── Factory functions ─────────────────────────────────────────────────────────

def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)

def get_llm_logger() -> StructuredLogger:
    return get_logger("llm")

def get_database_logger() -> StructuredLogger:
    return get_logger("database")

def get_agent_logger() -> StructuredLogger:
    return get_logger("agent")

def get_websocket_logger() -> StructuredLogger:
    return get_logger("websocket")

def get_audio_logger() -> StructuredLogger:
    return get_logger("audio")

def get_route_logger() -> StructuredLogger:
    return get_logger("route_agent")
