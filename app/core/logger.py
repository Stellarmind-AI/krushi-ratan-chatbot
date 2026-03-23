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
import sys
import time
from typing import Any, Dict, Optional
from datetime import datetime
from app.core.config import settings


class ColoredFormatter(logging.Formatter):
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
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        emoji = self.EMOJIS.get(record.levelname, '')
        reset = self.COLORS['RESET']
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
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, settings.LOG_LEVEL))
        self.logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
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
        self._log(logging.INFO,
                  f"{'='*60}\n🚀 PIPELINE START",
                  query=query[:100], client=client_id)

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
