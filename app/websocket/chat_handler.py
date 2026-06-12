"""
WebSocket Chat Handler — Krushiratn AI Backend.

Pipeline:
  Step 1: Detect input language (language_processor — detect only, no translation)
  Step 2: Stage 1 Unified NLU → NLUFrame (one LLM call: intent + verbatim
          entities + English paraphrase + explicit constraints)
  Step 3: Ambiguous frame → code-built clarification buttons; button click
          PATCHES the stored frame (no Stage-1 re-run) and resumes
  Step 4: Orchestrator dispatches the frame → always generates an ENGLISH answer
  Step 5: Translate English → user's language (exit boundary)
  Step 6: Send translated answer to client

This clean separation means:
  - One understanding step; the original language never crosses past Stage 1
  - All downstream LLMs (SQL gen, answer gen, navigation, general) work in English
  - Language handling is one centralized exit step — easy to maintain
"""

import json
import time
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

from app.core.logger import get_logger, get_websocket_logger, Timer
from app.core.config import settings
from app.models.chat_models import ChatHistory, ChatMessage
from app.models.nlu_frame import NLUFrame
from app.services.agent.orchestrator import get_orchestrator
from app.services.agent.nlu import get_stage1_nlu
from app.services.agent.clarification import (
    build_clarification, serialize_request, apply_choice,
    NAV_INTENT_KEY, INTENT_TO_TABLES,
)
from app.services.language_processor import get_language_processor
from app.services.translation_service import translate_to_user_language, translate_list_to_user_language

logger       = get_websocket_logger()
pipeline_log = get_logger("pipeline")

# intent_key values a clarification button may legally carry.
_VALID_CLARIFICATION_KEYS = set(INTENT_TO_TABLES.keys()) | {NAV_INTENT_KEY}


def _safe_serialize(obj):
    import decimal
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_safe_serialize(i) for i in obj]
    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    elif isinstance(obj, decimal.Decimal):
        return float(obj)
    elif hasattr(obj, "model_dump"):
        return _safe_serialize(obj.model_dump())
    elif hasattr(obj, "dict"):
        return _safe_serialize(obj.dict())
    return str(obj)


class ChatHandler:

    def __init__(self):
        self.orchestrator       = get_orchestrator()
        self.language_processor = get_language_processor()
        self.nlu                = get_stage1_nlu()
        self._sessions: dict    = {}
        # {session_id: {"frame": NLUFrame, "lang_type": str}} while the
        # pipeline is paused waiting for the user's clarification pick.
        self._pending_clarifications: dict = {}

    async def handle_connection(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        logger.websocket_connect(client_id)
        try:
            while True:
                raw = await websocket.receive_text()
                if not raw or not raw.strip():
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as je:
                    logger.warning(f"Invalid JSON: {je}", raw=raw[:80])
                    await self._send_error(websocket, f"Invalid JSON: {je}")
                    continue

                msg_type = data.get("type")
                if msg_type == "text_input":
                    await self._handle_text_input(websocket, data, client_id)
                elif msg_type == "clarification_response":
                    # User picked one of the clarification options
                    await self._handle_clarification_response(websocket, data, client_id)
                elif msg_type == "control":
                    await self._handle_control(websocket, data, client_id)
                elif msg_type == "audio_input":
                    await self._send_error(websocket, "Voice input is not supported. Please use text input.")
                else:
                    await self._send_error(websocket, f"Unknown message type: {msg_type}")

        except WebSocketDisconnect:
            logger.websocket_disconnect(client_id)
        except Exception as e:
            logger.error_with_context(e, {"client_id": client_id})
            try:
                await self._send_error(websocket, str(e))
            except Exception:
                pass

    async def _handle_text_input(self, ws: WebSocket, data: dict, client_id: str):
        text       = data.get("text", "").strip()
        session_id = data.get("session_id") or client_id
        if not text:
            await self._send_error(ws, "Empty text input")
            return
        # A fresh question invalidates any pending clarification for the session.
        self._pending_clarifications.pop(session_id, None)
        logger.info("TEXT INPUT", text=text[:80])
        await self._run_pipeline(ws, text, session_id, client_id)

    async def _handle_clarification_response(self, ws: WebSocket, data: dict, client_id: str):
        """
        User tapped a clarification button.

        Expected payload:
          {
            "type":       "clarification_response",
            "session_id": "<session>",        # optional
            "intent_key": "crop_price"        # from the option the user tapped
          }

        The stored frame is PATCHED in code (apply_choice) and the pipeline
        resumes at routing — Stage 1 is NOT re-run, so a second clarification
        for the same query is structurally impossible.
        """
        session_id = data.get("session_id") or client_id
        intent_key = data.get("intent_key", "").strip()

        if not intent_key:
            await self._send_error(ws, "clarification_response missing intent_key")
            return
        if intent_key not in _VALID_CLARIFICATION_KEYS:
            await self._send_error(ws, f"Unknown intent_key: {intent_key}")
            return

        pending = self._pending_clarifications.pop(session_id, None)
        if not pending:
            await self._send_error(
                ws,
                "No pending clarification for this session. Please send your question again.",
            )
            return

        frame: NLUFrame = pending["frame"]
        lang_type       = pending["lang_type"]

        frame = apply_choice(frame, intent_key)
        logger.info(
            "CLARIFICATION RESOLVED",
            session_id=session_id,
            intent_key=intent_key,
            resumed_intent=frame.intent,
        )
        await self._run_pipeline(
            ws,
            frame.raw_input,
            session_id,
            client_id,
            resumed_frame=frame,
            lang_type_override=lang_type,
        )

    async def _handle_control(self, ws: WebSocket, data: dict, client_id: str):
        action     = data.get("action")
        session_id = data.get("session_id") or client_id
        if action == "clear_history":
            self._sessions.pop(session_id, None)
            self._pending_clarifications.pop(session_id, None)
            await ws.send_text(json.dumps({"type": "control_ack", "action": "clear_history"}, ensure_ascii=False))
            logger.info("History cleared", session_id=session_id)
        else:
            await self._send_error(ws, f"Unknown control action: {action}")

    async def _run_pipeline(
        self,
        ws: WebSocket,
        user_text: str,
        session_id: str,
        client_id: str,
        resumed_frame: Optional[NLUFrame] = None,  # set when resuming after clarification
        lang_type_override: Optional[str] = None,  # reuse detected lang on resume
    ):
        """
        Full pipeline:
          Step 1: Detect language (skipped on clarification resume)
          Step 2: Stage 1 NLU → frame (skipped on resume — patched frame passed in)
          Step 3: Ambiguous → clarification pause (only on first pass)
          Step 4: Orchestrator → English answer
          Step 5: Translate English → user language
          Step 6: Send to client
        """
        pipeline_start = time.perf_counter()
        history = self._sessions.setdefault(session_id, ChatHistory(session_id=session_id))
        # Only add the user message to history on the first pass (not on resume).
        if resumed_frame is None:
            history.messages.append(ChatMessage(role="user", content=user_text))

        # ── Step 1: Language detection ───────────────────────────────────────
        if lang_type_override:
            processed_text = user_text
            lang_type      = lang_type_override
        else:
            with Timer() as t:
                processed_text, lang_type = await self.language_processor.process(user_text)
            logger.info("LANGUAGE DETECTED", lang_type=lang_type,
                        original=user_text[:60], elapsed_ms=f"{t.elapsed_ms:.1f}ms")

        # ── Step 2: Stage 1 Unified NLU ──────────────────────────────────────
        if resumed_frame is not None:
            frame = resumed_frame
        else:
            with Timer() as t:
                frame = await self.nlu.extract(processed_text, detected_language=lang_type)
            logger.info("STAGE1 FRAME", intent=frame.intent,
                        qtype=frame.query_type,
                        keyword=frame.primary_keyword()[:40],
                        elapsed_ms=f"{t.elapsed_ms:.0f}ms")

        # ── Step 3: Clarification (first pass only) ──────────────────────────
        if frame.needs_clarification and resumed_frame is None:
            if settings.is_sql_enabled:
                clarification = build_clarification(frame)
            else:
                clarification = None  # SQL off → clarifying SQL domains is pointless

            if clarification is not None:
                self._pending_clarifications[session_id] = {
                    "frame":     frame,
                    "lang_type": lang_type,
                }
                logger.info(
                    "PIPELINE PAUSED — clarification",
                    session_id=session_id,
                    scenario=clarification.scenario,
                    query=processed_text[:60],
                )
                payload = serialize_request(clarification)
                # Translate question + option labels to the user's language.
                payload["question"] = await translate_to_user_language(
                    clarification.question, lang_type
                )
                raw_labels = [opt["label"] for opt in payload["options"]]
                translated_labels = await translate_list_to_user_language(raw_labels, lang_type)
                for i, opt in enumerate(payload["options"]):
                    opt["label"] = translated_labels[i]
                payload["timestamp"] = datetime.now().isoformat()
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
                return  # paused; resumes via _handle_clarification_response
            else:
                # No buttons buildable (or SQL disabled) — degrade to general.
                frame.intent = "general"

        # ── Step 4: Orchestrator → English answer ────────────────────────────
        with Timer() as t:
            try:
                result = await self.orchestrator.process_query(frame)
                english_answer = result.get("answer", "")
            except Exception as e:
                logger.error_with_context(e, {"action": "orchestrator", "client": client_id})
                await self._send_error(ws, f"Processing error: {e}")
                return
        logger.info("ENGLISH ANSWER READY", chars=len(english_answer),
                    flow=result.get("flow"), elapsed_ms=f"{t.elapsed_ms:.0f}ms")

        # ── Step 5: Translate to user language ───────────────────────────────
        with Timer() as t:
            final_answer = await translate_to_user_language(english_answer, lang_type)
        logger.info("TRANSLATION DONE", lang_type=lang_type,
                    translated=(lang_type != "english"), elapsed_ms=f"{t.elapsed_ms:.0f}ms")

        # ── Step 6: Send response ────────────────────────────────────────────
        history.messages.append(ChatMessage(role="assistant", content=final_answer))
        history.updated_at = datetime.now()

        # Frontend contract: a single `query_data` field carrying the
        # post-status-filter rows (what the LLM saw and what's safe to
        # render).  When the status-filter step didn't run — non-SQL
        # flows (NAVIGATION/GENERAL/GREETING) and SQL fast-paths
        # (no-data / not-found / sql-disabled) — `query_results_filtered`
        # is absent from the orchestrator result; we fall back to the raw
        # `query_results` so `query_data` is always the canonical view.
        query_data = self._build_query_data(result, key="query_results_filtered")
        if not query_data:
            query_data = self._build_query_data(result, key="query_results")

        try:
            text_msg = {
                "type":                "text_output",
                "text":                final_answer,
                "is_complete":         True,
                "sources":             _safe_serialize(result.get("sources", [])),
                "flow":                result.get("flow", "SQL"),
                "lang_type":           lang_type,
                "cache_hit":           result.get("cache_hit", False),
                "timestamp":           datetime.now().isoformat(),
                "query_data":          _safe_serialize(query_data),
            }
            await ws.send_text(json.dumps(text_msg, ensure_ascii=False))
        except Exception as e:
            logger.error_with_context(e, {"action": "send_text_output"})
            await ws.send_text(json.dumps({
                "type": "text_output", "text": final_answer,
                "is_complete": True, "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False))

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info("PIPELINE COMPLETE", total_ms=f"{total_ms:.0f}ms",
                    flow=result.get("flow"), lang_type=lang_type,
                    cached=result.get("cache_hit", False))

    @staticmethod
    def _build_query_data(result: dict, key: str = "query_results_filtered") -> dict:
        """
        Build a structured dict of DB rows keyed by table name.

        key  — which result list to read from the orchestrator result dict:
                "query_results_filtered" (default) -> rows after status filter
                                                       (what LLM saw — canonical
                                                       view shipped to frontend)
                "query_results"                    -> raw rows (all statuses);
                                                       used as fallback for flows
                                                       that don't run the filter

        Returns {} for NAVIGATION / GENERAL / GREETING flows (no DB query runs).
        """
        query_data = {}
        for qr in result.get(key, []):
            table = getattr(qr, "table_name", None) or (qr.get("table_name") if isinstance(qr, dict) else None) or ""
            rows  = getattr(qr, "rows", None) or (qr.get("rows") if isinstance(qr, dict) else None) or []
            if table and rows:
                query_data[table] = [dict(row) if isinstance(row, dict) else row for row in rows]
        return query_data

    @staticmethod
    async def _send_error(ws: WebSocket, error: str):
        await ws.send_text(json.dumps({
            "type": "error", "error": error,
            "timestamp": datetime.now().isoformat()
        }, ensure_ascii=False))


_chat_handler: Optional[ChatHandler] = None

def get_chat_handler() -> ChatHandler:
    global _chat_handler
    if _chat_handler is None:
        _chat_handler = ChatHandler()
    return _chat_handler
