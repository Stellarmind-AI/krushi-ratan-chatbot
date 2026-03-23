"""
WebSocket Chat Handler — Krushiratn AI Backend.

Pipeline:
  Step 1: Detect input language (language_processor)
  Step 2: Orchestrator → always generates English answer
  Step 3: Translate English → user's language (Google Translate API)
  Step 4: Send translated answer to client

This clean separation means:
  - All LLMs (SQL gen, answer gen, navigation, general) always work in English
  - Language handling is one centralized step — easy to maintain
  - Adding new languages in future = update translation_service only
"""

import json
import time
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

from app.core.logger import get_logger, get_websocket_logger, Timer
from app.models.chat_models import ChatHistory, ChatMessage
from app.services.agent.orchestrator import get_orchestrator
from app.services.language_processor import get_language_processor
from app.services.translation_service import translate_to_user_language, translate_list_to_user_language
from app.services.agent.confirmation_layer import get_confirmation_layer, ConfirmedIntent

logger       = get_websocket_logger()
pipeline_log = get_logger("pipeline")


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
        self.orchestrator         = get_orchestrator()
        self.language_processor   = get_language_processor()
        self.confirmation_layer   = get_confirmation_layer()   # F1
        self._sessions: dict      = {}
        # F1: stores {session_id: {"query": str, "lang_type": str}}
        # when the pipeline is paused waiting for the user's clarification pick.
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
                    # F1: user picked one of the clarification options
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
        logger.info("TEXT INPUT", text=text[:80])
        await self._run_pipeline(ws, text, session_id, client_id)

    async def _handle_clarification_response(self, ws: WebSocket, data: dict, client_id: str):
        """
        F1 — Called when the front-end sends back the user's chosen option.

        Expected payload:
          {
            "type":       "clarification_response",
            "session_id": "<session>",        # optional
            "intent_key": "crop_price"        # the intent_key from the option the user tapped
          }

        Resumes the paused pipeline with the confirmed intent injected.
        """
        session_id  = data.get("session_id") or client_id
        intent_key  = data.get("intent_key", "").strip()

        if not intent_key:
            await self._send_error(ws, "clarification_response missing intent_key")
            return

        pending = self._pending_clarifications.pop(session_id, None)
        if not pending:
            await self._send_error(
                ws,
                "No pending clarification for this session. Please send your question again.",
            )
            return

        original_query = pending["query"]
        lang_type      = pending["lang_type"]

        logger.info(
            "F1 CLARIFICATION RESOLVED",
            session_id=session_id,
            intent_key=intent_key,
            original_query=original_query[:80],
        )

        keyword_hint = pending.get("keyword_hint", "")

        # Resume the pipeline — pass confirmed_intent and keyword_hint so
        # orchestrator skips tool-selection and SQL is targeted to the keyword.
        await self._run_pipeline(
            ws,
            original_query,
            session_id,
            client_id,
            confirmed_intent=intent_key,
            lang_type_override=lang_type,
            keyword_hint=keyword_hint,
        )

    async def _handle_control(self, ws: WebSocket, data: dict, client_id: str):
        action     = data.get("action")
        session_id = data.get("session_id") or client_id
        if action == "clear_history":
            self._sessions.pop(session_id, None)
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
        confirmed_intent: Optional[str] = None,   # F1: set after user picks clarification option
        lang_type_override: Optional[str] = None, # F1: reuse detected lang from paused query
        keyword_hint: str = "",                   # F1: the matched keyword (e.g. "kapas") for SQL accuracy
    ):
        """
        Full pipeline:
          Step 1: Detect language
          [F1]    Confirmation check — pause & ask user if intent is ambiguous
                  (skipped when confirmed_intent is already set)
          Step 2: Orchestrator → English answer
          Step 3: Translate English → user language (Google Translate)
          Step 4: Send to client
        """
        pipeline_start = time.perf_counter()
        history = self._sessions.setdefault(session_id, ChatHistory(session_id=session_id))
        # Only add user message to history on the first call (not on F1 resume)
        if not confirmed_intent:
            history.messages.append(ChatMessage(role="user", content=user_text))

        # ── Step 1: Language detection ───────────────────────────────────────
        # When resuming after F1 clarification, reuse the already-detected lang.
        if lang_type_override:
            processed_text = user_text
            lang_type      = lang_type_override
        else:
            with Timer() as t:
                processed_text, lang_type = await self.language_processor.process(user_text)
            logger.info("LANGUAGE DETECTED", lang_type=lang_type,
                        original=user_text[:60], elapsed_ms=f"{t.elapsed_ms:.1f}ms")

        # ── F1: Confirmation Layer ───────────────────────────────────────────
        # Only runs on the FIRST pass (confirmed_intent is None).
        # Returns one of:
        #   ConfirmedIntent      → confidence >= 80%, inject intent and skip F1 UI
        #   ClarificationRequest → confidence < 80%, pause and ask user
        #   None                 → no ambiguity, proceed normally
        if not confirmed_intent:
            f1_result = self.confirmation_layer.check(processed_text)

            if isinstance(f1_result, ConfirmedIntent):
                # High-confidence intent — skip F1 UI, inject directly into pipeline
                logger.info(
                    "F1 HIGH CONFIDENCE",
                    intent=f1_result.intent_key,
                    confidence=f"{f1_result.confidence:.0%}",
                    session_id=session_id,
                )
                confirmed_intent = f1_result.intent_key

            elif f1_result is not None:
                # Low-confidence — pipeline paused, ask user
                clarification = f1_result
                self._pending_clarifications[session_id] = {
                    "query":          processed_text,
                    "lang_type":      lang_type,
                    "keyword_hint":   clarification.matched_keyword,
                }
                logger.info(
                    "F1 PIPELINE PAUSED",
                    session_id=session_id,
                    scenario=clarification.scenario,
                    query=processed_text[:60],
                )
                payload = self.confirmation_layer.serialize_request(clarification)
                # Translate question to user's detected language
                payload["question"] = await translate_to_user_language(
                    clarification.question, lang_type
                )
                # Translate option labels to user's detected language
                raw_labels = [opt["label"] for opt in payload["options"]]
                translated_labels = await translate_list_to_user_language(raw_labels, lang_type)
                for i, opt in enumerate(payload["options"]):
                    opt["label"] = translated_labels[i]
                payload["timestamp"] = datetime.now().isoformat()
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
                return  # ← pipeline paused; resumes via _handle_clarification_response

        # ── Step 2: Orchestrator → English answer ────────────────────────────
        with Timer() as t:
            try:
                result = await self.orchestrator.process_query(
                    processed_text,
                    confirmed_intent=confirmed_intent,  # F1: None on first pass; set on resume
                    keyword_hint=keyword_hint,          # F1: matched keyword for SQL targeting
                )
                english_answer = result.get("answer", "")
            except Exception as e:
                logger.error_with_context(e, {"action": "orchestrator", "client": client_id})
                await self._send_error(ws, f"Processing error: {e}")
                return
        logger.info("ENGLISH ANSWER READY", chars=len(english_answer),
                    flow=result.get("flow"), elapsed_ms=f"{t.elapsed_ms:.0f}ms")

        # ── Step 3: Translate to user language ───────────────────────────────
        with Timer() as t:
            final_answer = await translate_to_user_language(english_answer, lang_type)
        logger.info("TRANSLATION DONE", lang_type=lang_type,
                    translated=(lang_type != "english"), elapsed_ms=f"{t.elapsed_ms:.0f}ms")

        # ── Step 4: Send response ────────────────────────────────────────────
        history.messages.append(ChatMessage(role="assistant", content=final_answer))
        history.updated_at = datetime.now()

        kshop_payload = self._build_kshop_payload(result)

        try:
            text_msg = {
                "type":        "text_output",
                "text":        final_answer,
                "is_complete": True,
                "sources":     _safe_serialize(result.get("sources", [])),
                "flow":        result.get("flow", "SQL"),
                "lang_type":   lang_type,
                "cache_hit":   result.get("cache_hit", False),
                "timestamp":   datetime.now().isoformat(),
            }
            if kshop_payload:
                text_msg["kshop_data"] = _safe_serialize(kshop_payload)
            await ws.send_text(json.dumps(text_msg, ensure_ascii=False))
        except Exception as e:
            logger.error_with_context(e, {"action": "send_text_output"})
            await ws.send_text(json.dumps({
                "type": "text_output", "text": final_answer,
                "is_complete": True, "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False))

        if kshop_payload:
            try:
                await ws.send_text(json.dumps({
                    "type": "kshop_products", "button_type": "product_view",
                    "products": _safe_serialize(kshop_payload), "count": len(kshop_payload),
                    "timestamp": datetime.now().isoformat(),
                }, ensure_ascii=False))
            except Exception as e:
                logger.warning(f"kshop_products send failed: {e}")

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info("PIPELINE COMPLETE", total_ms=f"{total_ms:.0f}ms",
                    flow=result.get("flow"), lang_type=lang_type,
                    cached=result.get("cache_hit", False))

    @staticmethod
    def _build_kshop_payload(result: dict) -> list:
        products = []
        for qr in result.get("query_results", []):
            table = getattr(qr, "table_name", None) or (qr.get("table_name") if isinstance(qr, dict) else None) or ""
            rows  = getattr(qr, "rows", None) or (qr.get("rows") if isinstance(qr, dict) else None) or []
            if "kshop" in table.lower():
                for row in rows:
                    p = dict(row) if isinstance(row, dict) else {}
                    p["button_type"] = "product_view"
                    products.append(p)
        return products

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