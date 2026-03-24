"""
Translation Service — Google Translate API with LLM fallback.

Translates LLM-generated English answers into the user's detected language.
This is the ONLY place where translation happens in the pipeline.

Flow:
  User input (any language)
    → language_processor detects lang_type
    → orchestrator generates English answer
    → translation_service translates English → user's language
    → chat_handler sends translated answer to client

Priority:
  1. Google Translate REST API (if GOOGLE_TRANSLATE_API_KEY is set in .env)
  2. Groq LLM translation fallback (uses the same LLM already running)
  3. Return original English (if both fail)

Supported lang_type → Google language code:
  "gujarati_script"    → "gu"
  "romanized_gujarati" → "gu"  (respond in proper Gujarati script)
  "english"            → None  (no translation needed)
"""

import httpx
from typing import Optional, List
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("translation_service")

# Map lang_type → Google target language code
_LANG_TYPE_TO_GOOGLE: dict = {
    "gujarati_script":    "gu",
    "romanized_gujarati": "gu",
    "english":            None,
}

_LANG_CODE_TO_NAME: dict = {
    "gu": "Gujarati",
    "hi": "Hindi",
}

_GOOGLE_TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"


# ─────────────────────────────────────────────────────────────────────────────
# LLM fallback translation
# Used when GOOGLE_TRANSLATE_API_KEY is not configured.
# Reuses the same Groq instance already running — zero extra cost.
# ─────────────────────────────────────────────────────────────────────────────

async def _translate_with_llm(text: str, target_lang_code: str) -> str:
    """
    Translate text using the LLM (Groq) when Google API key is not available.
    Batches multiple lines together to minimize token usage.
    """
    from app.services.llm.manager import get_llm_manager
    from app.models.chat_models import LLMMessage

    lang_name = _LANG_CODE_TO_NAME.get(target_lang_code, "Gujarati")

    try:
        llm = get_llm_manager()
        response = await llm.generate(
            messages=[
                LLMMessage(role="system", content=(
                    f"You are a professional translator. "
                    f"Translate the following text to {lang_name}. "
                    f"Rules:\n"
                    f"- Return ONLY the translated text, nothing else.\n"
                    f"- Keep all numbers, proper nouns, and product names unchanged.\n"
                    f"- Keep emojis as-is.\n"
                    f"- Do NOT add any explanation or preamble."
                )),
                LLMMessage(role="user", content=text),
            ],
            temperature=0.0,
            max_tokens=800,
        )
        translated = response.content.strip()
        logger.info(f"LLM translated en→{target_lang_code} | {len(text)} chars → {len(translated)} chars")
        print(f"\n{'─'*60}", flush=True)
        print(f"🌐 LLM TRANSLATION en→{target_lang_code}", flush=True)
        print(f"  INPUT:  {text}", flush=True)
        print(f"  OUTPUT: {translated}", flush=True)
        print(f"{'─'*60}", flush=True)
        return translated
    except Exception as e:
        logger.warning(f"LLM translation failed: {e} — returning original text")
        print(f"  ❌ LLM TRANSLATION FAILED: {e} — returning original", flush=True)
        return text


async def _translate_list_with_llm(items: List[str], target_lang_code: str) -> List[str]:
    """
    Translate a list of short strings in ONE LLM call.
    Uses a numbered format so the LLM returns them in the same order.
    Preserves emojis and proper nouns.
    """
    from app.services.llm.manager import get_llm_manager
    from app.models.chat_models import LLMMessage

    lang_name = _LANG_CODE_TO_NAME.get(target_lang_code, "Gujarati")
    if not items:
        return items

    # Build numbered input: "1. Check Wheat mandi price\n2. Buy Wheat from K-Shop\n..."
    numbered = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    try:
        llm = get_llm_manager()
        response = await llm.generate(
            messages=[
                LLMMessage(role="system", content=(
                    f"Translate each numbered line to {lang_name}. "
                    f"Return ONLY the numbered lines in the same format. "
                    f"Keep numbers, emojis, crop names, and product names unchanged. "
                    f"Do NOT add explanation."
                )),
                LLMMessage(role="user", content=numbered),
            ],
            temperature=0.0,
            max_tokens=400,
        )
        raw = response.content.strip()
        # Parse back: split by lines, strip "N. " prefix
        translated_items = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove leading "1. " / "2. " etc.
            if len(line) > 2 and line[0].isdigit() and line[1] in (".", ")"):
                line = line[2:].strip()
            elif len(line) > 3 and line[0].isdigit() and line[1].isdigit() and line[2] in (".", ")"):
                line = line[3:].strip()
            translated_items.append(line)

        # Safety: if parse failed, return originals
        if len(translated_items) != len(items):
            logger.warning(f"LLM list translation count mismatch {len(items)} vs {len(translated_items)} — using originals")
            print(f"  ⚠️ LIST TRANSLATE MISMATCH: expected {len(items)}, got {len(translated_items)}", flush=True)
            return items

        logger.info(f"LLM translated {len(items)} option labels en→{target_lang_code}")
        print(f"\n{'─'*60}", flush=True)
        print(f"🌐 LLM LIST TRANSLATION en→{target_lang_code} ({len(items)} items)", flush=True)
        for i, (orig, trans) in enumerate(zip(items, translated_items)):
            print(f"  {i+1}. {orig!r} → {trans!r}", flush=True)
        print(f"{'─'*60}", flush=True)
        return translated_items

    except Exception as e:
        logger.warning(f"LLM list translation failed: {e} — returning originals")
        return items


# ─────────────────────────────────────────────────────────────────────────────
# Google Translate
# ─────────────────────────────────────────────────────────────────────────────

async def _translate_with_google(text: str, target_lang: str, api_key: str) -> Optional[str]:
    """Returns translated text or None on failure."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                _GOOGLE_TRANSLATE_URL,
                params={"key": api_key},
                json={"q": text, "source": "en", "target": target_lang, "format": "text"},
            )
            resp.raise_for_status()
            data = resp.json()
            translated = (
                data.get("data", {})
                    .get("translations", [{}])[0]
                    .get("translatedText", None)
            )
            if translated:
                logger.info(f"Google translated en→{target_lang} | {len(text)} → {len(translated)} chars")
                print(f"\n{'─'*60}", flush=True)
                print(f"🌐 GOOGLE TRANSLATION en→{target_lang}", flush=True)
                print(f"  INPUT:  {text}", flush=True)
                print(f"  OUTPUT: {translated}", flush=True)
                print(f"{'─'*60}", flush=True)
            return translated
    except httpx.TimeoutException:
        logger.warning("Google Translate: timeout")
    except httpx.HTTPStatusError as e:
        logger.warning(f"Google Translate: HTTP {e.response.status_code}")
    except Exception as e:
        logger.warning(f"Google Translate failed: {e}")
    return None


async def _translate_list_with_google(items: List[str], target_lang: str, api_key: str) -> Optional[List[str]]:
    """Translate a list of strings via Google Translate. Returns None on failure."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                _GOOGLE_TRANSLATE_URL,
                params={"key": api_key},
                json={"q": items, "source": "en", "target": target_lang, "format": "text"},
            )
            resp.raise_for_status()
            data = resp.json()
            translations = data.get("data", {}).get("translations", [])
            result = [t.get("translatedText", items[i]) for i, t in enumerate(translations)]
            if len(result) == len(items):
                print(f"\n{'─'*60}", flush=True)
                print(f"🌐 GOOGLE LIST TRANSLATION en→{target_lang} ({len(items)} items)", flush=True)
                for i, (orig, trans) in enumerate(zip(items, result)):
                    print(f"  {i+1}. {orig!r} → {trans!r}", flush=True)
                print(f"{'─'*60}", flush=True)
                return result
    except Exception as e:
        logger.warning(f"Google Translate list failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def translate_to_user_language(text: str, lang_type: str) -> str:
    """
    Translate English text to user's language.

    Priority: Google Translate → LLM fallback → original English.

    Args:
        text:      English text generated by LLM
        lang_type: "gujarati_script" | "romanized_gujarati" | "english"

    Returns:
        Translated text, or original if no translation needed / both fail.
    """
    target_lang = _LANG_TYPE_TO_GOOGLE.get(lang_type)
    if not target_lang:
        print(f"🌐 TRANSLATE SKIPPED — lang_type={lang_type} (English, no translation needed)", flush=True)
        return text   # English — no translation needed
    if not text or not text.strip():
        return text

    print(f"\n{'='*60}", flush=True)
    print(f"🌐 TRANSLATE TO USER LANGUAGE: {lang_type} → {target_lang}", flush=True)
    print(f"  📥 ENGLISH INPUT ({len(text)} chars):", flush=True)
    print(f"    {text}", flush=True)
    print(f"{'='*60}", flush=True)

    api_key = getattr(settings, "GOOGLE_TRANSLATE_API_KEY", None)

    # Try Google first (fast, accurate)
    if api_key:
        result = await _translate_with_google(text, target_lang, api_key)
        if result:
            return result

    # LLM fallback (no extra API key required)
    return await _translate_with_llm(text, target_lang)


async def translate_list_to_user_language(items: List[str], lang_type: str) -> List[str]:
    """
    Translate a list of short strings (e.g. clarification option labels) to user's language.
    Batches all items into a single API/LLM call.

    Returns:
        Translated list, same length as input.
        Falls back to original items on any failure.
    """
    target_lang = _LANG_TYPE_TO_GOOGLE.get(lang_type)
    if not target_lang:
        print(f"🌐 LIST TRANSLATE SKIPPED — lang_type={lang_type} (English)", flush=True)
        return items   # English — no translation needed
    if not items:
        return items

    print(f"\n{'='*60}", flush=True)
    print(f"🌐 TRANSLATE LIST: {lang_type} → {target_lang} ({len(items)} items)", flush=True)
    for i, item in enumerate(items):
        print(f"  📥 [{i+1}]: {item}", flush=True)
    print(f"{'='*60}", flush=True)

    api_key = getattr(settings, "GOOGLE_TRANSLATE_API_KEY", None)

    # Try Google first
    if api_key:
        result = await _translate_list_with_google(items, target_lang, api_key)
        if result:
            return result

    # LLM fallback
    return await _translate_list_with_llm(items, target_lang)