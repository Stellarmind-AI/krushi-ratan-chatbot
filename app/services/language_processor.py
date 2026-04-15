"""
Language Processor — Language detection via Google Translate API.

Three input types:
  1. Gujarati script  (Unicode U+0A80–U+0AFF)
     → normalize unicode → pass to pipeline as-is
     → DB has Gujarati, query generator searches Gujarati directly

  2. Romanized Gujarati  (kapas, bhav, kevi rite...)
     → pass to pipeline as-is
     → query generator LLM handles transliteration internally
       (it already knows kapas=કપાસ, bhav=ભાવ etc.)

  3. English
     → pass to pipeline as-is
     → query generator searches both EN+GU in DB already

Detection: Google Translate detect language API.
Fallback:  Unicode range check (if Google API unavailable/fails).
"""

import re
import unicodedata
import httpx
from typing import Tuple, Optional
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("language_processor")

# ─────────────────────────────────────────────────────────────────────────────
# Indic unicode ranges — used as fallback detection only
# ─────────────────────────────────────────────────────────────────────────────
_GUJARATI_RANGE   = (0x0A80, 0x0AFF)   # Gujarati script
_DEVANAGARI_RANGE = (0x0900, 0x097F)   # Devanagari script (Hindi, Marathi, etc.)
_SCRIPT_THRESHOLD = 0.15   # 15% chars in a given script = that script's input


def _detect_by_unicode(text: str) -> Optional[str]:
    """
    Fallback detection using unicode range.
    Returns "gujarati_script", "hindi_script", "english", or None (unknown).
    """
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "english"

    gujarati_chars = sum(
        1 for c in text
        if _GUJARATI_RANGE[0] <= ord(c) <= _GUJARATI_RANGE[1]
    )
    if (gujarati_chars / total_alpha) >= _SCRIPT_THRESHOLD:
        return "gujarati_script"

    devanagari_chars = sum(
        1 for c in text
        if _DEVANAGARI_RANGE[0] <= ord(c) <= _DEVANAGARI_RANGE[1]
    )
    if (devanagari_chars / total_alpha) >= _SCRIPT_THRESHOLD:
        return "hindi_script"   # Devanagari → assume Hindi (most common)

    return None  # Cannot determine — could be Romanized Gujarati or English


async def detect_language_google(text: str) -> str:
    """
    Detect language using Google Translate detect API.

    Returns one of:
      "gujarati_script"    — Google detected "gu" (Gujarati)
      "hindi_script"       — Google detected "hi" (Hindi, Devanagari script)
      "romanized_gujarati" — Google detected "en" but text has Gujarati signal words
                             (Romanized Gujarati looks like English to Google)
      "english"            — Google detected "en" with no Gujarati signals

    Falls back to unicode check if API key not configured or request fails.
    """
    text = text.strip()
    if not text:
        return "english"

    # ── Try Google Translate detect API ─────────────────────────────────────
    api_key = getattr(settings, "GOOGLE_TRANSLATE_API_KEY", None)

    if api_key:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.post(
                    "https://translation.googleapis.com/language/translate/v2/detect",
                    params={"key": api_key},
                    json={"q": text[:200]},   # limit to 200 chars — detect needs only a snippet
                )
                resp.raise_for_status()
                data = resp.json()
                detected = (
                    data.get("data", {})
                        .get("detections", [[{}]])[0][0]
                        .get("language", "en")
                )
                confidence = (
                    data.get("data", {})
                        .get("detections", [[{}]])[0][0]
                        .get("confidence", 0.0)
                )
                logger.debug(f"Google detected: {detected} (confidence={confidence:.2f})")

                if detected == "gu":
                    return "gujarati_script"

                # Hindi → return as hindi_script (THE FIX)
                if detected == "hi":
                    return "hindi_script"

                # Google says "en" — could be Romanized Gujarati
                # Check for Gujarati-specific Romanized signal words
                if detected == "en" and _has_romanized_signals(text.lower()):
                    return "romanized_gujarati"

                # Other Indian scripts (Marathi/Punjabi/Bengali/etc.) → treat as Gujarati
                # (existing behavior preserved — change later if you add support for those)
                if detected in ("mr", "pa", "bn", "te", "ta", "kn", "ml", "ur"):
                    return "gujarati_script"

                return "english"

        except httpx.TimeoutException:
            logger.warning("Google Translate detect: timeout — using unicode fallback")
        except httpx.HTTPStatusError as e:
            logger.warning(f"Google Translate detect: HTTP {e.response.status_code} — using unicode fallback")
        except Exception as e:
            logger.warning(f"Google Translate detect failed: {e} — using unicode fallback")

    else:
        logger.debug("GOOGLE_TRANSLATE_API_KEY not set — using unicode fallback")

    # ── Unicode fallback ─────────────────────────────────────────────────────
    unicode_result = _detect_by_unicode(text)
    if unicode_result == "gujarati_script":
        return "gujarati_script"
    if unicode_result == "hindi_script":
        return "hindi_script"

    # For Latin-script text, check Romanized signals
    if _has_romanized_signals(text.lower()):
        return "romanized_gujarati"

    return "english"


# ─────────────────────────────────────────────────────────────────────────────
# Romanized Gujarati signal words
# Used when Google says "en" to distinguish Romanized Gujarati from real English.
#
# TIERED to prevent false positives on English questions about the app:
#
#   STRONG signals — Gujarati-only verbs/particles/question words.
#                    Almost never appear in English. ONE match triggers.
#
#   WEAK   signals — crop names, farming nouns. Common in Indian English too
#                    (e.g. "what is mandi bhav"). Need 2+ matches to trigger.
#
#   REMOVED — App/brand proper nouns ("krushiratn", "kshop", "suvidha", "krushi")
#             must NEVER be signals. English users naturally type the app name
#             (e.g. "show crops in krushiratn") and that should stay English.
# ─────────────────────────────────────────────────────────────────────────────
_ROMANIZED_STRONG_SIGNALS = {
    # Verbs / connectors only Gujarati uses
    "che", "chhe", "karvu", "karo", "kevi rite", "kevi",
    "joi", "joiyu", "levu", "vechuv", "vecho", "batao",
    # Pronouns / particles
    "mare", "maro", "mane", "tame",
    # Question words
    "shu", "kyay", "kem", "ketla",
}

_ROMANIZED_WEAK_SIGNALS = {
    # Crop / farming nouns — also used in Indian English, so require 2+
    "bhav", "mandi", "samachar", "khabar",
    "kapas", "bajri", "magfali", "bhens", "balwan", "balwaan",
    "khedut", "kisan", "jamin",
}

def _has_romanized_signals(text_lower: str) -> bool:
    """
    Return True if text contains Romanized Gujarati signals.
    - Any STRONG signal       → True
    - 2 or more WEAK signals  → True
    Otherwise                 → False (treat as English).
    """
    # Strong signals: any single match flips it
    for signal in _ROMANIZED_STRONG_SIGNALS:
        if re.search(r"(?<![a-z])" + re.escape(signal) + r"(?![a-z])", text_lower):
            return True

    # Weak signals: count distinct matches, require 2+
    weak_hits = 0
    for signal in _ROMANIZED_WEAK_SIGNALS:
        if re.search(r"(?<![a-z])" + re.escape(signal) + r"(?![a-z])", text_lower):
            weak_hits += 1
            if weak_hits >= 2:
                return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Gujarati unicode normalizer
# ─────────────────────────────────────────────────────────────────────────────

def normalize_gujarati_script(text: str) -> str:
    """
    Normalize Gujarati unicode text.
    Fixes: NFC composition, zero-width chars, whitespace.
    Does NOT translate or change any words.
    """
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width space and BOM (keep ZWJ — used in Gujarati conjuncts)
    for zw in ["\u200b", "\ufeff"]:
        text = text.replace(zw, "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_hindi_script(text: str) -> str:
    """
    Normalize Hindi (Devanagari) unicode text.
    Same operations as Gujarati: NFC + zero-width strip + whitespace.
    Kept as a separate function for clarity / future Hindi-specific tweaks.
    """
    text = unicodedata.normalize("NFC", text)
    for zw in ["\u200b", "\ufeff"]:
        text = text.replace(zw, "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

async def process_input(text: str) -> Tuple[str, str]:
    """
    Detect language and process text.

    Returns: (processed_text, lang_type)
      lang_type: "gujarati_script" | "romanized_gujarati" | "english"

    Processing:
      gujarati_script    → normalize unicode only
      romanized_gujarati → pass as-is (LLM in query generator handles transliteration)
      english            → pass as-is
    """
    if not text or not text.strip():
        return text, "english"

    lang_type = await detect_language_google(text)

    if lang_type == "gujarati_script":
        processed = normalize_gujarati_script(text)
    elif lang_type == "hindi_script":
        processed = normalize_hindi_script(text)
    else:
        # romanized_gujarati and english both pass through unchanged
        # query generator LLM handles both: it searches EN+GU and
        # knows common Romanized→Gujarati mappings from its training
        processed = text.strip()

    logger.info(f"Language: {lang_type} | Input: {text[:50]!r}")
    return processed, lang_type


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

class LanguageProcessor:
    """Async language processor — detection via Google Translate API."""

    async def process(self, text: str) -> Tuple[str, str]:
        """Process input. Returns (processed_text, language_type)."""
        return await process_input(text)

    async def detect(self, text: str) -> str:
        """Detect language type only."""
        return await detect_language_google(text)


_instance = None

def get_language_processor() -> LanguageProcessor:
    global _instance
    if _instance is None:
        _instance = LanguageProcessor()
    return _instance