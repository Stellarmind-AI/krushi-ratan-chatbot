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
# Gujarati unicode range — used as fallback detection only
# ─────────────────────────────────────────────────────────────────────────────
_GUJARATI_RANGE = (0x0A80, 0x0AFF)
_GUJARATI_THRESHOLD = 0.15   # 15% Gujarati chars = Gujarati script input


def _detect_by_unicode(text: str) -> Optional[str]:
    """
    Fallback detection using unicode range.
    Returns "gujarati_script", "english", or None (unknown).
    """
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "english"

    gujarati_chars = sum(
        1 for c in text
        if _GUJARATI_RANGE[0] <= ord(c) <= _GUJARATI_RANGE[1]
    )
    if (gujarati_chars / total_alpha) >= _GUJARATI_THRESHOLD:
        return "gujarati_script"

    return None  # Cannot determine — could be Romanized Gujarati or English


async def detect_language_google(text: str) -> str:
    """
    Detect language using Google Translate detect API.

    Returns one of:
      "gujarati_script"    — Google detected "gu" (Gujarati)
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

                # Google says "en" — could be Romanized Gujarati
                # Check for Gujarati-specific Romanized signal words
                if detected == "en" and _has_romanized_signals(text.lower()):
                    return "romanized_gujarati"

                # Hindi/Marathi/other Indian scripts → treat as non-English
                if detected in ("hi", "mr", "pa", "bn", "te", "ta", "kn", "ml", "ur"):
                    return "gujarati_script"  # handle similarly — pass to pipeline as-is

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

    # For Latin-script text, check Romanized signals
    if _has_romanized_signals(text.lower()):
        return "romanized_gujarati"

    return "english"


# ─────────────────────────────────────────────────────────────────────────────
# Romanized Gujarati signal words
# Used when Google says "en" to distinguish Romanized Gujarati from real English
# ─────────────────────────────────────────────────────────────────────────────
_ROMANIZED_SIGNALS = {
    # Verbs / connectors only Gujarati uses
    "che", "chhe", "karvu", "karo", "kevi rite", "kevi",
    "joi", "joiyu", "levu", "vechuv", "vecho", "batao",
    # Pronouns/particles
    "mare", "maro", "mane", "tame",
    # Key nouns not in English
    "bhav", "mandi", "samachar", "khabar", "krushi",
    "kapas", "bajri", "magfali", "bhens", "balwan", "balwaan",
    "khedut", "kisan", "jamin",
    # Question words
    "shu", "kyay", "kem", "ketla",
    # App words
    "kshop", "krushiratn", "suvidha",
}

def _has_romanized_signals(text_lower: str) -> bool:
    """Return True if text contains Romanized Gujarati signal words."""
    for signal in _ROMANIZED_SIGNALS:
        if re.search(r"(?<![a-z])" + re.escape(signal) + r"(?![a-z])", text_lower):
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