"""
Confirmation Layer (F1) — LLM-based multilingual intent classifier.

Handles queries in:
  - English             (tractor price, I want cow, mango bhav)
  - Romanized Gujarati  (kevi rite, kapas bhav, mane rotavater joie)
  - Gujarati script     (ઘઉં ભાવ, ટ્રેક્ટર, ગાય)

WHY LLM INSTEAD OF KEYWORDS:
  Keyword/regex matching breaks constantly as user language varies — especially
  with 7500+ users asking in Gujarati, Romanized Gujarati, and English.
  A single focused LLM call handles all variations naturally and can reason
  about domain context (e.g. cow = always buy_sell, never kshop).

THREE BEHAVIOURS (same as before — only the DETECTION mechanism changed):

1. SKIP (None returned)
   Navigation (how-to, steps, kevi rite), greetings, or general app info.
   Route agent already handles these correctly — F1 must not intercept.

2. CONFIRMED INTENT (ConfirmedIntent returned)
   Query is unambiguous — single clear domain detected with high confidence.
   Orchestrator skips table selection and uses the pre-confirmed tables.

3. CLARIFICATION REQUEST (ClarificationRequest returned)
   Query is genuinely ambiguous — 2+ possible domains.
   User is shown buttons to pick the intended domain.

COST:
  Single Groq LLM call, max_tokens=80 (~600 tokens total, ~150ms).
  Same pattern as route_agent — negligible overhead.

IMPORTANT — ASYNC CHANGE:
  check() is now async. The caller (chat_handler) must await it:
    result = await confirmation_layer.check(user_query)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Union

from app.core.logger import get_logger
from app.services.llm.manager import get_llm_manager
from app.models.chat_models import LLMMessage

logger = get_logger("confirmation_layer")


# -----------------------------------------------------------------------------
# Data structures — UNCHANGED (downstream code depends on these)
# -----------------------------------------------------------------------------

@dataclass
class ClarificationOption:
    label:      str
    emoji:      str
    intent_key: str
    domain:     str


@dataclass
class ClarificationRequest:
    """Pipeline paused — send options to user."""
    question:        str
    options:         List[ClarificationOption]
    scenario:        str
    matched_keyword: str = ""


@dataclass
class ConfirmedIntent:
    """Single clear intent detected — skip F1 UI, inject intent directly."""
    intent_key: str
    confidence: float
    domain:     str


# -----------------------------------------------------------------------------
# Intent -> table mapping — UNCHANGED
# -----------------------------------------------------------------------------

INTENT_TO_TABLES: Dict[str, List[str]] = {
    "crop_price":       ["query_products", "query_sub_categories", "query_yards", "query_cities", "query_talukas", "query_weights"],
    "kshop_product":    ["query_kshop_products", "query_kshop_companies", "query_kshop_categories", "query_kshop_weights"],
    "buy_sell_product": ["query_buy_sell_products", "query_buy_sell_categories", "query_users"],
    "seed_info":        ["query_seeds", "query_sub_categories"],
    "local_news":       ["query_news", "query_cities", "query_talukas", "query_states"],
    "video_search":     ["query_video_posts", "query_users", "query_video_categories"],
    "equipment_kshop":  ["query_kshop_products", "query_kshop_companies", "query_kshop_categories", "query_kshop_weights"],
    "equipment_used":   ["query_buy_sell_products", "query_buy_sell_categories", "query_users"],
}

INTENT_TO_PROMPT_NOTE: Dict[str, str] = {
    "crop_price":       "User confirmed: CROP MARKET PRICES at mandi/yard. Use only crop price tables.",
    "kshop_product":    "User confirmed: K-SHOP PRODUCTS (farm equipment/supplies). Use only kshop tables.",
    "buy_sell_product": "User confirmed: BUY/SELL MARKETPLACE listings. Use only buy_sell tables.",
    "seed_info":        "User confirmed: SEED/VARIETY information. Use only seeds table.",
    "local_news":       "User confirmed: LOCAL AGRICULTURAL NEWS. Use only news table.",
    "video_search":     "User confirmed: FARMING VIDEOS. Use only video_posts table.",
    "equipment_kshop":  "User confirmed: NEW EQUIPMENT from K-Shop. Use only kshop tables.",
    "equipment_used":   "User confirmed: USED/SECOND-HAND EQUIPMENT from Buy/Sell. Use only buy_sell tables.",
}

_VALID_INTENTS   = set(INTENT_TO_TABLES.keys())
_VALID_SCENARIOS = {"equipment", "price", "crop", "product", "location"}


# -----------------------------------------------------------------------------
# Option ordering helpers — still used by option builders below
# -----------------------------------------------------------------------------

_BUY_HINTS = [
    "levu", "kharidi", "kharido", "buy", "purchase", "joiye", "joiyu", "levo", "apo",
    "levu", "levo",
    "levun",
    "mane joie",
    "mane joiyu",
    "joie",
    "joiyu",
    "apo",
    "apavi",
]
_SELL_HINTS = [
    "vechuv", "vecho", "sell", "vechan", "sale", "muku", "mukuv",
    "vecchuv",
    "sathe",
]
_SEED_HINTS = [
    "seed", "bij", "variety", "nasal",
]

_CROPS_WITH_SEED_DATA: set = {
    "wheat", "ghau", "gahu", "kapas", "cotton",
    "bajra", "bajri", "bajro", "jowar", "jwari", "jwar",
    "corn", "maize", "makai", "mung", "moong",
    "chana", "ghana", "channa", "tal", "sesame",
    "soybean", "soya", "rice crop", "chaval",
    "magfali", "groundnut", "moongfali",
}

_CROPS_PRICE_ONLY: set = {
    "onion", "dungli", "kanda",
    "tomato", "tameta",
    "potato", "bataka", "bateta",
    "garlic", "lasan",
    "sugarcane", "sherdio",
    "tuveral", "tuver",
    "adadal", "adad",
}


# -----------------------------------------------------------------------------
# Option builders — UNCHANGED logic
# -----------------------------------------------------------------------------

def _build_crop_options(kw: str, q: str) -> List[ClarificationOption]:
    k      = kw.capitalize() if kw else "Crop"
    kw_low = kw.lower()
    has_sell = any(h in q for h in _SELL_HINTS)
    has_seed = any(h in q for h in _SEED_HINTS)

    _GU_TO_EN = {
        "kapas": "kapas", "cotton": "kapas",
        "wheat": "wheat", "ghau": "wheat", "gahu": "wheat",
        "bajra": "bajra", "bajri": "bajra", "jowar": "jowar",
        "corn": "corn", "maize": "corn", "makai": "corn",
        "mung": "mung", "moong": "mung", "chana": "chana",
        "tal": "tal", "sesame": "tal",
        "chaval": "chaval", "soybean": "soybean", "soya": "soybean",
        "magfali": "magfali", "groundnut": "magfali", "moongfali": "magfali",
        "onion": "onion", "dungli": "onion", "kanda": "onion",
        "tomato": "tomato", "tameta": "tomato",
        "potato": "potato", "bataka": "potato", "bateta": "potato",
        "garlic": "lasan", "lasan": "lasan",
        "sugarcane": "sugarcane", "sherdio": "sugarcane",
    }
    kw_check = _GU_TO_EN.get(kw_low, kw_low)

    opt_price = ClarificationOption(f"Check {k} mandi price",      "📈", "crop_price",       "crop_price")
    opt_sell  = ClarificationOption(f"View {k} buy/sell listings", "📦", "buy_sell_product", "buy_sell")

    has_seed_data = kw_low in _CROPS_WITH_SEED_DATA or kw_check in _CROPS_WITH_SEED_DATA
    opt_seed = ClarificationOption(f"{k} seed variety info", "🌱", "seed_info", "seeds") if has_seed_data else None

    if kw_low in _CROPS_PRICE_ONLY or kw_check in _CROPS_PRICE_ONLY:
        opt_seed = None

    if has_sell:
        opts = [opt_sell, opt_price]
        if opt_seed and has_seed:
            opts.append(opt_seed)
    elif has_seed and opt_seed:
        opts = [opt_seed, opt_price, opt_sell]
    else:
        opts = [opt_price, opt_sell]
        if opt_seed:
            opts.append(opt_seed)

    return [o for o in opts if o is not None]


def _build_product_options(kw: str, q: str) -> List[ClarificationOption]:
    has_sell = any(h in q for h in _SELL_HINTS)
    has_buy  = any(h in q for h in _BUY_HINTS)

    opt_kshop   = ClarificationOption("K-Shop (new farm equipment & supplies)", "🏪", "kshop_product",    "kshop")
    opt_buysell = ClarificationOption("Buy/Sell marketplace (farmer listings)",  "🔄", "buy_sell_product", "buy_sell")
    opt_price   = ClarificationOption("Mandi / yard crop prices",                "📊", "crop_price",       "crop_price")
    opt_seeds   = ClarificationOption("Crop seeds and varieties",                "🌾", "seed_info",        "seeds")

    if has_sell:
        return [opt_buysell, opt_price, opt_kshop]
    if has_buy:
        return [opt_kshop, opt_buysell, opt_price, opt_seeds]
    return [opt_kshop, opt_buysell, opt_price, opt_seeds]


def _build_price_options(kw: str, q: str) -> List[ClarificationOption]:
    label_crop = f"{kw.capitalize()} price at mandi / yard" if kw else "Crop price at mandi / yard"
    return [
        ClarificationOption(label_crop,               "📊", "crop_price",       "crop_price"),
        ClarificationOption("K-Shop product price",   "🛒", "kshop_product",    "kshop"),
        ClarificationOption("Buy/Sell listing price", "💰", "buy_sell_product", "buy_sell"),
    ]


def _build_equipment_options(kw: str, q: str) -> List[ClarificationOption]:
    k = kw.capitalize() if kw else "Equipment"
    has_used = any(h in q for h in [
        "used", "second hand", "juno", "purano", "old",
        "junum", "junun", "juna",
    ])
    opt_new  = ClarificationOption(f"New {k} from K-Shop",              "🏪", "equipment_kshop", "kshop")
    opt_used = ClarificationOption(f"Used {k} on Buy/Sell marketplace", "🔄", "equipment_used",  "buy_sell")
    return [opt_used, opt_new] if has_used else [opt_new, opt_used]


def _build_location_options(kw: str, q: str) -> List[ClarificationOption]:
    k = kw.capitalize() if kw else "this location"
    opt_price = ClarificationOption(f"Crop prices near {k}",       "📈", "crop_price",       "crop_price")
    opt_news  = ClarificationOption(f"Agricultural news from {k}", "📰", "local_news",       "news")
    opt_sell  = ClarificationOption(f"Buy/Sell listings in {k}",   "🏘️", "buy_sell_product", "buy_sell")
    return [opt_news, opt_price, opt_sell]


# -----------------------------------------------------------------------------
# Scenario -> question text and builder
# -----------------------------------------------------------------------------

_SCENARIO_QUESTIONS: Dict[str, str] = {
    "equipment": "Are you looking for new or used {keyword}?",
    "price":     "Which price are you asking about?",
    "crop":      "What would you like to know about {keyword}?",
    "product":   "Which section are you looking in?",
    "location":  "What are you looking for related to '{keyword}'?",
}

_SCENARIO_BUILDERS = {
    "equipment": _build_equipment_options,
    "price":     _build_price_options,
    "crop":      _build_crop_options,
    "product":   _build_product_options,
    "location":  _build_location_options,
}


# -----------------------------------------------------------------------------
# F1 system prompt — tight, domain-grounded, ~500 tokens
# -----------------------------------------------------------------------------

_F1_SYSTEM = (
    "You are F1, the intent pre-classifier for Krushi Ratn — a Gujarati farming marketplace app.\n"
    "Understand queries in English, Romanized Gujarati (bhav, kapas, kevi rite, mane joie), and Gujarati script (ભાવ, કપાસ, ઘઉં).\n"
    "\n"
    "DATABASE DOMAINS:\n"
    "* crop_price     — Mandi/yard market prices for ALL crops: wheat/ghau/ઘઉં, cotton/kapas/કપાસ, mango/કેરી, onion/ડુંગળી, tomato, potato, groundnut, bajra, vegetables, fruits, ALL farm produce\n"
    "* kshop_product  — NEW farm equipment sold by app store: tractor, water pump, sprayer, thresher, seeder, weeder, engine, cultivator, battery sprayer, jatka machine, flashlight, tools\n"
    "* buy_sell_product — Farmer-listed items: USED equipment + ALL ANIMALS (cow/ગાય, buffalo/ભેંસ, goat/બકરી, horse/ઘોડો, camel/ઊંટ, sheep/ઘેટું, ox/બળદ, bull) + used tractors + any farmer-sold item\n"
    "* local_news     — Agricultural news / samachar / ખબર\n"
    "* video_search   — Farming educational videos / વિડિઓ\n"
    "* seed_info      — Crop seed varieties / bij / બીજ\n"
    "\n"
    "CLASSIFICATION RULES (apply top-down, first match wins):\n"
    "\n"
    "SKIP — no database query needed, return {\"decision\":\"skip\"}:\n"
    "  How-to/steps: how to, how do i, how i [verb], kevi rite, kevi ret, kem karvu, steps, guide, register, track, cancel, upload, login\n"
    "  Sell process: how i sell, how to sell, kevi rite vechuv, pak vechuv, how to list\n"
    "  General info: what is krushi ratn, is app free, app features, greetings (hello/namaste/hi/kem cho)\n"
    "\n"
    "CLEAR — single unambiguous intent, return {\"decision\":\"clear\",\"intent\":\"<value>\",\"keyword\":\"<subject>\"}:\n"
    "  Animal name (cow/buffalo/goat/horse/camel/ox/sheep/bull/ગાય/ભેંસ/બકરી/ઘોડો/ઊંટ/ઘેટું/બળદ) → buy_sell_product [NEVER kshop]\n"
    "  Crop/vegetable/fruit name alone → crop_price\n"
    "  Explicit kshop/k-shop/k shop/k-store/કે-શોપ → kshop_product\n"
    "  Explicit buy sell/buysell/marketplace/vechuv/વેચવું → buy_sell_product\n"
    "  news/samachar/ખબર/ન્યૂઝ → local_news\n"
    "  video/વિડિઓ/watch → video_search\n"
    "  seed/bij/variety/બીજ with a crop → seed_info\n"
    "\n"
    "AMBIGUOUS — user must choose, return {\"decision\":\"ambiguous\",\"scenario\":\"<value>\",\"keyword\":\"<subject>\"}:\n"
    "  Equipment name (tractor/pump/sprayer/thresher/machine/weeder/rotavater/seeder/engine/cultivator/jatka/flashlight) without explicit new/used → scenario: equipment\n"
    "  price/bhav/keemat/ભાવ/કિંમત with no clear domain → scenario: price\n"
    "  product/item/vastu/ઉત્પાદ with no domain → scenario: product\n"
    "  City/location name alone → scenario: location\n"
    "\n"
    "RESPOND WITH JSON ONLY — no explanation, no markdown:\n"
    "{\"decision\":\"skip\"}\n"
    "{\"decision\":\"clear\",\"intent\":\"<intent>\",\"keyword\":\"<subject word>\"}\n"
    "{\"decision\":\"ambiguous\",\"scenario\":\"<scenario>\",\"keyword\":\"<subject word>\"}\n"
    "\n"
    "intent: crop_price | kshop_product | buy_sell_product | seed_info | local_news | video_search\n"
    "scenario: equipment | price | crop | product | location\n"
    "keyword: main subject as it appears in the query\n"
    "\n"
    "Examples:\n"
    "\"kapas bhav\" -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"kapas\"}\n"
    "\"I want tractor\" -> {\"decision\":\"ambiguous\",\"scenario\":\"equipment\",\"keyword\":\"tractor\"}\n"
    "\"I want cow\" -> {\"decision\":\"clear\",\"intent\":\"buy_sell_product\",\"keyword\":\"cow\"}\n"
    "\"how i sell any product\" -> {\"decision\":\"skip\"}\n"
    "\"mango price\" -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"mango\"}\n"
    "\"ફ્લેશ લાઈટ કિંમત\" -> {\"decision\":\"ambiguous\",\"scenario\":\"equipment\",\"keyword\":\"ફ્લેશ લાઈટ\"}\n"
    "\"ગાય\" -> {\"decision\":\"clear\",\"intent\":\"buy_sell_product\",\"keyword\":\"ગાય\"}\n"
    "\"rotavater joie\" -> {\"decision\":\"ambiguous\",\"scenario\":\"equipment\",\"keyword\":\"rotavater\"}\n"
    "\"ઘઉં ભાવ\" -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"ઘઉં\"}\n"
    "\"samachar\" -> {\"decision\":\"clear\",\"intent\":\"local_news\",\"keyword\":\"samachar\"}\n"
    "\"surat\" -> {\"decision\":\"ambiguous\",\"scenario\":\"location\",\"keyword\":\"surat\"}\n"
    "\"namaste\" -> {\"decision\":\"skip\"}\n"
    "\"what is krushi ratn\" -> {\"decision\":\"skip\"}"
)


# -----------------------------------------------------------------------------
# Core class
# -----------------------------------------------------------------------------

class ConfirmationLayer:
    """
    LLM-based intent pre-classifier. Stateless. Call: await .check(user_query)

    Returns:
        ConfirmedIntent      — single clear intent, skip F1 UI, inject directly
        ClarificationRequest — ambiguous intent, show options to user
        None                 — navigation/general/greeting, proceed normally

    NOTE: check() is async. Caller must await it:
        result = await get_confirmation_layer().check(user_query)
    """

    def __init__(self):
        self.llm_manager = get_llm_manager()

    async def check(
        self, user_query: str
    ) -> Optional[Union[ClarificationRequest, ConfirmedIntent]]:

        q_orig = user_query.strip()

        # ── LLM call ──────────────────────────────────────────────────────────
        try:
            response = await self.llm_manager.generate(
                messages=[
                    LLMMessage(role="system", content=_F1_SYSTEM),
                    LLMMessage(role="user",   content=q_orig),
                ],
                temperature=0.0,
                max_tokens=80,
            )

            raw = response.content.strip()
            # Strip markdown fences if the model wraps the response
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
            result: dict = json.loads(raw)

        except json.JSONDecodeError as e:
            logger.warning(
                f"F1 JSON parse error: {e!s} | raw={raw!r:.120} — passing through"
            )
            return None
        except Exception as e:
            logger.warning(f"F1 LLM call failed: {e!s} — passing through")
            return None

        # ── Route on decision ─────────────────────────────────────────────────
        decision = result.get("decision", "skip")
        keyword  = result.get("keyword", "").strip()

        # SKIP — navigation, greeting, general app question
        if decision == "skip":
            logger.info(f"F1 SKIP — nav/general/greeting: {q_orig[:60]!r}")
            return None

        # CLEAR — single unambiguous domain
        if decision == "clear":
            intent = result.get("intent", "")
            if intent not in _VALID_INTENTS:
                logger.warning(f"F1 unknown intent '{intent}' — passing through")
                return None
            logger.info(f"F1 CLEAR | intent={intent} keyword={keyword!r}")
            return ConfirmedIntent(
                intent_key=intent,
                confidence=0.92,
                domain=intent,
            )

        # AMBIGUOUS — show clarification buttons
        if decision == "ambiguous":
            scenario = result.get("scenario", "product")
            if scenario not in _VALID_SCENARIOS:
                logger.warning(f"F1 unknown scenario '{scenario}' — defaulting to product")
                scenario = "product"

            logger.info(f"F1 AMBIGUOUS | scenario={scenario} keyword={keyword!r}")

            builder  = _SCENARIO_BUILDERS[scenario]
            options  = builder(keyword, q_orig)
            question = _SCENARIO_QUESTIONS[scenario].format(keyword=keyword or "this")

            if not options:
                logger.warning(f"F1 no options built for scenario={scenario} — passing through")
                return None

            return ClarificationRequest(
                question=question,
                options=options,
                scenario=scenario,
                matched_keyword=keyword,
            )

        logger.warning(f"F1 unexpected decision='{decision}' — passing through")
        return None

    # ── Downstream helpers — UNCHANGED ───────────────────────────────────────

    @staticmethod
    def get_confirmed_tables(intent_key: str) -> List[str]:
        tables = INTENT_TO_TABLES.get(intent_key, [])
        if not tables:
            logger.warning(f"Unknown intent_key '{intent_key}'")
        return tables

    @staticmethod
    def get_intent_note(intent_key: str) -> str:
        return INTENT_TO_PROMPT_NOTE.get(intent_key, "")

    @staticmethod
    def serialize_request(req: ClarificationRequest) -> dict:
        return {
            "type":            "clarification_request",
            "scenario":        req.scenario,
            "matched_keyword": req.matched_keyword,
            "question":        req.question,
            "options": [
                {"label": f"{opt.emoji} {opt.label}", "intent_key": opt.intent_key}
                for opt in req.options
            ],
        }


# -----------------------------------------------------------------------------
# Singleton
# -----------------------------------------------------------------------------

_instance: Optional[ConfirmationLayer] = None


def get_confirmation_layer() -> ConfirmationLayer:
    global _instance
    if _instance is None:
        _instance = ConfirmationLayer()
    return _instance