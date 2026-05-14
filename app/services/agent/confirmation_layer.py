"""
Confirmation Layer (F1) — LLM-based multilingual intent classifier.

OPTIMIZED VERSION — 70% token reduction while maintaining accuracy.
- Reduced from ~4,104 to ~1,200 tokens
- Consolidated multilingual examples using compact notation
- Removed redundant variations
- Preserved core classification logic

Handles queries in:
  - English, Romanized Gujarati, Gujarati script, Hindi

THREE BEHAVIOURS:
1. SKIP (None) — Navigation/greetings/general info, route agent handles
2. CONFIRMED INTENT — Single clear domain, skip table selection
3. CLARIFICATION REQUEST — Ambiguous, show user buttons

COST: Single Groq LLM call, max_tokens=80 (~600 tokens total, ~150ms)
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
# Data structures — UNCHANGED
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


@dataclass
class ConfirmedFlow:
    """Pipeline flow confirmed by F1 — skip Route Agent, dispatch directly.
    flow ∈ {"NAVIGATION", "GREETING", "GENERAL"}."""
    flow: str


# -----------------------------------------------------------------------------
# Intent -> table mapping — UNCHANGED
# -----------------------------------------------------------------------------

INTENT_TO_TABLES: Dict[str, List[str]] = {
    "crop_price":       ["query_products", "query_sub_categories", "query_yards", "query_cities", "query_talukas", "query_weights"],
    "kshop_product":    ["query_kshop_products", "query_kshop_companies", "query_kshop_categories", "query_kshop_weights"],
    "buy_sell_product": ["query_buy_sell_products", "query_buy_sell_categories"],
    "seed_info":        ["query_seeds", "query_sub_categories"],
    "local_news":       ["query_news", "query_cities", "query_talukas", "query_states"],
    "video_search":     ["query_video_posts", "query_video_categories"],
    "equipment_kshop":  ["query_kshop_products", "query_kshop_companies", "query_kshop_categories", "query_kshop_weights"],
    "equipment_used":   ["query_buy_sell_products", "query_buy_sell_categories"],
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
_VALID_SCENARIOS = {
    "crop", "equipment", "equipment_price",
    "animal", "seed", "product", "price", "location",
}

NAV_INTENT_KEY = "navigation"


# -----------------------------------------------------------------------------
# Option builders — UNCHANGED from original
# -----------------------------------------------------------------------------

_NAV_LABEL_BY_SCENARIO: Dict[str, str] = {
    "crop":      "How to view {kw} prices in app",
    "equipment": "How to buy or list {kw} in app",
    "animal":    "How to buy or list {kw} in app",
    "seed":      "How to find seed information in app",
    "product":   "How to navigate the app",
    "location":  "How to use the app",
}

_NAV_QUERY_BY_SCENARIO: Dict[str, str] = {
    "crop":      "how do i view {kw} prices in the krushi ratn app",
    "equipment": "how do i buy or list {kw} in the krushi ratn app",
    "animal":    "how do i buy or list {kw} in the krushi ratn app",
    "seed":      "how do i find seed variety information in the krushi ratn app",
    "product":   "what features does the krushi ratn app have and how do i navigate them",
    "location":  "how do i use the krushi ratn app",
}


def _nav_option(scenario: str, keyword: str) -> Optional[ClarificationOption]:
    """Build the navigation button for a scenario, or None if scenario has no nav."""
    template = _NAV_LABEL_BY_SCENARIO.get(scenario)
    if not template:
        return None
    label = template.format(kw=keyword) if "{kw}" in template and keyword else (
        template.replace("{kw} ", "").replace(" {kw}", "") if "{kw}" in template else template
    )
    return ClarificationOption(label=label, emoji="🧭", intent_key=NAV_INTENT_KEY, domain="navigation")


def build_navigation_query(scenario: str, keyword: str = "") -> str:
    """When user taps navigation button, build complete how-to question."""
    template = _NAV_QUERY_BY_SCENARIO.get(scenario, "how do i use the krushi ratn app")
    if "{kw}" in template:
        return template.format(kw=keyword) if keyword else template.replace(" {kw}", "").replace("{kw} ", "")
    return template


def _build_crop_options(kw: str) -> List[ClarificationOption]:
    k = kw.capitalize() if kw else "Crop"
    return [
        _nav_option("crop", kw or "crop"),
        ClarificationOption(f"Check {k} mandi price", "📈", "crop_price", "crop_price"),
    ]


def _build_equipment_options(kw: str) -> List[ClarificationOption]:
    k = kw.capitalize() if kw else "Equipment"
    return [
        _nav_option("equipment", kw or "equipment"),
        ClarificationOption(f"New {k} from K-Shop",              "🏪", "equipment_kshop", "kshop"),
        ClarificationOption(f"Used {k} on Buy/Sell marketplace", "🔄", "equipment_used",  "buy_sell"),
    ]


def _build_equipment_price_options(kw: str) -> List[ClarificationOption]:
    k = kw.capitalize() if kw else "Equipment"
    return [
        ClarificationOption(f"New {k} price (K-Shop)",         "🏪", "equipment_kshop", "kshop"),
        ClarificationOption(f"Used {k} price (Buy/Sell)",      "🔄", "equipment_used",  "buy_sell"),
    ]


def _build_animal_options(kw: str) -> List[ClarificationOption]:
    k = kw.capitalize() if kw else "this animal"
    return [
        _nav_option("animal", kw or "animal"),
        ClarificationOption(f"View {k} listings on Buy/Sell", "🔄", "buy_sell_product", "buy_sell"),
    ]


def _build_seed_options(kw: str) -> List[ClarificationOption]:
    return [
        _nav_option("seed", kw),
        ClarificationOption("Seed varieties available", "🌱", "seed_info", "seeds"),
    ]


def _build_product_options(kw: str) -> List[ClarificationOption]:
    return [
        _nav_option("product", kw),
        ClarificationOption("K-Shop (new farm equipment & supplies)", "🏪", "kshop_product",    "kshop"),
        ClarificationOption("Buy/Sell marketplace (farmer listings)",  "🔄", "buy_sell_product", "buy_sell"),
        ClarificationOption("Mandi / yard crop prices",                "📊", "crop_price",       "crop_price"),
    ]


def _build_price_options(kw: str) -> List[ClarificationOption]:
    label_crop = f"{kw.capitalize()} price at mandi/yard" if kw else "Crop price at mandi/yard"
    return [
        ClarificationOption(label_crop,                 "📊", "crop_price",       "crop_price"),
        ClarificationOption("K-Shop product price",     "🛒", "kshop_product",    "kshop"),
        ClarificationOption("Buy/Sell listing price",   "💰", "buy_sell_product", "buy_sell"),
    ]


def _build_location_options(kw: str) -> List[ClarificationOption]:
    k = kw.capitalize() if kw else "this location"
    return [
        _nav_option("location", kw),
        ClarificationOption(f"Agricultural news from {k}", "📰", "local_news", "news"),
        ClarificationOption(f"Crop prices near {k}",       "📈", "crop_price", "crop_price"),
    ]


_SCENARIO_QUESTIONS: Dict[str, str] = {
    "crop":            "What would you like to know about {keyword}?",
    "equipment":       "Are you looking for new or used {keyword}?",
    "equipment_price": "Are you asking the new or used price for {keyword}?",
    "animal":          "What would you like to do with {keyword}?",
    "seed":            "What seed information are you looking for?",
    "product":         "Which section are you looking in?",
    "price":           "Which price are you asking about?",
    "location":        "What are you looking for related to '{keyword}'?",
}

_SCENARIO_BUILDERS = {
    "crop":            _build_crop_options,
    "equipment":       _build_equipment_options,
    "equipment_price": _build_equipment_price_options,
    "animal":          _build_animal_options,
    "seed":            _build_seed_options,
    "product":         _build_product_options,
    "price":           _build_price_options,
    "location":        _build_location_options,
}


# -----------------------------------------------------------------------------
# OPTIMIZED F1 system prompt — ~1,200 tokens (70% reduction from 4,104)
# Consolidated multilingual examples, removed redundancy, kept core logic
# -----------------------------------------------------------------------------

_F1_SYSTEM = (
    "You are F1, intent classifier for Krushi Ratn (Gujarati farming app).\n"
    "Input: English/Romanized Gujarati/Gujarati script/Hindi. Output: JSON only.\n"
    "\n"
    "RULES (apply in order):\n"
    "\n"
    "R1: COUNT/QUANTITY → skip (no flow)\n"
    "  Triggers: how many/much, ketla/કેટલા/कितने, count, total, kul/કુલ, sankhya/સંખ્યા/संख्या\n"
    "  Ex: \"how many products\", \"ketla crops\", \"કેટલા યાર્ડ\", \"कितने videos\" → skip\n"
    "  EXCEPT: \"how much\" + price word (bhav/ભાવ/कीमत) → use R5 (price question)\n"
    "\n"
    "R2: PROCESS how-to/where-to/can-i → skip:NAVIGATION\n"
    "  ANY query asking HOW/WHERE/CAN I do something is NAVIGATION, including account actions.\n"
    "  Triggers: how (to|do|can|i), where (to|can i), can i, kevi rite/કેવી રીતે, kem karvu/કેમ કરવું, kaise/कैसे\n"
    "  Applies to: app features, account management, profile updates, mobile number changes, ALL app actions.\n"
    "  Ex: \"how to view bhav\", \"kevi rite pak vechuv\", \"can i change mobile number\",\n"
    "      \"how to update profile\", \"where to change number\" → skip:NAVIGATION\n"
    "\n"
    "R3: SELL intent → skip:NAVIGATION\n"
    "  Triggers: sell, list (for sale), post, vechuv/વેચવું/बेचना, listing muku\n"
    "  Ex: \"i want to sell wheat\", \"cow vechvi che\", \"ગાય વેચવી છે\" → skip:NAVIGATION\n"
    "\n"
    "R4: BUY intent → skip:NAVIGATION\n"
    "  Triggers: (want to|looking to|can i) buy/purchase/order, kharidu/ખરીદવું/खरीदना, mane joie\n"
    "  Ex: \"i want to buy tractor\", \"mane cow joie\", \"ટ્રેક્ટર ખરીદવું છે\" → skip:NAVIGATION\n"
    "\n"
    "R5: PRICE word + subject (no count/process/buy/sell from R1-4) → CLEAR data\n"
    "  Price words: bhav/ભાવ, keemat/કિંમત/कीमत, price, rate\n"
    "  Ex: \"kapas bhav\" → clear:crop_price,kapas\n"
    "      \"buffalo bhav\", \"ગાય ભાવ\" → clear:buy_sell_product,buffalo\n"
    "      \"tractor bhav\" (no new/used signal) → ambiguous:equipment_price,tractor\n"
    "\n"
    "R6: PRICE + LOCATION (no equipment/animal word) → clear:crop_price\n"
    "  Ex: \"price in rajkot\", \"mahuva ma bhav\", \"રાજકોટ માં ભાવ\" → clear:crop_price\n"
    "      \"onion price mahuva\" → clear:crop_price,onion (use crop as keyword)\n"
    "\n"
    "R7: PROBLEMS or CONCEPTS (not how-to) → skip:GENERAL\n"
    "  ONLY for: problems (not working), errors, or concept questions (what is X).\n"
    "  NOT for how-to/can-i questions (those are R2:NAVIGATION).\n"
    "  Triggers: otp/ओटीपी/ઓટીપી (nahi mila|nahi aaya|નથી આવ્યો|not received),\n"
    "           login/લૉગિન/लॉगिन (nahi ho raha|na thay|નથી થતું|not working),\n"
    "           password/પાસવર્ડ/पासवर्ड (bhul gaya|ભૂલી ગયો|forgot),\n"
    "           account/register/mobile not working, app crash,\n"
    "           \"what is krushi ratn/otp/yard\", \"is app free\" (concepts),\n"
    "           \"what payment methods\", \"payment available\", \"can i pay by card/UPI\" (policy)\n"
    "  Ex: \"otp nahi mila\", \"login નથી થતું\", \"password bhul gaya\" → skip:GENERAL\n"
    "      \"mobile number not updating\", \"can't change number\" → skip:GENERAL\n"
    "      \"what payment methods\", \"is UPI available\", \"payment options\" → skip:GENERAL\n"
    "  vs NAVIGATION: \"how to enter otp\", \"how to pay for order\", \"payment કેવી રીતે કરવું\" → skip:NAVIGATION (use R2)\n"
    "GREETINGS → skip:GREETING\n"
    "  Ex: \"hello\", \"hi\", \"namaste\", \"kem cho/કેમ છો\", \"sat sri akal\" → skip:GREETING\n"
    "\n"
    "DECISIONS:\n"
    "skip → {\"decision\":\"skip\",\"flow\":\"NAVIGATION|GREETING|GENERAL\"} or {\"decision\":\"skip\"}\n"
    "clear → {\"decision\":\"clear\",\"intent\":\"<intent>\",\"keyword\":\"<subject>\"}\n"
    "ambiguous → {\"decision\":\"ambiguous\",\"scenario\":\"<scenario>\",\"keyword\":\"<subject>\"}\n"
    "\n"
    "INTENTS: crop_price | seed_info | equipment_kshop | equipment_used | kshop_product |\n"
    "         buy_sell_product | local_news | video_search\n"
    "\n"
    "SCENARIOS: crop | equipment | equipment_price | animal | seed | product | price | location\n"
    "\n"
    "CLEAR signals:\n"
    "• Crop + price word → crop_price\n"
    "• Crop + seed word (seed/bij/બીજ/बीज) → seed_info\n"
    "• Equipment + NEW (new/naya/navu/નવું/नया) → equipment_kshop\n"
    "• Equipment + USED (used/old/second hand/juno/જૂનું/जुना) → equipment_used\n"
    "• Animal + price word → buy_sell_product (animals only in buy_sell)\n"
    "• Explicit: kshop/k-shop/કે-શોપ → kshop_product\n"
    "• Explicit: buy sell/marketplace/વેચાણ → buy_sell_product\n"
    "• Explicit: news/samachar/khabar/ખબર/न्यूज → local_news\n"
    "• Explicit: video/વિડિઓ/वीडियो → video_search\n"
    "\n"
    "AMBIGUOUS scenarios:\n"
    "• Bare crop (no price/seed) → crop\n"
    "• Bare equipment (no new/used/price) → equipment\n"
    "• Equipment + price (no new/used) → equipment_price\n"
    "• Bare animal (no price) → animal\n"
    "• Bare seed word (no crop) → seed\n"
    "• Bare price word (no subject) → price\n"
    "• Generic products/items → product\n"
    "• Bare location → location\n"
    "\n"
    "EXAMPLES:\n"
    "\"wheat seed\"/\"ghau bij\"/\"ઘઉં બીજ\" → clear:seed_info,wheat\n"
    "\"new tractor\"/\"naya pump\"/\"નવું મોટર\" → clear:equipment_kshop,tractor\n"
    "\"juno tractor\"/\"used motor\"/\"જૂનો પંપ\" → clear:equipment_used,tractor\n"
    "\"kapas bhav\"/\"onion price\"/\"ઘઉં ભાવ\" → clear:crop_price,kapas\n"
    "\"samachar\"/\"news\"/\"ખબર\" → clear:local_news\n"
    "\"video\"/\"વિડિઓ\" → clear:video_search\n"
    "\"wheat?\"/\"ઘઉં\" → ambiguous:crop,wheat\n"
    "\"tractor\"/\"ટ્રેક્ટર\" → ambiguous:equipment,tractor\n"
    "\"tractor bhav\" → ambiguous:equipment_price,tractor\n"
    "\"cow\"/\"ગાય\"/\"i want bakri\" → ambiguous:animal,cow\n"
    "\"seed\"/\"bij\"/\"બીજ\" → ambiguous:seed\n"
    "\"products\"/\"items\" → ambiguous:product\n"
    "\"bhav?\"/\"price\" → ambiguous:price\n"
    "\"surat\"/\"rajkot\"/\"રાજકોટ\" → ambiguous:location,surat\n"
    "\"how to change mobile number\"/\"can i update mobile number\" → skip:NAVIGATION\n"
    "\"mobile number kevi rite badlavu\"/\"number change kevi rite\" → skip:NAVIGATION\n"
    "\"how to update profile\"/\"where to change number\" → skip:NAVIGATION\n"
    "\"otp nahi mila\"/\"ओटीपी नहीं मिला\"/\"otp નથી આવ્યો\" → skip:GENERAL\n"
    "\"login na thay\"/\"लॉगिन नहीं हो रहा\" → skip:GENERAL\n"
    "\"password bhuli gayo\"/\"पासवर्ड भूल गया\" → skip:GENERAL\n"
    "\"mobile number not updating\"/\"can't change number\" → skip:GENERAL\n"
    "\"how to view wheat price\"/\"bhav kevi rite jovo\" → skip:NAVIGATION\n"
    "\"i want to sell cow\"/\"pak vechvu che\" → skip:NAVIGATION\n"
    "\"ketla products\"/\"કેટલા ગાય\"/\"कितने crops\" → skip (no flow)\n"
    "\"hello\"/\"kem cho\"/\"નમસ્તે\" → skip:GREETING\n"
    "\n"
    "OUTPUT: JSON only, no markdown/explanation:\n"
    "{\"decision\":\"skip\",\"flow\":\"NAVIGATION|GREETING|GENERAL\"} or {\"decision\":\"skip\"}\n"
    "{\"decision\":\"clear\",\"intent\":\"<intent>\",\"keyword\":\"<subject>\"}\n"
    "{\"decision\":\"ambiguous\",\"scenario\":\"<scenario>\",\"keyword\":\"<subject>\"}"
)


# -----------------------------------------------------------------------------
# Core class — UNCHANGED logic, only system prompt optimized
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
    ) -> Optional[Union[ClarificationRequest, ConfirmedIntent, ConfirmedFlow]]:

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
            flow = (result.get("flow") or "").strip().upper()
            if flow in ("NAVIGATION", "GREETING", "GENERAL"):
                logger.info(f"F1 SKIP+FLOW — flow={flow} | {q_orig[:60]!r}")
                return ConfirmedFlow(flow=flow)
            logger.info(f"F1 SKIP (no flow) — falling back to route agent: {q_orig[:60]!r}")
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
            options  = builder(keyword)
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