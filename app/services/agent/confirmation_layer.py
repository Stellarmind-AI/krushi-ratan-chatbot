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

# ─────────────────────────────────────────────────────────────────────────────
# Navigation pseudo-intent — used as intent_key on the "How to use the app"
# button across scenarios. chat_handler intercepts this value and routes the
# resumed pipeline through _flow_navigation instead of _flow_sql.
# This is NOT a member of INTENT_TO_TABLES (no SQL tables to confirm).
# ─────────────────────────────────────────────────────────────────────────────
NAV_INTENT_KEY = "navigation"


# -----------------------------------------------------------------------------
# Option builders — LLM-driven, scenario-only logic.
#
# Design principle (per user mandate):
#   F1 (the LLM) does ALL classification — language detection, intent extraction,
#   ambiguity resolution. The option builders are now DUMB DISPATCHERS that
#   emit a fixed, predictable button set per scenario. NO keyword inspection
#   inside the builders.
#
# Scenario → button matrix:
#   crop             — Nav + Crop price                                   (2)
#   equipment        — Nav + K-Shop new + Buy/Sell used                   (3)
#   equipment_price  — K-Shop price + Buy/Sell price (no nav, golden rule)(2)
#   animal           — Nav + Buy/Sell                                     (2)
#   seed             — Nav + Seed info                                    (2)
#   product          — Nav + K-Shop + Buy/Sell + Crop price               (4)
#   price            — Crop price + K-Shop price + Buy/Sell price          (3)
#   location         — Nav + News + Crop prices (no buy_sell — no FK)     (3)
#
# Golden rule reminder: scenarios containing a price word (price,
# equipment_price) NEVER show a navigation button. If the user mentions
# price, they want DB data, not how-to instructions.
# -----------------------------------------------------------------------------

# Per-scenario navigation button label (User-facing — translated by chat_handler).
# Empty entries (price, equipment_price) mean "no nav button for this scenario".
_NAV_LABEL_BY_SCENARIO: Dict[str, str] = {
    "crop":      "How to view {kw} prices in app",
    "equipment": "How to buy or list {kw} in app",
    "animal":    "How to buy or list {kw} in app",
    "seed":      "How to find seed information in app",
    "product":   "How to navigate the app",
    "location":  "How to use the app",
}

# Per-scenario synthetic question used when user TAPS the navigation button.
# The original user query is often too vague for answer_navigation() to give
# a clean answer — we substitute a complete how-to question so the navigation
# LLM has unambiguous context.
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
    """
    When the user taps the navigation button after F1 paused for clarification,
    chat_handler calls this to build a complete, navigation-flavored question
    that answer_navigation() can match against navigation.json screens.

    Example: scenario='equipment', keyword='tractor' →
             "how do i buy or list tractor in the krushi ratn app"
    """
    template = _NAV_QUERY_BY_SCENARIO.get(scenario, "how do i use the krushi ratn app")
    if "{kw}" in template:
        return template.format(kw=keyword) if keyword else template.replace(" {kw}", "").replace("{kw} ", "")
    return template


def _build_crop_options(kw: str) -> List[ClarificationOption]:
    """2 buttons: Nav + Crop price. Seed-bearing crops still get 2 buttons —
    user can ask 'wheat seed' separately for seed info."""
    k = kw.capitalize() if kw else "Crop"
    return [
        _nav_option("crop", kw or "crop"),
        ClarificationOption(f"Check {k} mandi price", "📈", "crop_price", "crop_price"),
    ]


def _build_equipment_options(kw: str) -> List[ClarificationOption]:
    """3 buttons: Nav + K-Shop new + Buy/Sell used.
    User who said 'use motor' / 'new motor' never reaches here — F1 marks those CLEAR."""
    k = kw.capitalize() if kw else "Equipment"
    return [
        _nav_option("equipment", kw or "equipment"),
        ClarificationOption(f"New {k} from K-Shop",              "🏪", "equipment_kshop", "kshop"),
        ClarificationOption(f"Used {k} on Buy/Sell marketplace", "🔄", "equipment_used",  "buy_sell"),
    ]


def _build_equipment_price_options(kw: str) -> List[ClarificationOption]:
    """2 buttons: K-Shop price + Buy/Sell price.
    NO navigation — golden rule: price word = SQL, never nav."""
    k = kw.capitalize() if kw else "Equipment"
    return [
        ClarificationOption(f"New {k} price (K-Shop)",         "🏪", "equipment_kshop", "kshop"),
        ClarificationOption(f"Used {k} price (Buy/Sell)",      "🔄", "equipment_used",  "buy_sell"),
    ]


def _build_animal_options(kw: str) -> List[ClarificationOption]:
    """2 buttons: Nav + Buy/Sell. Animals only exist in buy_sell_products
    so there is no SQL ambiguity — the choice is 'how-to' vs 'show listings'."""
    k = kw.capitalize() if kw else "this animal"
    return [
        _nav_option("animal", kw or "animal"),
        ClarificationOption(f"View {k} listings on Buy/Sell", "🔄", "buy_sell_product", "buy_sell"),
    ]


def _build_seed_options(kw: str) -> List[ClarificationOption]:
    """2 buttons: Nav + Seed info. Triggered for bare 'seed'/'bij' queries
    with no specific crop attached."""
    return [
        _nav_option("seed", kw),
        ClarificationOption("Seed varieties available", "🌱", "seed_info", "seeds"),
    ]


def _build_product_options(kw: str) -> List[ClarificationOption]:
    """4 buttons: Nav + K-Shop + Buy/Sell + Crop price.
    For generic 'items?' / 'products' / 'vastu' queries — covers the realistic
    intents without showing every domain."""
    return [
        _nav_option("product", kw),
        ClarificationOption("K-Shop (new farm equipment & supplies)", "🏪", "kshop_product",    "kshop"),
        ClarificationOption("Buy/Sell marketplace (farmer listings)",  "🔄", "buy_sell_product", "buy_sell"),
        ClarificationOption("Mandi / yard crop prices",                "📊", "crop_price",       "crop_price"),
    ]


def _build_price_options(kw: str) -> List[ClarificationOption]:
    """3 buttons: Crop price + K-Shop price + Buy/Sell price.
    NO navigation — golden rule: price word = SQL, never nav."""
    label_crop = f"{kw.capitalize()} price at mandi/yard" if kw else "Crop price at mandi/yard"
    return [
        ClarificationOption(label_crop,                 "📊", "crop_price",       "crop_price"),
        ClarificationOption("K-Shop product price",     "🛒", "kshop_product",    "kshop"),
        ClarificationOption("Buy/Sell listing price",   "💰", "buy_sell_product", "buy_sell"),
    ]


def _build_location_options(kw: str) -> List[ClarificationOption]:
    """3 buttons: Nav + News + Crop prices.
    buy_sell_products is excluded — the table has no location columns
    (no city_id/state_id/taluka_id), so location filtering doesn't apply."""
    k = kw.capitalize() if kw else "this location"
    return [
        _nav_option("location", kw),
        ClarificationOption(f"Agricultural news from {k}", "📰", "local_news", "news"),
        ClarificationOption(f"Crop prices near {k}",       "📈", "crop_price", "crop_price"),
    ]


# -----------------------------------------------------------------------------
# Scenario → question text + builder
# -----------------------------------------------------------------------------

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
# F1 system prompt — tight, domain-grounded, ~500 tokens
# -----------------------------------------------------------------------------

_F1_SYSTEM = (
    "You are F1, intent pre-classifier for Krushi Ratn (Gujarati farming app).\n"
    "Inputs may be in English, Romanized Gujarati, Hindi, or Gujarati script.\n"
    "\n"
    "GOLDEN RULES — apply in order, earlier rule overrides later.\n"
    "\n"
    "RULE 1 — PROCESS INTERROGATIVES → skip.\n"
    "  Any query asking HOW/WHERE to do/find/use/check/view something is navigation.\n"
    "  Triggers: how to, how do/can/i, where to/can i, steps to, way to, kevi rite,\n"
    "  kem karvu, kayi rite, kaise, કેવી રીતે, કેમ કરવું.\n"
    "  Applies EVEN IF query also contains price/crop/equipment/animal words.\n"
    "  Examples: \"how to view kapas bhav\", \"where can i find onion price\",\n"
    "  \"ghau bhav kevi rite jovo\", \"kevi rite pak vechuv\" → skip.\n"
    "\n"
    "RULE 2 — SELL/LIST PROCESS → skip.\n"
    "  Any sell/list intent (even without \"how\") needs the app's sell flow.\n"
    "  Triggers: sell, list for sale, post for sale, vechuv, vechan, listing muku,\n"
    "  વેચવું, વેચાણ, મૂકવું.\n"
    "  Examples: \"i want to sell wheat\", \"mare cow vechvi che\", \"pak vechuv\" → skip.\n"
    "\n"
    "RULE 3 — BUY/PURCHASE PROCESS → skip.\n"
    "  Any buy/purchase/order intent is navigation, EVEN IF the query also names\n"
    "  a section (kshop, buy/sell), crop, animal, or equipment.\n"
    "  Triggers: i want to/looking to/planning to/can i/where to/how to/what can i\n"
    "  do to + buy|purchase|order; kharidu, kharidvu, kharido, kharidi, mane joie,\n"
    "  mare ... kharidvu chhe, kharidna, ખરીદવું, ખરીદવી, ખરીદી, ખરીદો, ખરીદવા.\n"
    "  Examples: \"i want to buy wheat\", \"i want to buy kshop products\",\n"
    "  \"i want to buy cow from buy/sell\", \"what can i do to buy ginger\",\n"
    "  \"mane tractor kharidvu chhe\", \"ગાય ખરીદવી છે\" → skip.\n"
    "  When in doubt with \"buy\", skip.\n"
    "\n"
    "RULE 4 — PRICE WORD + INFO (no process trigger from rules 1-3) → CLEAR data.\n"
    "  Price words: bhav, keemat, kimat, price, rate, ભાવ, કિંમત.\n"
    "  Examples:\n"
    "    \"kapas bhav\", \"what is wheat price\", \"tell me kapas keemat\" → clear:crop_price\n"
    "    \"buffalo bhav\", \"ગાય ભાવ\"                                       → clear:buy_sell_product\n"
    "    \"tractor bhav\", \"pump price\"                                    → ambiguous:equipment_price\n"
    "\n"
    "RULE 5 — PRICE WORD + LOCATION (no equipment/animal/seed/news/video word)\n"
    "        → CLEAR crop_price. Krushi Ratn shows location-filtered prices ONLY\n"
    "        for crops, so this combination is unambiguous. NEVER ask clarification.\n"
    "  Treat ANY plausible Indian/Gujarati city or taluka name as a location\n"
    "  (e.g. rajkot, surat, mahuva, bhavnagar, gondal, ગોંડલ, મહુવા, રાજકોટ).\n"
    "  When BOTH a crop name AND a location appear, use the CROP as the keyword\n"
    "  (more useful filter for SQL than the location).\n"
    "  Examples:\n"
    "    \"price in rajkot\", \"bhav in mahuva\", \"રાજકોટ માં ભાવ\" → clear:crop_price (kw=location)\n"
    "    \"onion price in mahuva\", \"કપાસ ભાવ રાજકોટ\"            → clear:crop_price (kw=crop)\n"
    "\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "DECISIONS\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "\n"
    "SKIP — {\"decision\":\"skip\",\"flow\":\"<NAVIGATION|GREETING|GENERAL>\"}\n"
    "  When skipping, ALSO include the downstream flow so the route agent can be\n"
    "  bypassed. Map cleanly:\n"
    "  • Rules 1, 2, 3 (process / sell / buy)             → flow=NAVIGATION\n"
    "  • Pure greetings (hello, hi, namaste, kem cho)     → flow=GREETING\n"
    "  • Generic app concepts (\"what is krushi ratn\",\n"
    "    \"is app free\", \"app features\", \"what languages\",\n"
    "    \"what is yard\" concept)                          → flow=GENERAL\n"
    "  If you are unsure which flow applies, omit the flow field — emit just\n"
    "  {\"decision\":\"skip\"} and the route agent will classify.\n"
    "\n"
    "CLEAR — {\"decision\":\"clear\",\"intent\":\"<X>\",\"keyword\":\"<subject>\"}\n"
    "  Intent values: crop_price | seed_info | equipment_kshop | equipment_used |\n"
    "                 kshop_product | buy_sell_product | local_news | video_search\n"
    "\n"
    "  • Crop name + price word                          → crop_price\n"
    "  • Crop name + seed word (seed/bij/બીજ/બિયારણ)     → seed_info\n"
    "  • Equipment + NEW signal\n"
    "      (new, brand new, naya, navu, navi, નવું, નવી, नया)         → equipment_kshop\n"
    "  • Equipment + USED signal\n"
    "      (use, used, old, second hand, juno/juni/junu/junum/junun/juna,\n"
    "       purano/purani/jaani, જૂનું/જૂનો/જૂની, जुना/पुराना)         → equipment_used\n"
    "  • Animal name + price word (animals are buy_sell-only) → buy_sell_product\n"
    "  • Explicit kshop / k-shop / k-store / કે-શોપ          → kshop_product\n"
    "  • Explicit buy sell / buysell / marketplace / વેચાણ   → buy_sell_product\n"
    "  • news / samachar / khabar / ખબર / ન્યૂઝ              → local_news\n"
    "  • video / વિડિઓ / watch                                → video_search\n"
    "\n"
    "AMBIGUOUS — {\"decision\":\"ambiguous\",\"scenario\":\"<X>\",\"keyword\":\"<subject>\"}\n"
    "  Scenario values: crop | equipment | equipment_price | animal | seed |\n"
    "                   product | price | location\n"
    "\n"
    "  • Bare crop name (no price/seed)                    → crop\n"
    "  • Bare equipment name (no new/used/price)           → equipment\n"
    "  • Equipment + price word (no new/used signal)       → equipment_price\n"
    "  • Bare animal name (no price)                       → animal\n"
    "  • Bare seed word (no crop)                          → seed\n"
    "  • Bare price word (no subject)                      → price\n"
    "  • Generic items/products word (no domain)           → product\n"
    "  • Bare location name (no subject)                   → location\n"
    "\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "RESPONSE — JSON ONLY, NO MARKDOWN, NO EXPLANATION\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "  {\"decision\":\"skip\",\"flow\":\"NAVIGATION|GREETING|GENERAL\"}  (preferred)\n"
    "  {\"decision\":\"skip\"}                                          (only if unsure)\n"
    "  {\"decision\":\"clear\",\"intent\":\"<intent>\",\"keyword\":\"<subject>\"}\n"
    "  {\"decision\":\"ambiguous\",\"scenario\":\"<scenario>\",\"keyword\":\"<subject>\"}\n"
    "\n"
    "  keyword: subject as written by user (tractor, ઘઉં, kapas).\n"
    "  Empty string \"\" only when no specific subject (e.g. \"video\").\n"
    "\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "EXTRA EXAMPLES (cases not yet covered above)\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "  \"wheat seed\" → clear:seed_info,wheat        \"ghau bij\" → clear:seed_info,ghau\n"
    "  \"new tractor\" → clear:equipment_kshop,tractor   \"naya pump\" → clear:equipment_kshop,pump\n"
    "  \"used motor\" → clear:equipment_used,motor   \"juno tractor\" → clear:equipment_used,tractor\n"
    "  \"second hand sprayer\" → clear:equipment_used,sprayer\n"
    "  \"samachar\" → clear:local_news,samachar      \"video\" → clear:video_search,\n"
    "  \"kshop products\" → clear:kshop_product,\n"
    "  \"wheat?\" → ambiguous:crop,wheat            \"tractor\" → ambiguous:equipment,tractor\n"
    "  \"tractor bhav\" → ambiguous:equipment_price,tractor\n"
    "  \"cow\" → ambiguous:animal,cow                \"i want bakri\" → ambiguous:animal,bakri\n"
    "  \"seed\" → ambiguous:seed,seed                \"products\" → ambiguous:product,products\n"
    "  \"bhav?\" → ambiguous:price,bhav              \"surat\" → ambiguous:location,surat"
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

        # SKIP — navigation, greeting, general app question.
        # If F1 also returned a confident `flow`, surface it via ConfirmedFlow so
        # chat_handler can bypass route_agent (Phase 5 optimization). When the
        # flow field is missing/invalid, return None as before — route_agent
        # falls back to its normal classification (zero-regression path).
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