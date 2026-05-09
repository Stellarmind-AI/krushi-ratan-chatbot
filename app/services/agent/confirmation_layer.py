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
    "You are F1, the intent pre-classifier for Krushi Ratn — a Gujarati farming marketplace app.\n"
    "\n"
    "You understand queries in:\n"
    "  • English (price, tractor, cow, news, what, how, where)\n"
    "  • Romanized Gujarati (bhav, kapas, kevi rite, mane joie, kem)\n"
    "  • Hindi (mujhe, chahiye, naya, kya, kaise)\n"
    "  • Gujarati script (ભાવ, કપાસ, ઘઉં, નવું, ગાય, કેવી, શું)\n"
    "\n"
    "DATABASE DOMAINS:\n"
    "  • crop_price       — Mandi/yard market prices for crops\n"
    "                        (wheat/ghau/ઘઉં, cotton/kapas/કપાસ, onion/ડુંગળી, tomato/ટામેટા,\n"
    "                         potato/બટાકા, mango/કેરી, all vegetables/fruits/grains/pulses)\n"
    "  • seed_info        — Crop seed varieties (bij/બીજ/બિયારણ)\n"
    "  • equipment_kshop  — NEW farming equipment from K-Shop store\n"
    "                        (tractor, pump, sprayer, weeder, thresher, motor, seeder,\n"
    "                         cultivator, engine, jatka machine, flashlight, tools)\n"
    "  • equipment_used   — USED/old farming equipment listed by farmers in Buy/Sell\n"
    "  • buy_sell_product — Marketplace listings: ALL ANIMALS\n"
    "                        (cow/ગાય, buffalo/ભેંસ, goat/બકરી, horse/ઘોડો, camel/ઊંટ,\n"
    "                         sheep/ઘેટું, ox/બળદ, bull) + used farmer-listed items\n"
    "  • kshop_product    — Browsing K-Shop section explicitly\n"
    "  • local_news       — Agricultural news / samachar / khabar / ખબર\n"
    "  • video_search     — Educational farming videos / વિડિઓ\n"
    "\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "GOLDEN RULES — APPLY IN ORDER, EARLIER RULE OVERRIDES LATER\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "\n"
    "RULE 1 — PROCESS INTERROGATIVES ALWAYS GO TO NAVIGATION (skip):\n"
    "  When the user asks HOW to do/find/use/check something, or describes a\n"
    "  step-by-step process, the intent is navigation. Return {\"decision\":\"skip\"}\n"
    "  EVEN IF the query also contains a price word, crop name, equipment, or\n"
    "  any other DB-related word.\n"
    "\n"
    "  Process triggers (any of these in the query):\n"
    "    English:    how to, how do i, how can i, how i, where to, where can i,\n"
    "                where do i, steps to, process for, way to, guide for\n"
    "    Romanized:  kevi rite, kem karvu, kem kari, kayi rite, kaise, kevi reete\n"
    "    Gujarati:   કેવી રીતે, કેમ કરવું, કયા રીતે, કેવી રીત\n"
    "\n"
    "  Examples that MUST go to skip:\n"
    "    \"how i sell wheat\"             -> {\"decision\":\"skip\"} (sell process)\n"
    "    \"how to view kapas bhav\"       -> {\"decision\":\"skip\"} (process despite bhav)\n"
    "    \"where can i find onion price\" -> {\"decision\":\"skip\"} (process despite price)\n"
    "    \"kevi rite pak vechuv\"         -> {\"decision\":\"skip\"} (sell process)\n"
    "    \"ghau bhav kevi rite jovo\"     -> {\"decision\":\"skip\"} (process despite bhav)\n"
    "    \"how do i check tractor price\" -> {\"decision\":\"skip\"} (process despite price)\n"
    "\n"
    "RULE 2 — SELL/LIST PROCESS ALWAYS GOES TO NAVIGATION (skip):\n"
    "  Selling/listing crops, animals, equipment, or any product on Krushi Ratn\n"
    "  ALWAYS requires the app's sell flow. Any sell/list intent — even without\n"
    "  an explicit \"how\" — is navigation.\n"
    "\n"
    "  Sell triggers:\n"
    "    English:    i want to sell, sell my, list for sale, list my, post for sale\n"
    "    Romanized:  vechuv, vechan, vechan karvu, listing muku, post karvu\n"
    "    Gujarati:   વેચવું, વેચાણ, મૂકવું, પોસ્ટ કરવું\n"
    "\n"
    "  Examples:\n"
    "    \"i want to sell wheat\"   -> {\"decision\":\"skip\"}\n"
    "    \"pak vechuv\"             -> {\"decision\":\"skip\"}\n"
    "    \"mare cow vechvi che\"    -> {\"decision\":\"skip\"}\n"
    "    \"list my motor for sale\" -> {\"decision\":\"skip\"}\n"
    "\n"
    "RULE 3 — BUY/PURCHASE PROCESS ALWAYS GOES TO NAVIGATION (skip):\n"
    "  Buying/purchasing/ordering crops, animals, equipment, or any product on\n"
    "  Krushi Ratn requires the app's buy/order flow. ANY buy/purchase intent —\n"
    "  even without an explicit \"how\" — is navigation. This applies EVEN IF\n"
    "  the query also names a section (kshop, k-shop, buy/sell, marketplace),\n"
    "  a crop, an animal, or an equipment word.\n"
    "\n"
    "  Buy triggers (any of these in the query):\n"
    "    English:    i want to buy, want to buy, i want to purchase, want to purchase,\n"
    "                i want to order, want to order, looking to buy, planning to buy,\n"
    "                can i buy, where to buy, where can i buy, how to buy, how do i buy,\n"
    "                what can i do to buy, i need to buy, i wish to buy\n"
    "    Romanized:  kharidu, kharidvu, kharidvi, kharidva, kharido, kharidi, kharidva chhe,\n"
    "                kharidvu chhe, mare kharidvu, mare kharidva, mane joie, mare joie,\n"
    "                mane <X> kharidvu chhe, order karvu, order karva\n"
    "    Hindi:      kharidna, kharidna chahta hu, kharidne ke liye, lena chahta hu\n"
    "    Gujarati:   ખરીદવું, ખરીદવી, ખરીદી, ખરીદો, ખરીદવાનું, ખરીદવા\n"
    "\n"
    "  Examples that MUST go to skip:\n"
    "    \"i want to buy wheat\"             -> {\"decision\":\"skip\"} (buy crop)\n"
    "    \"i want to buy kshop products\"    -> {\"decision\":\"skip\"} (buy from kshop)\n"
    "    \"i want to buy cow from buy/sell\" -> {\"decision\":\"skip\"} (buy from buy/sell)\n"
    "    \"what can i do to buy ginger\"     -> {\"decision\":\"skip\"} (buy crop)\n"
    "    \"i want to purchase tractor\"      -> {\"decision\":\"skip\"} (buy equipment)\n"
    "    \"mane tractor kharidvu chhe\"      -> {\"decision\":\"skip\"} (buy equipment)\n"
    "    \"mare kapas kharidvo chhe\"        -> {\"decision\":\"skip\"} (buy crop)\n"
    "    \"ગાય ખરીદવી છે\"                    -> {\"decision\":\"skip\"} (buy animal)\n"
    "    \"ઘઉં ખરીદવા છે\"                    -> {\"decision\":\"skip\"} (buy crop)\n"
    "\n"
    "  EXCEPTION — \"buy\" + price word together (e.g. \"buy wheat at what price\",\n"
    "  \"kapas buy bhav\") is rare and ambiguous; prefer the price route only when\n"
    "  the query is dominated by a price word. When in doubt with \"buy\", skip.\n"
    "\n"
    "RULE 4 — PRICE WORD + INFO QUESTION ALWAYS RESOLVES TO DATA:\n"
    "  When NO process trigger from RULE 1/RULE 2/RULE 3 is present AND a price word\n"
    "  is in the query, route to live DB data — never navigation. Even queries\n"
    "  like \"what is X price\" or \"tell me X bhav\" are info requests, NOT\n"
    "  process requests.\n"
    "\n"
    "  Price words: bhav, keemat, kimat, price, rate, ભાવ, કિંમત\n"
    "\n"
    "  Examples:\n"
    "    \"kapas bhav\"            -> clear:crop_price\n"
    "    \"what is wheat price\"  -> clear:crop_price (info, not process)\n"
    "    \"tell me kapas keemat\" -> clear:crop_price\n"
    "    \"buffalo bhav\"          -> clear:buy_sell_product (animals + price)\n"
    "    \"tractor bhav\"          -> ambiguous:equipment_price (kshop+buy_sell)\n"
    "\n"
    "RULE 5 — PRICE WORD + LOCATION (NO OTHER DOMAIN HINT) → CLEAR crop_price:\n"
    "  When the query contains a price word AND a location name (city/taluka/yard\n"
    "  in Gujarat) AND has NO equipment/animal/seed/news/video word, the user is\n"
    "  asking about MANDI / YARD CROP PRICES at that location. This is the most\n"
    "  common Krushi Ratn query pattern. NEVER ask clarification — return CLEAR\n"
    "  crop_price with the location as the keyword.\n"
    "\n"
    "  Reasoning: Krushi Ratn shows ONLY crop prices (mandi/yard) by location.\n"
    "  K-Shop and Buy/Sell listings are not location-filtered. So when a user\n"
    "  asks 'price in <location>', the only sensible interpretation is\n"
    "  crop/mandi price. Clarification is wasted effort.\n"
    "\n"
    "  Common Gujarat location names (cities + talukas — non-exhaustive):\n"
    "    rajkot, ahmedabad, surat, vadodara, baroda, bhavnagar, jamnagar,\n"
    "    junagadh, gandhinagar, anand, mehsana, patan, palanpur, bharuch,\n"
    "    navsari, valsad, amreli, porbandar, kutch, bhuj, morbi, gondal,\n"
    "    mahuva, talaja, dhrol, jetpur, upleta, dhoraji, jasdan, savarkundla,\n"
    "    kalavad, una, kodinar, veraval, mangrol, keshod, visavadar, jamkandorna,\n"
    "    રાજકોટ, અમદાવાદ, સુરત, વડોદરા, ભાવનગર, જામનગર, જૂનાગઢ,\n"
    "    ગાંધીનગર, આણંદ, મહેસાણા, પાટણ, ભરૂચ, નવસારી, વલસાડ, અમરેલી,\n"
    "    પોરબંદર, કચ્છ, ભુજ, મોરબી, ગોંડલ, મહુવા, તળાજા, જેતપુર, ઉપલેટા\n"
    "  Treat any plausible Indian/Gujarati city or taluka name as a location\n"
    "  even if not in the list above (e.g. small talukas, unfamiliar names).\n"
    "\n"
    "  Examples:\n"
    "    \"what is the price available in rajkot\"   -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"rajkot\"}\n"
    "    \"price in surat\"                            -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"surat\"}\n"
    "    \"bhav in mahuva\"                            -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"mahuva\"}\n"
    "    \"રાજકોટ માં ભાવ\"                            -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"રાજકોટ\"}\n"
    "    \"મહુવા ભાવ\"                                  -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"મહુવા\"}\n"
    "\n"
    "  Crop name + price word + location is also CLEAR crop_price (use the CROP as\n"
    "  the keyword — crop is the more useful filter for SQL than the location):\n"
    "    \"onion price in mahuva\"     -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"onion\"}\n"
    "    \"કપાસ ભાવ રાજકોટ\"           -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"કપાસ\"}\n"
    "    \"what is onion price in મહુવા\" -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"onion\"}\n"
    "\n"
    "  EXCEPTIONS — DO NOT apply RULE 5 if:\n"
    "    • Equipment word present (tractor, pump, sprayer, …) → use RULE 4 path\n"
    "    • Animal word present (cow, buffalo, …)               → use RULE 4 path\n"
    "    • News/video/seed word present                        → use normal classification\n"
    "\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "CLASSIFICATION RULES (apply after Golden Rules)\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "\n"
    "SKIP — return {\"decision\":\"skip\"}:\n"
    "  In addition to RULE 1 and RULE 2 above:\n"
    "  • Generic app concepts: \"what is krushi ratn\", \"is app free\",\n"
    "    \"app features\", \"what languages\", \"what is yard\" (concept)\n"
    "  • Greetings only: hello, hi, namaste, kem cho\n"
    "\n"
    "CLEAR — return {\"decision\":\"clear\",\"intent\":\"<X>\",\"keyword\":\"<subject>\"}:\n"
    "  Single unambiguous DB intent. Choose ONE of these intent values:\n"
    "    crop_price | seed_info | equipment_kshop | equipment_used |\n"
    "    kshop_product | buy_sell_product | local_news | video_search\n"
    "\n"
    "  • Crop name + price word                  -> crop_price\n"
    "  • Crop name + seed word (seed/bij/બીજ)    -> seed_info\n"
    "  • Equipment name + NEW signal\n"
    "      (new, brand new, naya, navu, navi, નવું, નવી, नया)\n"
    "                                             -> equipment_kshop\n"
    "  • Equipment name + USED signal\n"
    "      (use, used, old, second hand, juno, juni, junu, junum, junun, juna,\n"
    "       purano, purani, jaani, જૂનું, જૂનો, જૂની, जुना, पुराना)\n"
    "                                             -> equipment_used\n"
    "  • Animal name + price word\n"
    "      (animals only exist in buy/sell — price unambiguates)\n"
    "                                             -> buy_sell_product\n"
    "  • Explicit kshop/k-shop/k-store/કે-શોપ    -> kshop_product\n"
    "  • Explicit buy sell/buysell/marketplace/વેચાણ (browsing intent)\n"
    "                                             -> buy_sell_product\n"
    "  • news/samachar/khabar/ખબર/ન્યૂઝ           -> local_news\n"
    "  • video/વિડિઓ/watch                        -> video_search\n"
    "\n"
    "AMBIGUOUS — return {\"decision\":\"ambiguous\",\"scenario\":\"<X>\",\"keyword\":\"<subject>\"}:\n"
    "  Choose ONE of these scenario values:\n"
    "    crop | equipment | equipment_price | animal | seed |\n"
    "    product | price | location\n"
    "\n"
    "  • Bare crop name with NO price/seed word         -> scenario: crop\n"
    "  • Bare equipment name with NO new/used/price word -> scenario: equipment\n"
    "  • Equipment name + price word (no new/used signal) -> scenario: equipment_price\n"
    "  • Bare animal name with NO price word             -> scenario: animal\n"
    "  • Bare seed word with NO crop attached            -> scenario: seed\n"
    "  • Bare price word with NO subject                 -> scenario: price\n"
    "  • Generic item/product word with NO domain hint   -> scenario: product\n"
    "  • Bare location name with NO subject              -> scenario: location\n"
    "\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "RESPONSE FORMAT — JSON ONLY, NO MARKDOWN, NO EXPLANATION\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "  {\"decision\":\"skip\"}\n"
    "  {\"decision\":\"clear\",\"intent\":\"<intent>\",\"keyword\":\"<subject>\"}\n"
    "  {\"decision\":\"ambiguous\",\"scenario\":\"<scenario>\",\"keyword\":\"<subject>\"}\n"
    "\n"
    "  keyword: main subject as written in the user query (e.g. tractor, ઘઉં, kapas).\n"
    "  Use empty string \"\" only when there is no specific subject (e.g. \"video\").\n"
    "\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "EXAMPLES — STUDY THESE CAREFULLY\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "\n"
    "  Process interrogatives — all skip:\n"
    "    \"how i sell wheat\"             -> {\"decision\":\"skip\"}\n"
    "    \"how to view kapas bhav\"       -> {\"decision\":\"skip\"}\n"
    "    \"where can i find onion price\" -> {\"decision\":\"skip\"}\n"
    "    \"kevi rite pak vechuv\"         -> {\"decision\":\"skip\"}\n"
    "    \"ghau bhav kevi rite jovo\"     -> {\"decision\":\"skip\"}\n"
    "    \"how do i register\"             -> {\"decision\":\"skip\"}\n"
    "    \"kevi rite kshop ma jovu\"      -> {\"decision\":\"skip\"}\n"
    "\n"
    "  Sell intent — all skip:\n"
    "    \"i want to sell wheat\"   -> {\"decision\":\"skip\"}\n"
    "    \"mare cow vechvi che\"    -> {\"decision\":\"skip\"}\n"
    "    \"pak vechuv\"             -> {\"decision\":\"skip\"}\n"
    "\n"
    "  Buy intent — all skip (even with explicit section/crop/animal/equipment):\n"
    "    \"i want to buy wheat\"             -> {\"decision\":\"skip\"}\n"
    "    \"i want to buy kshop products\"    -> {\"decision\":\"skip\"}\n"
    "    \"i want to buy cow from buy/sell\" -> {\"decision\":\"skip\"}\n"
    "    \"what can i do to buy ginger\"     -> {\"decision\":\"skip\"}\n"
    "    \"i want to purchase tractor\"      -> {\"decision\":\"skip\"}\n"
    "    \"mane tractor kharidvu chhe\"      -> {\"decision\":\"skip\"}\n"
    "    \"mare kapas kharidvo chhe\"        -> {\"decision\":\"skip\"}\n"
    "    \"ગાય ખરીદવી છે\"                    -> {\"decision\":\"skip\"}\n"
    "\n"
    "  Crop + price word (info, not process):\n"
    "    \"kapas bhav\"               -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"kapas\"}\n"
    "    \"wheat price\"              -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"wheat\"}\n"
    "    \"ઘઉં ભાવ\"                  -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"ઘઉં\"}\n"
    "    \"what is mango price\"     -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"mango\"}\n"
    "    \"tell me kapas keemat\"   -> {\"decision\":\"clear\",\"intent\":\"crop_price\",\"keyword\":\"kapas\"}\n"
    "\n"
    "  Crop + seed word:\n"
    "    \"wheat seed\"   -> {\"decision\":\"clear\",\"intent\":\"seed_info\",\"keyword\":\"wheat\"}\n"
    "    \"ghau bij\"     -> {\"decision\":\"clear\",\"intent\":\"seed_info\",\"keyword\":\"ghau\"}\n"
    "    \"કપાસ બીજ\"     -> {\"decision\":\"clear\",\"intent\":\"seed_info\",\"keyword\":\"કપાસ\"}\n"
    "\n"
    "  Equipment + new signal:\n"
    "    \"new tractor\"   -> {\"decision\":\"clear\",\"intent\":\"equipment_kshop\",\"keyword\":\"tractor\"}\n"
    "    \"naya pump\"     -> {\"decision\":\"clear\",\"intent\":\"equipment_kshop\",\"keyword\":\"pump\"}\n"
    "    \"નવું મોટર\"     -> {\"decision\":\"clear\",\"intent\":\"equipment_kshop\",\"keyword\":\"મોટર\"}\n"
    "\n"
    "  Equipment + used signal:\n"
    "    \"used motor\"          -> {\"decision\":\"clear\",\"intent\":\"equipment_used\",\"keyword\":\"motor\"}\n"
    "    \"i want use motor\"    -> {\"decision\":\"clear\",\"intent\":\"equipment_used\",\"keyword\":\"motor\"}\n"
    "    \"juno tractor\"         -> {\"decision\":\"clear\",\"intent\":\"equipment_used\",\"keyword\":\"tractor\"}\n"
    "    \"second hand sprayer\" -> {\"decision\":\"clear\",\"intent\":\"equipment_used\",\"keyword\":\"sprayer\"}\n"
    "    \"junum motor\"          -> {\"decision\":\"clear\",\"intent\":\"equipment_used\",\"keyword\":\"motor\"}\n"
    "\n"
    "  Animal + price word:\n"
    "    \"buffalo bhav\" -> {\"decision\":\"clear\",\"intent\":\"buy_sell_product\",\"keyword\":\"buffalo\"}\n"
    "    \"ગાય ભાવ\"      -> {\"decision\":\"clear\",\"intent\":\"buy_sell_product\",\"keyword\":\"ગાય\"}\n"
    "\n"
    "  Bare ambiguous queries:\n"
    "    \"wheat?\"              -> {\"decision\":\"ambiguous\",\"scenario\":\"crop\",\"keyword\":\"wheat\"}\n"
    "    \"tell me about onion\" -> {\"decision\":\"ambiguous\",\"scenario\":\"crop\",\"keyword\":\"onion\"}\n"
    "    \"kapas\"                -> {\"decision\":\"ambiguous\",\"scenario\":\"crop\",\"keyword\":\"kapas\"}\n"
    "    \"tractor\"              -> {\"decision\":\"ambiguous\",\"scenario\":\"equipment\",\"keyword\":\"tractor\"}\n"
    "    \"i want motor\"         -> {\"decision\":\"ambiguous\",\"scenario\":\"equipment\",\"keyword\":\"motor\"}\n"
    "    \"મોટર\"                -> {\"decision\":\"ambiguous\",\"scenario\":\"equipment\",\"keyword\":\"મોટર\"}\n"
    "    \"tractor bhav\"         -> {\"decision\":\"ambiguous\",\"scenario\":\"equipment_price\",\"keyword\":\"tractor\"}\n"
    "    \"pump price\"            -> {\"decision\":\"ambiguous\",\"scenario\":\"equipment_price\",\"keyword\":\"pump\"}\n"
    "    \"cow\"                   -> {\"decision\":\"ambiguous\",\"scenario\":\"animal\",\"keyword\":\"cow\"}\n"
    "    \"i want cow\"            -> {\"decision\":\"ambiguous\",\"scenario\":\"animal\",\"keyword\":\"cow\"}\n"
    "    \"ગાય\"                  -> {\"decision\":\"ambiguous\",\"scenario\":\"animal\",\"keyword\":\"ગાય\"}\n"
    "    \"i want bakri\"         -> {\"decision\":\"ambiguous\",\"scenario\":\"animal\",\"keyword\":\"bakri\"}\n"
    "    \"seed\"                  -> {\"decision\":\"ambiguous\",\"scenario\":\"seed\",\"keyword\":\"seed\"}\n"
    "    \"bij\"                   -> {\"decision\":\"ambiguous\",\"scenario\":\"seed\",\"keyword\":\"bij\"}\n"
    "    \"items?\"                -> {\"decision\":\"ambiguous\",\"scenario\":\"product\",\"keyword\":\"items\"}\n"
    "    \"products\"              -> {\"decision\":\"ambiguous\",\"scenario\":\"product\",\"keyword\":\"products\"}\n"
    "    \"bhav?\"                 -> {\"decision\":\"ambiguous\",\"scenario\":\"price\",\"keyword\":\"bhav\"}\n"
    "    \"keemat\"                -> {\"decision\":\"ambiguous\",\"scenario\":\"price\",\"keyword\":\"keemat\"}\n"
    "    \"surat\"                 -> {\"decision\":\"ambiguous\",\"scenario\":\"location\",\"keyword\":\"surat\"}\n"
    "    \"ગોંડલ\"                -> {\"decision\":\"ambiguous\",\"scenario\":\"location\",\"keyword\":\"ગોંડલ\"}\n"
    "\n"
    "  Generic skip:\n"
    "    \"namaste\"             -> {\"decision\":\"skip\"}\n"
    "    \"what is krushi ratn\" -> {\"decision\":\"skip\"}\n"
    "    \"is app free\"         -> {\"decision\":\"skip\"}\n"
    "\n"
    "  Explicit domain:\n"
    "    \"samachar\"        -> {\"decision\":\"clear\",\"intent\":\"local_news\",\"keyword\":\"samachar\"}\n"
    "    \"video\"            -> {\"decision\":\"clear\",\"intent\":\"video_search\",\"keyword\":\"\"}\n"
    "    \"kshop products\" -> {\"decision\":\"clear\",\"intent\":\"kshop_product\",\"keyword\":\"\"}"
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