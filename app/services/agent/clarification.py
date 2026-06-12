"""
Clarification — code-built buttons for ambiguous queries + frame patching.

NO LLM here. Stage 1 (nlu.py) marks a frame ambiguous with a scenario; this
module builds the clarification question + options (migrated from the old
confirmation_layer), and applies the user's button choice as a deterministic
PATCH to the stored frame (no Stage-1 re-run → no loop risk, no extra cost).

Also owns the legacy intent→tables / intent→prompt-note maps used by the SQL
flow until Phase 3 makes it frame-native.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from app.models.nlu_frame import NLUFrame, EquipmentEntity
from app.core.logger import get_logger

logger = get_logger("clarification")


# ── Wire-format dataclasses (shape unchanged — frontend + run_audit rely on it) ──

@dataclass
class ClarificationOption:
    label:      str
    emoji:      str
    intent_key: str
    domain:     str


@dataclass
class ClarificationRequest:
    question:        str
    options:         List[ClarificationOption]
    scenario:        str
    matched_keyword: str = ""


NAV_INTENT_KEY = "navigation"


# ── Legacy intent → tables / prompt-note maps (Phase-1 SQL bridge) ───────────
# Keys are LEGACY intent keys (frame.legacy_intent_key maps news→local_news,
# video→video_search). Removed in Phase 3 when table selection is frame-native.

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


def get_confirmed_tables(intent_key: str) -> List[str]:
    tables = INTENT_TO_TABLES.get(intent_key, [])
    if not tables:
        logger.warning(f"Unknown intent_key '{intent_key}' — no tables mapped")
    return tables


def get_intent_note(intent_key: str) -> str:
    return INTENT_TO_PROMPT_NOTE.get(intent_key, "")


# ── Option builders (migrated unchanged from confirmation_layer) ─────────────

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
    template = _NAV_LABEL_BY_SCENARIO.get(scenario)
    if not template:
        return None
    label = template.format(kw=keyword) if "{kw}" in template and keyword else (
        template.replace("{kw} ", "").replace(" {kw}", "") if "{kw}" in template else template
    )
    return ClarificationOption(label=label, emoji="🧭", intent_key=NAV_INTENT_KEY, domain="navigation")


def build_navigation_query(scenario: str, keyword: str = "") -> str:
    """When the user taps the navigation button, build a complete how-to question."""
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
        ClarificationOption(f"New {k} price (K-Shop)",    "🏪", "equipment_kshop", "kshop"),
        ClarificationOption(f"Used {k} price (Buy/Sell)", "🔄", "equipment_used",  "buy_sell"),
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
        ClarificationOption("Buy/Sell marketplace (farmer listings)", "🔄", "buy_sell_product", "buy_sell"),
        ClarificationOption("Mandi / yard crop prices",               "📊", "crop_price",       "crop_price"),
    ]


def _build_price_options(kw: str) -> List[ClarificationOption]:
    label_crop = f"{kw.capitalize()} price at mandi/yard" if kw else "Crop price at mandi/yard"
    return [
        ClarificationOption(label_crop,               "📊", "crop_price",       "crop_price"),
        ClarificationOption("K-Shop product price",   "🛒", "kshop_product",    "kshop"),
        ClarificationOption("Buy/Sell listing price", "💰", "buy_sell_product", "buy_sell"),
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


# ── Public API ───────────────────────────────────────────────────────────────

def build_clarification(frame: NLUFrame) -> Optional[ClarificationRequest]:
    """Build the clarification request for an ambiguous frame. Code-only."""
    scenario = frame.ambiguity_scenario or "product"
    keyword = frame.primary_keyword()
    builder = _SCENARIO_BUILDERS.get(scenario, _build_product_options)
    options = [o for o in builder(keyword) if o is not None]
    if not options:
        return None
    question = _SCENARIO_QUESTIONS.get(scenario, "What are you looking for?").format(
        keyword=keyword or "this"
    )
    return ClarificationRequest(
        question=question, options=options,
        scenario=scenario, matched_keyword=keyword,
    )


def serialize_request(req: ClarificationRequest) -> dict:
    """Wire format — IDENTICAL to the old confirmation_layer payload."""
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


def apply_choice(frame: NLUFrame, intent_key: str) -> NLUFrame:
    """Apply the user's button choice as a deterministic PATCH to the frame.

    No LLM, no Stage-1 re-run. The patched frame resumes at routing, so a
    second clarification for the same query is structurally impossible
    (max-1 guard by construction).
    """
    scenario = frame.ambiguity_scenario or "product"
    keyword = frame.primary_keyword()

    if intent_key == NAV_INTENT_KEY:
        frame.intent = "navigation"
        frame.query_type = "general_knowledge"
        # The bare original ("tractor") is too vague for the navigation
        # handler — substitute a complete how-to question.
        frame.question_en = build_navigation_query(scenario, keyword)
    else:
        # Buttons carry legacy keys for news/video — map to the new taxonomy.
        frame.intent = {"local_news": "news", "video_search": "video"}.get(
            intent_key, intent_key
        )
        # Equipment condition is implied by the chosen domain.
        condition = {"equipment_kshop": "new", "equipment_used": "used"}.get(intent_key)
        if condition:
            if frame.equipment:
                for eq in frame.equipment:
                    eq.condition = eq.condition or condition
            elif keyword:
                frame.equipment = [EquipmentEntity(name=keyword, condition=condition)]

    frame.ambiguity_scenario = None
    frame.intent_confidence = "high"
    logger.info(f"🎯 CLARIFICATION PATCH | picked={intent_key} → intent={frame.intent}")
    return frame
