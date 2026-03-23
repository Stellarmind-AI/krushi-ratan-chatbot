"""
Confirmation Layer (F1) — Confidence-scored, multilingual, keyword-aware.

Handles queries in:
  - English         (thresher, wheat, kapas bhav)
  - Romanized Gujarati  (thresar, kapas, ghau bhav)
  - Gujarati script     (થ્રેસર, ઘઉં, કપાસ ભાવ)

THREE behaviours:

1. CONFIDENCE SCORING (>= 80% → bypass F1, inject intent directly)
   Domain signals scored per language. High-confidence = direct answer.

2. SCENARIO MATCHING (< 80% confidence → pause, show options)
   Trigger keywords per scenario in all three languages/scripts.

3. SMART OPTION ORDERING
   Options ordered by what the query hints at (sell/buy/price/seed).
   Irrelevant options excluded (e.g. veggie crops don't show seed option).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Union
from app.core.logger import get_logger

logger = get_logger("confirmation_layer")

CONFIDENCE_THRESHOLD = 0.80

# ─────────────────────────────────────────────────────────────────────────────
# Navigation / HOW-TO signals — if ANY of these appear, F1 must NOT trigger.
# These are questions about HOW TO USE the app, not requests for data.
# Route agent already handles them correctly as NAVIGATION flow.
# English / Romanized Gujarati / Gujarati script all covered.
# ─────────────────────────────────────────────────────────────────────────────
_NAV_SIGNALS: List[str] = [
    # English — generic how-to
    "how to", "how do i", "how can i", "step by step", "steps to",
    "where to find", "where is", "how do", "guide", "tutorial for",
    "set up", "setup", "configure", "enable", "activate",
    # English — switch role (nav_switch_role)
    "switch account", "switch role", "role change", "farmer company switch",
    "change role", "account switch",
    # English — video creator / upload (nav_video_create, nav_video_upload)
    "video creator", "create video", "upload video", "video banavu",
    # English — mobile number (nav_mobile_number_update)
    "mobile number change", "mobile number update", "change mobile number",
    "verify mobile", "number verification", "change phone number",
    # English — customer support (nav_customer_support)
    "contact support", "customer support", "help and support",
    "contact us", "helpline", "customer care",
    # Romanized Gujarati — generic how-to
    "kevi rite", "kevi reet", "kevi reete", "keva steps",
    "kyay malse", "kyay che", "shu karvanu", "shikho",
    "mate shu", "kevi rite karvanu", "kevi rite karu",
    "kevi rite set", "kevi rite nokhi", "kevi rite muku",
    "mate kevi", "keva step", "process shu", "kevi rite thay",
    # Gujarati script — generic how-to
    "કેવી રીતે", "કઈ રીતે", "કેવી રીત", "સ્ટેપ્સ",
    "ક્યાં મળશે", "ક્યાં છે", "કેવી રીતે કરવું", "સેટ કરવું",
    "કેવી રીતે સેટ", "કેવી રીતે ચાલુ", "કેવી રીતે ઉપયોગ",
    "કેવી રીતે ખરીદ", "કેવી રીતે વેચ", "કેવી રીતે નોંધ",
    "કેવી રીતે મૂક", "કેવી રીતે ચેક", "કેવી રીતે ઉમેર",
    "શીખો", "ગાઇડ", "પ્રક્રિયા",
    # Gujarati script — switch role (nav_switch_role)
    "ભૂમિકા સ્વિચ", "સ્વિચ એકાઉન્ટ", "ભૂમિકા બદલો",
    # Gujarati script — video creator / upload (nav_video_create, nav_video_upload)
    "વિડિઓ ક્રિએટર", "અપલોડ", "વિડિઓ અપલોડ",
    # Gujarati script — mobile number (nav_mobile_number_update)
    "મોબાઇલ નંબર બદલો", "નંબર બદલો", "મોબાઇલ અપડેટ",
    # Gujarati script — customer support (nav_customer_support)
    "ગ્રાહક સેવા", "સપોર્ટ", "સહાય", "સંપર્ક",
]


def _is_navigation_query(q: str) -> bool:
    """
    Returns True if the query is asking HOW TO do something
    rather than requesting data. F1 must never intercept these.
    """
    return any(sig in q for sig in _NAV_SIGNALS)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

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
    """Confidence >= 80% — skip F1 UI, inject intent directly."""
    intent_key: str
    confidence: float
    domain:     str


# ─────────────────────────────────────────────────────────────────────────────
# Intent → table mapping
# ─────────────────────────────────────────────────────────────────────────────

INTENT_TO_TABLES: Dict[str, List[str]] = {
    "crop_price":       ["query_products", "query_sub_categories", "query_yards", "query_cities", "query_weights"],
    "kshop_product":    ["query_kshop_products", "query_kshop_companies", "query_kshop_categories", "query_kshop_weights"],
    "buy_sell_product": ["query_buy_sell_products", "query_buy_sell_categories", "query_users"],
    "seed_info":        ["query_seeds", "query_sub_categories"],
    "local_news":       ["query_news", "query_cities", "query_states"],
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


# ─────────────────────────────────────────────────────────────────────────────
# Confidence scoring — domain signals in EN + Romanized GU + Gujarati script
# High = 0.90, Medium = 0.75
# If best score >= CONFIDENCE_THRESHOLD (0.80) → bypass F1
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_SIGNALS: Dict[str, Dict[str, List[str]]] = {
    "crop_price": {
        "high": [
            # English
            "bhav", "mandi bhav", "yard bhav", "market price",
            "aaj nu bhav", "today price", "mandi", "yard price",
            "ni keemat", "no bhav", "nu bhav", "na bhav",
            # Gujarati script
            "ભાવ", "મંડી ભાવ", "ભાવ જોઈએ", "ભાવ બતાઓ", "ભાવ આપો",
            "આજ નો ભાવ", "આજ નું ભાવ", "મંડી", "ભાવ શું છે",
            "ભાવ કેટલો", "ભાવ કેટલા", "ભાવ જણાવો",
        ],
        "medium": [
            "price", "keemat", "rate", "mol", "daam",
            "કિંમત", "રેટ", "ભાવ",
        ],
    },
    "kshop_product": {
        "high": [
            "kshop", "k-shop", "k shop", "k-store", "kstore", "k store",
            "કે-શોપ", "કે શોપ", "કૃષિ શોપ",
        ],
        "medium": ["online shop", "online store"],
    },
    "buy_sell_product": {
        "high": [
            # English / Romanized
            "buy sell", "buysell", "buy/sell", "marketplace", "for sale",
            "vechuv", "vecho", "vechan", "sale karvanu", "vecho chhe",
            "sell my", "sell karvanu",
            # Gujarati script
            "વેચવું", "વેચો", "વેચાણ", "ખરીદ-વેચ", "ખરીદ વેચ",
            "વેચવા", "વેચવું છે", "વેચુ છુ", "વેચવો",
        ],
        "medium": ["sale", "sell", "second hand", "used", "વેચ", "ખરીદ"],
    },
    "seed_info": {
        "high": [
            "seed", "bij", "bia", "variety", "beej", "varieti",
            "બીજ", "બી", "વેરાઈટી", "જાત", "નાસ",
        ],
        "medium": ["nasal", "jat", "નાસ", "જાત"],
    },
    "local_news": {
        "high": [
            "news", "samachar", "khabar", "akhbar", "latest news",
            "સમાચાર", "ખબર", "અખબાર", "ન્યૂઝ", "સમાચારો",
        ],
        "medium": ["update", "notification", "અપડેટ"],
    },
    "video_search": {
        "high": [
            "video", "watch", "juo", "tutorial", "farming video", "kheti video",
            "વિડિઓ", "વીડિઓ", "જુઓ", "વિડીઓ",
        ],
        "medium": ["clip", "reel", "ક્લિપ"],
    },
}

# Buy / sell / price / seed hints — used for option ordering
_BUY_HINTS = [
    "levu", "kharidi", "kharido", "buy", "purchase", "joiye", "joiyu", "levo", "apo",
    "લેવું", "ખરીદો", "ખરીદી", "ખરીદવું", "જોઈએ", "આપો",
]
_SELL_HINTS = [
    "vechuv", "vecho", "sell", "vechan", "sale", "muku", "mukuv",
    "વેચવું", "વેચો", "વેચાણ", "વેચ", "મૂકો", "મૂકવું",
]
_PRICE_HINTS = [
    "bhav", "keemat", "rate", "price", "kitno", "ketla", "mol",
    "ભાવ", "કિંમત", "રેટ", "કેટલો", "કેટલા",
]
_SEED_HINTS = [
    "seed", "bij", "variety", "nasal",
    "બીજ", "વેરાઈટી", "જાત",
]


def _score_query(q: str) -> Optional[tuple]:
    """
    Score query against all domain signals (all languages).
    Returns (best_intent_key, confidence) if any >= threshold, else None.
    """
    scores: Dict[str, float] = {}
    for intent_key, signals in _DOMAIN_SIGNALS.items():
        for sig in signals["high"]:
            if sig in q:
                scores[intent_key] = max(scores.get(intent_key, 0.0), 0.90)
        for sig in signals.get("medium", []):
            if sig in q:
                scores[intent_key] = max(scores.get(intent_key, 0.0), 0.75)
    if not scores:
        return None
    best = max(scores, key=lambda k: scores[k])
    return (best, scores[best])


# ─────────────────────────────────────────────────────────────────────────────
# Trigger keyword lists — EN + Romanized GU + Gujarati script
# ─────────────────────────────────────────────────────────────────────────────

# Scenario 1: Crop names
_CROP_KEYWORDS: List[str] = [
    # English / Romanized
    "wheat", "ghau", "gahu",
    "kapas", "cotton",
    "bajra", "bajri", "bajro",
    "jowar", "jwari", "jwar",
    "corn", "maize", "makai",
    "magfali", "groundnut", "moongfali",
    "mung", "moong",
    "chana", "ghana", "channa",
    "tal", "sesame",
    "rice crop", "chaval",
    "soybean", "soya",
    "onion", "dungli", "kanda",
    "garlic", "lasan",
    "tomato", "tameta",
    "potato", "bataka",
    "sugarcane", "sherdio",
    # Gujarati script
    "ઘઉં", "ઘઉ",
    "કપાસ",
    "બાજરો", "બાજરી",
    "જુવાર",
    "મકાઈ",
    "મગફળી",
    "મગ",
    "ચણા",
    "તલ",
    "ચોખા", "ડાંગર",
    "સોયાબીન",
    "ડુંગળી",
    "લસણ",
    "ટામેટા", "ટામેટું",
    "બટાકા", "બટેટા",
    "શેરડી",
    "તુવર", "અડદ", "મઠ",
]

# Scenario 2: Generic product
_PRODUCT_KEYWORDS: List[str] = [
    "product", "products", "item", "items",
    "vastu", "vasthu",
    "ઉત્પાદ", "ઉત્પાદન", "વસ્તુ",
]

# Scenario 3: Price without source
_PRICE_KEYWORDS: List[str] = [
    "how much", "kitna", "kitno",
    "ketla", "ketlu", "mool",
    "kem malshe", "kya bhav",
    "કેટલો", "કેટલા", "કેટલું",
    "કિંમત", "ભાવ",          # standalone without domain context → ambiguous
    "shuno bhav", "shu bhav",
]

# Scenario 4: Equipment / machinery
_EQUIPMENT_KEYWORDS: List[str] = [
    # English / Romanized
    "machine", "yantra",
    "tractor",
    "pump", "motor pump", "water pump",
    "sprayer", "duster",
    "engine", "implement",
    "harvester", "thresher", "thresar", "thraser",
    "auger", "cultivator", "planter",
    "weeder", "seeder",
    "balwan", "balwaan",
    # Gujarati script
    "મશીન", "યંત્ર",
    "ટ્રેક્ટર",
    "પંપ", "મોટર પંપ", "વોટર પંપ",
    "સ્પ્રેયર", "ફ્વારો", "ડસ્ટર",
    "એન્જિન", "ઈમ્પ્લીમેન્ટ",
    "થ્રેસર", "થ્રેશર", "હાર્વેસ્ટર",
    "ઓગર", "કલ્ટીવેટર", "પ્લાન્ટર",
    "વીડર", "સીડર",
    "બળવાન",
]

# Scenario 5: Location
_LOCATION_KEYWORDS: List[str] = [
    # English / Romanized
    "surat", "ahmedabad", "rajkot", "vadodara", "mehsana",
    "gandhinagar", "anand", "bharuch", "junagadh",
    "bhavnagar", "jamnagar", "amreli", "navsari", "valsad",
    "kutch", "morbi", "patan", "sabarkantha", "banaskantha",
    "nearby", "nazdik", "local",
    "mara gaon", "mara taluka", "mara jilla",
    # Gujarati script
    "સુરત", "અમદાવાદ", "રાજકોટ", "વડોદરા", "મહેસાણા",
    "ગાંધીનગર", "આણંદ", "ભરૂચ", "જૂનાગઢ",
    "ભાવનગર", "જામનગર", "અમરેલી", "નવસારી", "વલસાડ",
    "કચ્છ", "મોરબી", "પાટણ",
    "નજીક", "સ્થાનિક", "મારા ગામ", "મારા તાલુકા",
]


# ─────────────────────────────────────────────────────────────────────────────
# Domain availability per crop keyword
#
# KEY DESIGN RULE: Only show an option if data ACTUALLY EXISTS in that domain
# for that crop. This prevents "no data found" responses after the user picks.
#
# Domains:
#   crop_price       → products/mandi table — has prices for ALL major crops
#   buy_sell_product → buy_sell_products table — farmers list anything for sale
#   seed_info        → seeds table — only crops where seed varieties are stocked
#
# NOTE: kshop_product is NEVER shown for raw crops. K-Shop sells farm
# EQUIPMENT (tractors, pumps, sprayers) — NOT raw agricultural produce.
# ─────────────────────────────────────────────────────────────────────────────

# Crops that have seed variety data in the seeds table
# (conservative whitelist — only add if you know data exists)
_CROPS_WITH_SEED_DATA: set = {
    "wheat", "ghau", "gahu", "ઘઉં", "ઘઉ",
    "kapas", "cotton", "કપાસ",           # cotton seeds are stocked
    "bajra", "bajri", "bajro", "બાજરો", "બાજરી",
    "jowar", "jwari", "jwar", "જુવાર",
    "corn", "maize", "makai", "મકાઈ",
    "mung", "moong", "મગ",
    "chana", "ghana", "channa", "ચણા",
    "tal", "sesame", "તલ",
    "soybean", "soya", "સોયાબીન",
    "rice crop", "chaval", "ચોખા", "ડાંગર",
    "magfali", "groundnut", "moongfali", "મગફળી",
}

# Crops that are ONLY tradeable (no seeds data, no processing equipment)
# These get only price + buy_sell options
_CROPS_PRICE_ONLY: set = {
    "onion", "dungli", "kanda", "ડુંગળી",
    "tomato", "tameta", "ટામેટા", "ટામેટું",
    "potato", "bataka", "bateta", "બટાકા", "બટેટા",
    "garlic", "lasan", "લસણ",
    "sugarcane", "sherdio", "શેરડી",
    "tuveral", "tuver", "તુવેર",
    "adadal", "adad", "અડદ",
}


def _build_crop_options(kw: str, q: str) -> List[ClarificationOption]:
    """
    Build crop options with only the domains that HAVE data for this keyword.
    - K-Shop is NEVER shown for crops (K-Shop = equipment, not raw crops)
    - Seeds only shown if crop is in _CROPS_WITH_SEED_DATA whitelist
    - Options ordered by what the query context hints at
    """
    k       = kw.capitalize()
    kw_low  = kw.lower()
    has_sell = any(h in q for h in _SELL_HINTS)
    has_seed = any(h in q for h in _SEED_HINTS)

    # Normalize kw_low to also check English equivalents for Gujarati input
    # so "કપાસ" resolves seed_data correctly
    _GU_TO_EN = {
        "કપાસ": "kapas", "ઘઉં": "wheat", "ઘઉ": "wheat",
        "બાજરો": "bajra", "બાજરી": "bajra", "જુવાર": "jowar",
        "મકાઈ": "corn", "મગ": "mung", "ચણા": "chana",
        "તલ": "tal", "ચોખા": "chaval", "ડાંગર": "chaval",
        "સોયાબીન": "soybean", "મગફળી": "magfali",
        "ડુંગળી": "onion", "ટામેટા": "tomato", "ટામેટું": "tomato",
        "બટાકા": "potato", "બટેટા": "potato", "લસણ": "lasan",
        "શેરડી": "sugarcane",
    }
    kw_check = _GU_TO_EN.get(kw_low, kw_low)  # map Gujarati → English for set lookups

    # Base options — always available for crops
    opt_price = ClarificationOption(f"Check {k} mandi price",           "📈", "crop_price",       "crop_price")
    opt_sell  = ClarificationOption(f"View {k} buy/sell listings",      "📦", "buy_sell_product", "buy_sell")

    # Seed option — only if this crop actually has seed data
    has_seed_data = kw_low in _CROPS_WITH_SEED_DATA or kw_check in _CROPS_WITH_SEED_DATA
    opt_seed = ClarificationOption(f"{k} seed variety info", "🌱", "seed_info", "seeds") if has_seed_data else None

    # Price-only crops — don't show seed option
    if kw_low in _CROPS_PRICE_ONLY or kw_check in _CROPS_PRICE_ONLY:
        opt_seed = None

    # Build ordered list based on query hints
    if has_sell:
        opts = [opt_sell, opt_price]
        if opt_seed and has_seed:
            opts.append(opt_seed)
    elif has_seed and opt_seed:
        opts = [opt_seed, opt_price, opt_sell]
    else:
        # Default: price first (most common query)
        opts = [opt_price, opt_sell]
        if opt_seed:
            opts.append(opt_seed)

    return [o for o in opts if o is not None]


def _build_product_options(kw: str, q: str) -> List[ClarificationOption]:
    """Generic product — could be equipment (K-Shop) or marketplace or crop price."""
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
    found_crop = next((c for c in _CROP_KEYWORDS if c in q), None)
    label_crop = f"{found_crop.capitalize()} price at mandi / yard" if found_crop else "Crop price at mandi / yard"
    return [
        ClarificationOption(label_crop,               "📊", "crop_price",       "crop_price"),
        ClarificationOption("K-Shop product price",   "🛒", "kshop_product",    "kshop"),
        ClarificationOption("Buy/Sell listing price", "💰", "buy_sell_product", "buy_sell"),
    ]


def _build_equipment_options(kw: str, q: str) -> List[ClarificationOption]:
    k = kw.capitalize()
    has_used = any(h in q for h in ["used", "second hand", "juno", "purano", "old", "જૂનું", "જૂના", "પૂરાણું"])
    opt_new  = ClarificationOption(f"New {k} from K-Shop",              "🏪", "equipment_kshop", "kshop")
    opt_used = ClarificationOption(f"Used {k} on Buy/Sell marketplace", "🔄", "equipment_used",  "buy_sell")
    return [opt_used, opt_new] if has_used else [opt_new, opt_used]


def _build_location_options(kw: str, q: str) -> List[ClarificationOption]:
    k = kw.capitalize()
    has_crop = any(c in q for c in _CROP_KEYWORDS)
    opt_price = ClarificationOption(f"Crop prices near {k}",       "📈", "crop_price",       "crop_price")
    opt_news  = ClarificationOption(f"Agricultural news from {k}", "📰", "local_news",       "news")
    opt_sell  = ClarificationOption(f"Buy/Sell listings in {k}",   "🏘️", "buy_sell_product", "buy_sell")
    return [opt_price, opt_news, opt_sell] if has_crop else [opt_news, opt_price, opt_sell]


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class ConfirmationLayer:
    """
    Stateless. Call .check(user_query) on every incoming query.

    Returns:
        ConfirmedIntent      — confidence >= 80%, skip F1, inject directly
        ClarificationRequest — confidence < 80%, show options
        None                 — no ambiguity, proceed normally
    """

    def check(self, user_query: str) -> Optional[Union[ClarificationRequest, ConfirmedIntent]]:
        # Do NOT lowercase Gujarati script — Gujarati has no case.
        # Keep original for Gujarati keyword matching, use lower only for English.
        q_orig  = user_query.strip()
        q_lower = q_orig.lower()
        # Combined string for matching — catches both scripts
        q = q_lower + " " + q_orig

        # ── Step 0: Navigation bypass ─────────────────────────────────────
        # If the query is a HOW-TO / navigation question, F1 must NOT trigger.
        # The route_agent handles these as NAVIGATION flow correctly.
        # Examples: "kevi rite set karvanu", "કેવી રીતે સેટ કરવું", "how to enable"
        if _is_navigation_query(q):
            logger.info(f"✅ F1 NAV BYPASS — navigation question detected: {user_query[:60]!r}")
            return None

        # ── Step 1: Confidence scoring ────────────────────────────────────
        scored = _score_query(q)
        if scored:
            intent_key, confidence = scored
            if confidence >= CONFIDENCE_THRESHOLD:
                logger.info(f"✅ F1 BYPASSED | intent={intent_key} confidence={confidence:.0%}")
                return ConfirmedIntent(intent_key=intent_key, confidence=confidence, domain=intent_key)

        # ── Step 2: Scenario keyword matching ────────────────────────────

        # Scenario 1: Crop name
        for kw in _CROP_KEYWORDS:
            if kw in q:
                logger.info(f"🔔 F1 triggered | scenario=crop_name keyword='{kw}'")
                return ClarificationRequest(
                    question=f"What would you like to know about '{kw.capitalize()}'?",
                    options=_build_crop_options(kw, q),
                    scenario="crop_name",
                    matched_keyword=kw,
                )

        # Scenario 2: Generic product
        for kw in _PRODUCT_KEYWORDS:
            if kw in q:
                logger.info(f"🔔 F1 triggered | scenario=generic_product keyword='{kw}'")
                return ClarificationRequest(
                    question="Which section are you looking in?",
                    options=_build_product_options(kw, q),
                    scenario="generic_product",
                    matched_keyword=kw,
                )

        # Scenario 3: Price without source
        for kw in _PRICE_KEYWORDS:
            if kw in q:
                logger.info(f"🔔 F1 triggered | scenario=price_query keyword='{kw}'")
                return ClarificationRequest(
                    question="Which price are you asking about?",
                    options=_build_price_options(kw, q),
                    scenario="price_query",
                    matched_keyword=kw,
                )

        # Scenario 4: Equipment
        for kw in _EQUIPMENT_KEYWORDS:
            if kw in q:
                logger.info(f"🔔 F1 triggered | scenario=equipment keyword='{kw}'")
                return ClarificationRequest(
                    question=f"Are you looking for new or used {kw}?",
                    options=_build_equipment_options(kw, q),
                    scenario="equipment_query",
                    matched_keyword=kw,
                )

        # Scenario 5: Location
        for kw in _LOCATION_KEYWORDS:
            if kw in q:
                logger.info(f"🔔 F1 triggered | scenario=location keyword='{kw}'")
                return ClarificationRequest(
                    question=f"What are you looking for related to '{kw}'?",
                    options=_build_location_options(kw, q),
                    scenario="location_query",
                    matched_keyword=kw,
                )

        logger.info("✅ ConfirmationLayer: no ambiguity — proceeding")
        return None

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def get_confirmed_tables(intent_key: str) -> List[str]:
        tables = INTENT_TO_TABLES.get(intent_key, [])
        if not tables:
            logger.warning(f"⚠️ Unknown intent_key '{intent_key}'")
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


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[ConfirmationLayer] = None

def get_confirmation_layer() -> ConfirmationLayer:
    global _instance
    if _instance is None:
        _instance = ConfirmationLayer()
    return _instance