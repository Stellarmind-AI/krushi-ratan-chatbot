"""
Intent Router — Zero-cost pre-routing before any LLM call.

For ~30-40% of agricultural queries, the correct table(s) are
OBVIOUS from keywords alone. No LLM needed to figure this out.

Examples:
  "balwan power weeder" → kshop keywords → query_kshop_products
  "kapas bhav surat"    → price keywords  → query_products + query_yards...
  "tractor for sale"    → buy_sell keywords → query_buy_sell_products

This runs before Tool Selector LLM call. If it matches,
we skip LLM call #1 entirely and save ~325 tokens + ~200ms latency.
"""

import re
from typing import List, Optional, Tuple
from app.core.logger import get_logger

logger = get_logger("intent_router")


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION-first patterns: if question contains these, skip SQL routing
# entirely and let the route_agent send it to NAVIGATION flow.
# This prevents "how to buy from k-shop" hitting kshop_product_search.
# ─────────────────────────────────────────────────────────────────────────────
_NAV_OVERRIDE_PATTERNS = [
    "kevi rite", "how to", "how do i", "where is", "where to find",
    "keva steps", "step by step", "vechan mate kevi", "muki shakhu",
    "muki shaku", "mukuv kevi", "nokhi rite", "mate kevi",
]

def _is_navigation_question(q: str) -> bool:
    """Return True if question is asking HOW TO do something — not asking for data."""
    ql = q.lower()
    return any(p in ql for p in _NAV_OVERRIDE_PATTERNS)


ROUTING_RULES = [
    {
        "name": "kshop_product_search",
        # NOTE: only matches "k-shop" or "kshop" when NOT asking HOW TO
        # "how to buy from k-shop" → navigation, not SQL
        "keywords": [
            "kshop products", "k-shop products",
            "power weeder", "weeder",
            "water pump", "motor pump",
            "sprayer", "duster",
            "seeder", "planter", "cultivator",
            "balwaan", "balwan",
            "thresher", "harvester",
            "k-store products", "k store products",
        ],
        "tables": [
            "query_kshop_products", "query_kshop_companies",
            "query_kshop_categories", "query_kshop_weights"
        ],
    },
    {
        "name": "kshop_browse",
        # For "show me k-shop" / "K-Store ma shu che" — browsing, not how-to
        "keywords": [
            "k-store ma", "k store ma", "kshop ma", "k-shop ma",
            "show me k-shop", "show kshop", "kshop batao",
            "k-store batao", "kshop products dikhao",
        ],
        "tables": [
            "query_kshop_products", "query_kshop_companies",
            "query_kshop_categories", "query_kshop_weights"
        ],
    },
    {
        "name": "crop_price",
        "keywords": [
            "bhav", "ભાવ", "mandi bhav", "market price", "yard bhav",
            "kapas", "કપાસ", "wheat", "ઘઉં", "ghau",
            # NOTE: "rice" REMOVED — it's a substring of "price" causing false positives
            # Use "rice crop", "chaval", "ચોખા" instead
            "rice crop", "chaval", "ચોખા",
            "jowar", "bajra", "bajri", "corn",
            "aaj nu bhav", "today price", "magfali bhav",
            "tal bhav", "chana bhav", "ghana bhav",
        ],
        "tables": [
            "query_products", "query_sub_categories",
            "query_yards", "query_cities", "query_weights"
        ],
    },
    {
        "name": "buy_sell",
        "keywords": [
            "buy sell", "buysell", "for sale",
            "animal", "pashu", "cow", "gai", "buffalo", "bhens",
            "land", "jamin",
            "vechuv valo", "vechavanu", "sale mate muku",
        ],
        "tables": [
            "query_buy_sell_products", "query_buy_sell_categories", "query_users"
        ],
    },
    {
        "name": "video_search",
        "keywords": ["video", "watch", "juo", "farming video", "kheti video", "tutorial"],
        "tables": ["query_video_posts", "query_users", "query_video_categories"],
    },
    {
        "name": "news_search",
        "keywords": ["news", "samachar", "update", "khabar", "akhbar"],
        "tables": ["query_news", "query_cities", "query_states"],
    },
    {
        "name": "seed_search",
        "keywords": ["seed", "bij", "બીજ", "variety"],
        "tables": ["query_seeds", "query_sub_categories"],
    },
    {
        "name": "order_lookup",
        "keywords": ["order", "my order", "purchase history", "order status", "pending order"],
        "tables": ["query_kshop_orders", "query_kshop_products", "query_order_statuses"],
    },
]


class IntentRouter:
    """
    Fast keyword-based router. Runs in microseconds, costs zero tokens.
    Use BEFORE Tool Selector LLM to skip LLM call #1 for obvious queries.
    """

    def route(self, question: str) -> Optional[Tuple[List[str], str]]:
        """
        Returns (table_list, rule_name) if confident match, else None.

        Respects navigation questions: if the question is HOW-TO style,
        we skip SQL routing so route_agent can send it to NAVIGATION flow.
        """
        q = question.lower()

        # If question is "how to X" style, don't short-circuit to SQL tables
        if _is_navigation_question(q):
            logger.info(f"⚡ IntentRouter: navigation question detected — skipping SQL pre-route")
            return None

        for rule in ROUTING_RULES:
            for keyword in rule["keywords"]:
                if keyword.lower() in q:
                    logger.info(f"⚡ Intent routed without LLM: {rule['name']} (keyword='{keyword}')")
                    return rule["tables"], rule["name"]
        return None


intent_router = IntentRouter()

def get_intent_router() -> IntentRouter:
    return intent_router