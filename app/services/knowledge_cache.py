"""
Knowledge Cache — In-memory cache of database content for routing decisions.

WHY THIS EXISTS:
The bot needs to know what's actually in the database to route queries correctly.
For example:
  - "wheat" → user wants crop prices (sub_categories has 'wheat')
  - "balwan power weeder" → user wants K-Shop product (kshop_products has 'balwan')
  - "seeds" → user wants to BROWSE the seeds catalog (table name, not a product)

Hardcoding these lists is fragile — every new product needs a code change.
This cache reads the database at startup and refreshes every 30 minutes.

USAGE:
    from app.services.knowledge_cache import knowledge_cache
    
    if knowledge_cache.is_crop("wheat"):     # True
    if knowledge_cache.is_kshop_product("weeder"):  # True (substring match)
    if knowledge_cache.is_browse_keyword("seeds"):  # True (it's a table name)
    
    crops = knowledge_cache.get_all_crops()  # set of crop names
"""

import asyncio
from typing import Set, Optional
from app.core.database import db_manager
from app.core.logger import get_logger

logger = get_logger("knowledge_cache")

# Refresh interval in seconds (30 minutes)
_REFRESH_INTERVAL = 30 * 60

# Words that are TABLE NAMES, not actual product/crop names.
# When the user says these, they're browsing a category, not searching for an item.
# Example: "I want to buy seeds" = browse seeds catalog, not find a seed named "seeds"
_BROWSE_KEYWORDS: Set[str] = {
    "seeds", "seed", "બીજ",
    "products", "product", "items", "item", "ઉત્પાદ", "વસ્તુ",
    "crops", "crop", "પાક",
    "videos", "video", "વિડિઓ",
    "news", "samachar", "khabar", "સમાચાર",
    "prices", "bhav", "ભાવ",  # generic — not tied to a specific crop
    "categories", "category", "શ્રેણી",
}


class KnowledgeCache:
    """
    In-memory cache of database content (crops, K-Shop products, categories).
    
    Loaded at server startup. Refreshes every 30 minutes in the background.
    Used by intent_router, query_generator, and confirmation_layer.
    """

    def __init__(self):
        self._crops: Set[str] = set()             # sub_categories.name (lowercased)
        self._kshop_products: Set[str] = set()    # kshop_products.name (lowercased, tokenized)
        self._kshop_categories: Set[str] = set()  # kshop_categories.name (lowercased)
        self._seed_varieties: Set[str] = set()    # seeds.name (lowercased)
        self._loaded = False
        self._refresh_task: Optional[asyncio.Task] = None

    # ─────────────────────────────────────────────────────────────────────
    # Public lookup methods — used by intent_router, query_generator, etc.
    # ─────────────────────────────────────────────────────────────────────

    def is_crop(self, keyword: str) -> bool:
        """True if keyword matches a crop name (sub_categories table)."""
        if not keyword:
            return False
        kw = keyword.lower().strip()
        # Exact match or substring match
        if kw in self._crops:
            return True
        # Check if any crop contains this keyword (e.g. "kapas" in "kapas hybrid")
        return any(kw in crop or crop in kw for crop in self._crops)

    def is_kshop_product(self, keyword: str) -> bool:
        """True if keyword matches a K-Shop product or category name."""
        if not keyword:
            return False
        kw = keyword.lower().strip()
        # Check products
        if any(kw in prod for prod in self._kshop_products):
            return True
        # Check categories
        if any(kw in cat for cat in self._kshop_categories):
            return True
        return False

    def is_browse_keyword(self, keyword: str) -> bool:
        """
        True if the keyword is a TABLE NAME (browse intent), not a specific item.
        Example: "seeds" → browse seeds catalog (not find a seed named 'seeds').
        """
        if not keyword:
            return False
        return keyword.lower().strip() in _BROWSE_KEYWORDS

    def is_seed_variety(self, keyword: str) -> bool:
        """True if keyword matches an actual seed variety name."""
        if not keyword:
            return False
        kw = keyword.lower().strip()
        return any(kw in seed for seed in self._seed_varieties)

    # ─────────────────────────────────────────────────────────────────────
    # Bulk getters — for prompts and debugging
    # ─────────────────────────────────────────────────────────────────────

    def get_all_crops(self) -> Set[str]:
        return self._crops.copy()

    def get_all_kshop_categories(self) -> Set[str]:
        return self._kshop_categories.copy()

    def get_summary(self) -> str:
        """Return a one-line summary for logs."""
        return (
            f"crops={len(self._crops)} "
            f"kshop_products={len(self._kshop_products)} "
            f"kshop_categories={len(self._kshop_categories)} "
            f"seed_varieties={len(self._seed_varieties)}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Prompt snippets — injected into LLM prompts so the LLM knows what's
    # actually in the database, instead of guessing literal column matches.
    # ─────────────────────────────────────────────────────────────────────

    def get_crops_summary_for_prompt(self) -> str:
        """
        Text block for SQL generation prompts describing what crops
        actually exist in sub_categories. Prevents the LLM from searching
        for literal words like 'seeds' inside name columns.
        """
        if not self._loaded or not self._crops:
            return ""

        # Show up to 40 crop names — enough context, not too long
        crops_sample = sorted(self._crops)[:40]
        crops_text = ", ".join(crops_sample)
        more = f" (+ {len(self._crops) - 40} more)" if len(self._crops) > 40 else ""

        # Sample seed variety names to show what's really stored
        seed_sample = sorted(self._seed_varieties)[:10]
        seed_text = ", ".join(f'"{s}"' for s in seed_sample) if seed_sample else "(none loaded)"

        return (
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"LIVE DATABASE CONTENT (loaded at startup, refreshed every 30 min)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"\nAVAILABLE CROPS in sub_categories.name ({len(self._crops)} total):\n"
            f"  {crops_text}{more}\n"
            f"\nSEEDS TABLE — actual content:\n"
            f"  seeds.name contains VARIETY NAMES like: {seed_text}\n"
            f"  seeds.name does NOT contain the literal words 'seed' or 'seeds' or 'બીજ'.\n"
            f"  To find seeds for a specific crop, JOIN sub_categories ON\n"
            f"  seeds.subcategory_id = sub_categories.id and filter by\n"
            f"  sub_categories.name LIKE '%cropname%'.\n"
            f"  To browse all seeds, just SELECT all rows from seeds (no WHERE on name).\n"
        )

    def get_kshop_summary_for_prompt(self) -> str:
        """
        Text block for SQL generation prompts describing what K-Shop
        actually sells. Helps LLM distinguish K-Shop equipment from raw crops.
        """
        if not self._loaded or not self._kshop_categories:
            return ""

        cats_sample = sorted(self._kshop_categories)[:30]
        cats_text = ", ".join(cats_sample)
        more = f" (+ {len(self._kshop_categories) - 30} more)" if len(self._kshop_categories) > 30 else ""

        # Show a few example product names so LLM sees real format
        prod_sample = sorted(self._kshop_products)[:10]
        prod_text = ", ".join(f'"{p}"' for p in prod_sample) if prod_sample else "(none loaded)"

        return (
            f"\nK-SHOP CATEGORIES in kshop_categories.name ({len(self._kshop_categories)} total):\n"
            f"  {cats_text}{more}\n"
            f"\nK-SHOP PRODUCT NAME EXAMPLES from kshop_products.name:\n"
            f"  {prod_text}\n"
            f"  K-Shop sells farm EQUIPMENT (tractors, sprayers, weeders, pumps,\n"
            f"  motors) and supplies — NOT raw agricultural crops.\n"
            f"  For raw crops (wheat, cotton, onion), use the products/sub_categories tables.\n"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Loading and refreshing
    # ─────────────────────────────────────────────────────────────────────

    async def load(self) -> None:
        """Load all cached data from the database. Called at startup."""
        if not db_manager.is_ready:
            logger.warning("⚠️ KnowledgeCache: db_manager not ready, skipping load")
            return

        logger.info("🧠 KnowledgeCache: loading from database...")

        try:
            # Load crops (sub_categories.name)
            crops_rows = await db_manager.execute_query(
                "SELECT name FROM sub_categories "
                "WHERE deleted_at IS NULL AND status = 1"
            )
            self._crops = {
                row["name"].lower().strip()
                for row in crops_rows
                if row.get("name")
            }

            # Load K-Shop categories
            cat_rows = await db_manager.execute_query(
                "SELECT name FROM kshop_categories "
                "WHERE deleted_at IS NULL AND status = 1"
            )
            self._kshop_categories = {
                row["name"].lower().strip()
                for row in cat_rows
                if row.get("name")
            }

            # Load K-Shop product names (tokenized into individual words for matching)
            prod_rows = await db_manager.execute_query(
                "SELECT DISTINCT name FROM kshop_products "
                "WHERE deleted_at IS NULL AND status = 1 "
                "LIMIT 2000"
            )
            self._kshop_products = set()
            for row in prod_rows:
                name = (row.get("name") or "").lower().strip()
                if name:
                    # Store the full name AND individual words for substring matching
                    self._kshop_products.add(name)
                    for word in name.split():
                        if len(word) >= 3:  # skip very short words
                            self._kshop_products.add(word)

            # Load seed varieties
            seed_rows = await db_manager.execute_query(
                "SELECT DISTINCT name FROM seeds "
                "WHERE deleted_at IS NULL AND status = 1 "
                "LIMIT 1000"
            )
            self._seed_varieties = {
                row["name"].lower().strip()
                for row in seed_rows
                if row.get("name")
            }

            self._loaded = True
            logger.info(f"✅ KnowledgeCache loaded | {self.get_summary()}")

        except Exception as e:
            logger.error_with_context(e, {"action": "knowledge_cache_load"})
            # Don't crash — just log. The bot will fall back to LLM-only routing.

    async def _refresh_loop(self) -> None:
        """Background task that refreshes the cache every 30 minutes."""
        while True:
            try:
                await asyncio.sleep(_REFRESH_INTERVAL)
                logger.info("🔄 KnowledgeCache: scheduled refresh starting...")
                await self.load()
            except asyncio.CancelledError:
                logger.info("KnowledgeCache refresh loop cancelled")
                break
            except Exception as e:
                logger.error_with_context(e, {"action": "knowledge_cache_refresh"})
                # Sleep a bit before retrying to avoid tight error loops
                await asyncio.sleep(60)

    def start_background_refresh(self) -> None:
        """Start the background refresh task. Called once at startup."""
        if self._refresh_task is None:
            try:
                loop = asyncio.get_event_loop()
                self._refresh_task = loop.create_task(self._refresh_loop())
                logger.info("✅ KnowledgeCache background refresh started (every 30 min)")
            except Exception as e:
                logger.warning(f"Could not start refresh task: {e}")

    def stop_background_refresh(self) -> None:
        """Stop the background refresh task. Called at shutdown."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            self._refresh_task = None


# Singleton instance
knowledge_cache = KnowledgeCache()


def get_knowledge_cache() -> KnowledgeCache:
    return knowledge_cache