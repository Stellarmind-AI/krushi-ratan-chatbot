"""
Entity Normalizer — maps a user keyword to the canonical Gujarati value
the database actually stores, using the catalog built by
scripts/build_entity_catalog.py.

Why this exists
---------------
The DB stores names in Gujarati only. Users ask in English, Hindi,
Romanized Gujarati, or Gujarati script. Letting the SQL-generation LLM
invent multi-language LIKE clauses caused hallucination — it would emit
English/Hindi variants and miss the Gujarati variant that is the only
one that actually matches DB rows.

This module resolves the keyword to its canonical Gujarati form BEFORE
the SQL-generation LLM runs, so the prompt can instruct the LLM to use
exactly ONE Gujarati value (no variants).

Usage
-----
    from app.services.entity_normalizer import get_entity_normalizer

    normalizer = get_entity_normalizer()
    canonical  = normalizer.normalize("crop_price", "cotton")
    # → "કપાસ" (or None if no match)

Behaviour
---------
1. Look up the F1 intent in INTENT_TO_TABLES → ordered list of catalog tables.
2. Pass 1 — try exact match (case-folded) across every listed table in
   priority order; return the canonical Gujarati on the first hit.
3. Pass 2 — fall back to RapidFuzz WRatio fuzzy match (threshold
   _FUZZ_THRESHOLD) across the same tables in the same order.
4. Return the canonical Gujarati string, or None when nothing resolves.

Some intents have multiple tables in their priority list so that a single
intent can carry either a subject keyword OR a location keyword — e.g.
crop_price queries may use "kapas" (→ sub_categories) or "rajkot"
(→ cities). Tables are tried in priority order: subject first, locations
after, so an unambiguous subject hit beats any location collision.

If the catalog file is missing the module loads an empty index — every
normalize() call returns None and the SQL prompt falls back to its
"translate keyword to Gujarati" path. No exceptions are raised at import.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from rapidfuzz import fuzz, process

from app.core.logger import get_logger

logger = get_logger("entity_normalizer")


# ── F1 intent → catalog source tables (in priority order) ───────────────────
#
# Each intent maps to a LIST of tables whose name columns are searched
# in priority order. The first table is the "natural" subject table for
# the intent (the crop name for crop_price, the news type for local_news);
# the rest are location/context tables that can carry a keyword for that
# intent (e.g. crop_price queries are often location-scoped — "kapas bhav
# rajkot" → keyword could be either કપાસ or રાજકોટ).
#
# Lookup rules:
#   1. Try EXACT match across every listed table, in order — first hit wins.
#   2. Only if NO exact match anywhere, try fuzzy match across the same list
#      in the same priority order.
#   This guarantees a high-confidence hit always beats a low-confidence one,
#   even if the low-confidence hit is in the highest-priority table.
#
# Notes:
#   • crop_price / seed_info → sub_categories holds crop names (કપાસ, ઘઉં …).
#     seeds.name holds variety strings users rarely type by name.
#   • crop_price gets cities/talukas/yards/states fallback because location
#     queries are common ("kapas bhav rajkot", "ગોંડલ માં ભાવ").
#   • local_news gets cities/talukas/states fallback (news has no yard FK).
#   • equipment_kshop / kshop_product share kshop_categories. buy_sell shares
#     buy_sell_categories. Those product tables have no location FKs, so no
#     location fallback (would only invite false matches).
#   • video_search is intentionally absent — video titles are free text;
#     normalize() will return None and the SQL prompt will translate.
INTENT_TO_TABLES: Dict[str, List[str]] = {
    "crop_price":       ["sub_categories", "cities", "talukas", "yards", "states"],
    "seed_info":        ["sub_categories"],
    "kshop_product":    ["kshop_categories"],
    "buy_sell_product": ["buy_sell_categories"],
    "equipment_kshop":  ["kshop_categories"],
    "equipment_used":   ["buy_sell_categories"],
    "local_news":       ["news_types", "cities", "talukas", "states"],
}

# RapidFuzz WRatio cutoff. Empirically tight enough to reject unrelated
# words but loose enough to absorb common typos and inflections.
_FUZZ_THRESHOLD = 85

# Catalog path: <repo_root>/data/entity_catalog.json
_CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "entity_catalog.json"


class EntityNormalizer:
    """Loads the entity catalog once and resolves keywords against it."""

    def __init__(self):
        self._catalog: Dict[str, Dict[str, str]] = {}
        self._load()

    # ── load ─────────────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not _CATALOG_PATH.exists():
            logger.warning(
                f"entity_catalog.json not found at {_CATALOG_PATH} — "
                f"normalizer will return None for every call. "
                f"Run scripts/build_entity_catalog.py to populate it."
            )
            return
        try:
            with _CATALOG_PATH.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.error_with_context(
                e, {"action": "load_entity_catalog", "path": str(_CATALOG_PATH)}
            )
            return

        if not isinstance(raw, dict):
            logger.warning(
                f"entity_catalog.json root is not an object — ignoring"
            )
            return

        # Defensive normalisation: lowercase non-Gujarati keys so exact
        # lookup is case-insensitive without re-lowering every call.
        cleaned: Dict[str, Dict[str, str]] = {}
        for table, sub in raw.items():
            if not isinstance(sub, dict):
                continue
            new_sub: Dict[str, str] = {}
            for k, v in sub.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    continue
                # Keep original key form, AND add a lowercase variant —
                # cheap, and protects against catalog data with mixed case.
                new_sub.setdefault(k, v)
                kl = k.lower()
                if kl != k:
                    new_sub.setdefault(kl, v)
            cleaned[table] = new_sub
        self._catalog = cleaned

        total = sum(len(t) for t in self._catalog.values())
        logger.info(
            f"📚 EntityNormalizer loaded: tables={len(self._catalog)} entries={total}"
        )

    # ── public ───────────────────────────────────────────────────────────────
    def normalize(self, intent: str, keyword: str) -> Optional[str]:
        """
        Resolve (intent, keyword) → canonical Gujarati value, or None.

        Walks INTENT_TO_TABLES[intent] in order:
          Pass 1 — exact match across every table; first hit wins.
          Pass 2 — fuzzy match across every table; first hit wins.
        Exact-first ordering means an exact hit in a low-priority table
        beats a fuzzy hit in the top-priority table — high confidence
        always wins over low confidence.

        Returns None when:
          • intent not in INTENT_TO_TABLES (e.g. video_search)
          • all listed tables are empty / missing from the catalog
          • keyword is empty
          • neither exact nor fuzzy match in any table clears the threshold
        """
        if not intent or not keyword:
            return None

        tables = INTENT_TO_TABLES.get(intent)
        if not tables:
            return None

        kw = keyword.strip()
        if not kw:
            return None
        kw_low = kw.lower()

        # Pass 1 — exact match across every table in priority order.
        for table in tables:
            sub = self._catalog.get(table)
            if not sub:
                continue
            if kw in sub:
                return sub[kw]
            if kw_low in sub:
                return sub[kw_low]

        # Pass 2 — fuzzy match across every table in priority order.
        for table in tables:
            sub = self._catalog.get(table)
            if not sub:
                continue
            match = process.extractOne(
                kw_low,
                sub.keys(),
                scorer=fuzz.WRatio,
                score_cutoff=_FUZZ_THRESHOLD,
            )
            if match:
                matched_key, score, _ = match
                canonical = sub[matched_key]
                logger.debug(
                    f"fuzzy normalize intent={intent} table={table} keyword={kw!r} "
                    f"→ {matched_key!r} (score={score:.1f}) → {canonical!r}"
                )
                return canonical

        return None


# ── singleton ───────────────────────────────────────────────────────────────
_instance: Optional[EntityNormalizer] = None


def get_entity_normalizer() -> EntityNormalizer:
    global _instance
    if _instance is None:
        _instance = EntityNormalizer()
    return _instance
