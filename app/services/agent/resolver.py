"""
Stage 2 — Entity Resolver (code only, NO LLM).

Takes the verbatim entities Stage 1 extracted and resolves each to the exact
Gujarati value(s) + row id(s) the database actually stores, using the v2
entity catalog (data/entity_catalog_v2.json + manual overrides).

LOCKED DESIGN (agreed in design review):
  • Match by NAME; resolve to row IDs. Phase 3 filters SQL by id =/IN(ids)
    when resolution succeeds, and falls back to canonical-name LIKE only when
    it doesn't. No LIMIT 1 anywhere — multi-matches all flow through.
  • Per-entity strategy order:
        1. EXACT   — normalized index lookup (highest precision, single hit)
        2. CONTAIN — whole-word containment scan over canonicals; a partial /
                     generic word ("તલ" → તલ કાળા, તલ સફેદ) returns ALL matches
                     (the IN(ids) case). No LIMIT 1.
        3. FUZZY   — RapidFuzz WRatio for typos / dialect (મૂંગફળી → મગફળી)
        4. MISS    — nothing cleared the bar.
  • Two-threshold fuzzy UX:
        score ≥ FUZZ_HIGH  → silent accept
        FUZZ_MID ≤ score   → accept but DISCLOSE ("showing results for X")
        score < FUZZ_MID   → MISS + did-you-mean candidates
    (exact + containment are always high-confidence / silent.)
  • Crop name+variety composed in CODE: "શીંગ" + "કાદરી" → try "શીંગ કાદરી"
    combined first (exact), else resolve the name and carry the variety.
  • Location level: if the user named the level (taluka/city/yard/state) search
    that table only; otherwise search city→taluka→yard→state and the matched
    table BECOMES the level (the LLM is never trusted to guess the level).
  • Anchor for crops comes from the catalog (which table matched = category vs
    sub_category). A live DB UNION query is the fallback ONLY when the catalog
    misses (a row added since the last catalog build).
  • Every miss is logged to data/resolution_misses.jsonl for weekly review →
    fold into entity_catalog_manual.json.
  • Catalog hot-reloads on file mtime change (cron rebuild needs no restart).
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from app.core.logger import get_logger

logger = get_logger("resolver")

# ── Paths ────────────────────────────────────────────────────────────────────
# resolver.py lives at app/services/agent/ → repo root is parents[3].
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_CATALOG_V2_PATH = _DATA_DIR / "entity_catalog_v2.json"
_MANUAL_PATH = _DATA_DIR / "entity_catalog_manual.json"
_MISS_LOG_PATH = _DATA_DIR / "resolution_misses.jsonl"

# ── Thresholds (tunable; validated against the audit entity set) ─────────────
FUZZ_HIGH = 90.0   # ≥ → silent accept
FUZZ_MID  = 78.0   # ≥ → accept but disclose; < → miss + did-you-mean
_DYM_CANDIDATES = 3   # how many did-you-mean suggestions to surface

# ── Catalog table groups per domain ──────────────────────────────────────────
# Which catalog tables back each entity kind. Equipment depends on the intent
# (new = K-Shop, used = Buy/Sell). Locations are searched level-first.
_CROP_SUBJECT_TABLES   = ["sub_categories"]
_CROP_CATEGORY_TABLES  = ["categories"]
_LOCATION_BY_LEVEL = {
    "state":  ["states"],
    "city":   ["cities"],
    "taluka": ["talukas"],
    "yard":   ["yards"],
}
_LOCATION_DEFAULT_ORDER = ["cities", "talukas", "yards", "states"]
_NEWS_TYPE_TABLES = ["news_types"]


def norm_key(s: str) -> str:
    """NFC + collapse whitespace + lowercase — MUST match build_entity_catalog."""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


# ── Result models ────────────────────────────────────────────────────────────

@dataclass
class ResolvedEntity:
    surface: str                       # verbatim user text
    canonical: str                     # DB-exact value (or surface on miss)
    canonicals: List[str]              # all matched canonicals (containment → many)
    ids: List[int]                     # all matched row ids (empty on miss)
    table: str                         # catalog table that matched ("" on miss)
    match_type: str                    # exact | contain | fuzzy_high | fuzzy_mid | miss | live
    score: float                       # 100 for exact/contain
    candidates: List[str] = field(default_factory=list)  # did-you-mean (miss only)

    @property
    def resolved(self) -> bool:
        return self.match_type != "miss"

    @property
    def needs_disclosure(self) -> bool:
        return self.match_type == "fuzzy_mid"


@dataclass
class ResolvedFrame:
    crops:       List[ResolvedEntity] = field(default_factory=list)
    categories:  List[ResolvedEntity] = field(default_factory=list)
    locations:   List[ResolvedEntity] = field(default_factory=list)
    equipment:   List[ResolvedEntity] = field(default_factory=list)
    animals:     List[ResolvedEntity] = field(default_factory=list)
    news_types:  List[ResolvedEntity] = field(default_factory=list)

    def all_entities(self) -> List[ResolvedEntity]:
        return (self.crops + self.categories + self.locations
                + self.equipment + self.animals + self.news_types)

    def subject_entities(self) -> List[ResolvedEntity]:
        """Entities that name WHAT the user wants (not the location scope).
        These are the ones whose miss means 'item not in DB'."""
        return self.crops + self.categories + self.equipment + self.animals + self.news_types

    @property
    def disclosures(self) -> List[str]:
        out = []
        for e in self.all_entities():
            if e.needs_disclosure and e.canonical:
                out.append(e.canonical)
        return out

    def hard_miss_subjects(self) -> List[ResolvedEntity]:
        """Subject entities that did not resolve at all (→ did-you-mean / not found)."""
        return [e for e in self.subject_entities() if not e.resolved]


# ── Resolver ─────────────────────────────────────────────────────────────────

class EntityResolver:
    """Loads the v2 catalog (+ manual overrides) once, hot-reloads on change,
    and resolves NLU-frame entities to canonical Gujarati values + row ids."""

    def __init__(self):
        # Per table: index {normkey: canonical}, and entries {canonical: [ids]}.
        self._index: Dict[str, Dict[str, str]] = {}
        self._ids: Dict[str, Dict[str, List[int]]] = {}
        # Pre-split canonical word sets for the containment scan.
        self._canon_words: Dict[str, List[Tuple[str, set]]] = {}
        self._mtimes: Tuple[float, float] = (0.0, 0.0)
        self._load()

    # ── load / hot-reload ────────────────────────────────────────────────────
    def _current_mtimes(self) -> Tuple[float, float]:
        def mt(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except Exception:
                return 0.0
        return (mt(_CATALOG_V2_PATH), mt(_MANUAL_PATH))

    def _maybe_reload(self) -> None:
        if self._current_mtimes() != self._mtimes:
            logger.info("🔁 entity catalog changed on disk — hot-reloading")
            self._load()

    def _load(self) -> None:
        index: Dict[str, Dict[str, str]] = {}
        ids: Dict[str, Dict[str, List[int]]] = {}

        # v2 catalog: {table: {entries: {canonical: {ids:[...]}}, index: {key: canonical}}}
        v2 = self._read_json(_CATALOG_V2_PATH)
        for table, payload in (v2 or {}).items():
            if not isinstance(payload, dict):
                continue
            idx = {norm_key(k): v for k, v in payload.get("index", {}).items()
                   if isinstance(k, str) and isinstance(v, str)}
            ent = {}
            for canon, meta in payload.get("entries", {}).items():
                if isinstance(canon, str) and isinstance(meta, dict):
                    ent[canon] = [int(i) for i in meta.get("ids", []) if isinstance(i, int)]
            index[table] = idx
            ids[table] = ent

        # Manual overrides (flat v1 shape) — manual always wins; no ids known.
        manual = self._read_json(_MANUAL_PATH)
        for table, flat in (manual or {}).items():
            if not isinstance(flat, dict):
                continue
            index.setdefault(table, {})
            ids.setdefault(table, {})
            for k, v in flat.items():
                if isinstance(k, str) and isinstance(v, str):
                    index[table][norm_key(k)] = v
                    ids[table].setdefault(v, [])  # canonical exists, ids filled by v2 if present

        # Pre-compute word sets per canonical for the containment scan.
        canon_words: Dict[str, List[Tuple[str, set]]] = {}
        for table, ent in ids.items():
            canon_words[table] = [
                (canon, set(norm_key(canon).split())) for canon in ent.keys()
            ]

        self._index, self._ids, self._canon_words = index, ids, canon_words
        self._mtimes = self._current_mtimes()
        total = sum(len(t) for t in index.values())
        logger.info(f"📚 Resolver loaded v2 catalog | tables={len(index)} index_keys={total}")

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception as e:
            logger.error_with_context(e, {"action": "resolver_load", "path": str(path)})
            return {}

    # ── single-value resolution ──────────────────────────────────────────────
    def resolve(self, value: str, tables: List[str], gather: bool = False) -> ResolvedEntity:
        """Resolve one surface string against the given catalog tables in order.

        Strategy: EXACT / CONTAINMENT → FUZZY(two-threshold) → MISS.

        gather=False (locations, categories, animals, equipment): EXACT wins
          first (precise); CONTAINMENT only if no exact hit anywhere.
        gather=True (crop subjects): UNION exact + whole-word containment so a
          family word ("તલ" / "મગફળી") pulls every variety ("તલ કાળા", "મગફળી 24")
          — the recall a farmer asking "<crop> ભાવ" expects. No LIMIT 1.
        """
        surface = (value or "").strip()
        if not surface:
            return ResolvedEntity(surface, "", [], [], "", "miss", 0.0)
        key = norm_key(surface)
        key_words = set(key.split())

        for table in tables:
            exact = self._index.get(table, {}).get(key)
            contained = [
                canon for canon, words in self._canon_words.get(table, [])
                if key_words and key_words.issubset(words) and canon != exact
            ]
            if gather:
                # Union exact + containment family.
                canonicals: List[str] = ([exact] if exact else []) + contained
                if canonicals:
                    all_ids: List[int] = []
                    for c in canonicals:
                        all_ids.extend(self._ids.get(table, {}).get(c, []))
                    primary = exact or min(canonicals, key=len)
                    return ResolvedEntity(
                        surface, primary, sorted(set(canonicals), key=len),
                        sorted(set(all_ids)), table,
                        "exact" if exact else "contain", 100.0,
                    )
            else:
                # Precise: exact wins immediately.
                if exact:
                    return ResolvedEntity(
                        surface, exact, [exact],
                        self._ids.get(table, {}).get(exact, []),
                        table, "exact", 100.0,
                    )
                if contained:
                    all_ids = []
                    for c in contained:
                        all_ids.extend(self._ids.get(table, {}).get(c, []))
                    primary = min(contained, key=len)
                    return ResolvedEntity(
                        surface, primary, sorted(set(contained), key=len),
                        sorted(set(all_ids)), table, "contain", 100.0,
                    )

        # FUZZY — typos / dialect, two-threshold.
        best: Optional[Tuple[str, float, str]] = None  # (canonical, score, table)
        for table in tables:
            idx = self._index.get(table, {})
            if not idx:
                continue
            m = process.extractOne(key, idx.keys(), scorer=fuzz.WRatio)
            if m:
                matched_key, score, _ = m
                if best is None or score > best[1]:
                    best = (idx[matched_key], score, table)
        if best:
            canon, score, table = best
            if score >= FUZZ_HIGH:
                mt = "fuzzy_high"
            elif score >= FUZZ_MID:
                mt = "fuzzy_mid"
            else:
                mt = None
            if mt:
                return ResolvedEntity(
                    surface, canon, [canon],
                    self._ids.get(table, {}).get(canon, []),
                    table, mt, float(score),
                )

        # 4. MISS — gather did-you-mean candidates from the best table.
        candidates = self._did_you_mean(key, tables)
        return ResolvedEntity(surface, surface, [], [], "", "miss", 0.0, candidates)

    def _did_you_mean(self, key: str, tables: List[str]) -> List[str]:
        pool: List[Tuple[str, float]] = []
        for table in tables:
            idx = self._index.get(table, {})
            for cand in process.extract(key, idx.keys(), scorer=fuzz.WRatio,
                                        limit=_DYM_CANDIDATES):
                matched_key, score, _ = cand
                pool.append((idx[matched_key], score))
        pool.sort(key=lambda x: x[1], reverse=True)
        seen, out = set(), []
        for canon, _ in pool:
            if canon not in seen:
                seen.add(canon)
                out.append(canon)
            if len(out) >= _DYM_CANDIDATES:
                break
        return out

    # ── frame resolution ─────────────────────────────────────────────────────
    def resolve_frame(self, frame) -> ResolvedFrame:
        """Resolve every entity in an NLUFrame. Catalog-only (sync)."""
        self._maybe_reload()
        rf = ResolvedFrame()
        intent = getattr(frame, "intent", "")

        # Crops: category word (if any) → categories; name(+variety) → sub_categories.
        for crop in getattr(frame, "crops", []):
            if crop.category:
                rf.categories.append(self.resolve(crop.category, _CROP_CATEGORY_TABLES))
            # Only resolve a sub_category crop when an actual crop NAME exists.
            # A category-only block (name=None) is fully handled above — do NOT
            # also probe sub_categories with the category word (phantom miss).
            if crop.name:
                combined = crop.surface()  # "શીંગ કાદરી" or "કપાસ"
                res = self.resolve(combined, _CROP_SUBJECT_TABLES, gather=True)
                if not res.resolved and crop.variety:
                    # Combined missed — resolve the base name, keep variety for
                    # Phase 3 AND-ed filtering.
                    res = self.resolve(crop.name, _CROP_SUBJECT_TABLES, gather=True)
                rf.crops.append(res)

        # Equipment: table set depends on new (K-Shop) vs used (Buy/Sell).
        eq_tables = (["kshop_categories"] if intent == "equipment_kshop"
                     else ["buy_sell_categories"] if intent == "equipment_used"
                     else ["kshop_categories", "buy_sell_categories"])
        for eq in getattr(frame, "equipment", []):
            if eq.name:
                rf.equipment.append(self.resolve(eq.name, eq_tables))

        # Animals → buy_sell_categories.
        for animal in getattr(frame, "animals", []):
            if animal:
                rf.animals.append(self.resolve(animal, ["buy_sell_categories"]))

        # Locations: level-first, else default order; matched table = the level.
        for loc in getattr(frame, "locations", []):
            tables = _LOCATION_BY_LEVEL.get(loc.level or "", _LOCATION_DEFAULT_ORDER)
            rf.locations.append(self.resolve(loc.name, tables))

        # News type (the category word). Topics stay free-text (Phase 3 LIKE).
        if getattr(frame, "news", None) and frame.news.type:
            rf.news_types.append(self.resolve(frame.news.type, _NEWS_TYPE_TABLES))

        self._log_misses(intent, rf)
        return rf

    # ── live DB fallback (only on catalog miss) ──────────────────────────────
    async def resolve_live(self, value: str, db_tables: List[str]) -> Optional[ResolvedEntity]:
        """Live UNION query against the DB for a value the catalog missed
        (e.g. a row added since the last catalog build). Match-quality ordered,
        NO LIMIT 1 — returns every row sharing the best-matching name.

        db_tables are REAL table names (sub_categories, cities, ...). Returns
        None when the DB has nothing either.
        """
        surface = (value or "").strip()
        if not surface or not db_tables:
            return None
        try:
            from app.core.database import get_db_manager
            db = await get_db_manager()
        except Exception as e:
            logger.warning(f"resolve_live: DB unavailable: {e}")
            return None

        like = f"%{surface}%"
        union = " UNION ALL ".join(
            f"SELECT '{t}' AS src, id, name FROM {t} "
            f"WHERE name LIKE %s AND deleted_at IS NULL"
            for t in db_tables
        )
        params = tuple(like for _ in db_tables)
        try:
            rows = await db.execute_query(union, params)
        except Exception as e:
            logger.warning(f"resolve_live query failed: {e}")
            return None
        if not rows:
            return None

        # Group by source table; pick the table whose best name is the closest
        # match, then return ALL ids from that table's matching names.
        best_table, best_score = "", -1.0
        for r in rows:
            score = fuzz.WRatio(norm_key(surface), norm_key(r["name"]))
            if score > best_score:
                best_score, best_table = score, r["src"]
        matched = [r for r in rows if r["src"] == best_table]
        canonicals = sorted({r["name"] for r in matched}, key=len)
        ids = sorted({int(r["id"]) for r in matched})
        logger.info(f"🛟 resolve_live HIT | {surface!r} → {best_table} "
                    f"canonicals={canonicals[:3]} ids={ids[:5]}")
        return ResolvedEntity(surface, canonicals[0], canonicals, ids,
                              best_table, "live", float(best_score))

    # ── miss logging ─────────────────────────────────────────────────────────
    def _log_misses(self, intent: str, rf: ResolvedFrame) -> None:
        misses = rf.hard_miss_subjects()
        if not misses:
            return
        try:
            with _MISS_LOG_PATH.open("a", encoding="utf-8") as f:
                for e in misses:
                    f.write(json.dumps({
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "intent": intent,
                        "surface": e.surface,
                        "candidates": e.candidates,
                    }, ensure_ascii=False) + "\n")
        except Exception as ex:
            logger.warning(f"could not write miss log: {ex}")


_instance: Optional[EntityResolver] = None


def get_resolver() -> EntityResolver:
    global _instance
    if _instance is None:
        _instance = EntityResolver()
    return _instance
