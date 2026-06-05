"""
Query Cache — Skips LLM SQL generation for repeated questions.

STORAGE: app/cache/query_cache.json
  Persists across server restarts. No Redis needed.
  Each new entry is saved to disk immediately.

WHAT IS CACHED:
  question → SQL query string (NOT the database result rows)
  Rows change daily (prices update). SQL stays the same.
  So the cache never returns stale data — it only avoids
  regenerating the same SQL query over and over.

HOW KEYS WORK:
  "Balwan Power Weeder kharido"  →  strip intent words, sort  →  "balwan weeder"
  "balwan power weeder joiyu"    →  strip intent words, sort  →  "balwan weeder"
  Both normalize to same key → same SQL → one cache entry for both.

HIT RATE EXPECTATION:
  Farmers ask the same things repeatedly. After 1-2 days of usage,
  60-70% of queries will hit the cache and skip 2 LLM calls each.
"""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from app.core.logger import get_logger, Timer

logger = get_logger("query_cache")

# ─────────────────────────────────────────────────────────────────────────────
# Storage path
# app/cache/ = runtime-generated files, NOT committed to git
# app/schemas/ = source files you edit and commit
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_DIR  = Path("app/cache")
_CACHE_FILE = _CACHE_DIR / "query_cache.json"

# ─────────────────────────────────────────────────────────────────────────────
# Filler/intent word detection — REGEX-FIRST design.
#
# Why not a flat set: a flat set misses every inflection and typo it wasn't
# hand-entered for.  "any" vs "some", "product" vs "products", "waht" vs
# "what" — each new variant required a manual edit, and a single miss (like
# "any" in "how i sell any product") caused the exploratory-query detector to
# treat filler as a search keyword.
#
# New design — two layers of defense:
#   1. REGEX LAYER — semantic root groups × inflection-suffix pattern.
#      One root "sell" auto-matches sell/sells/selling/seller/sellers.
#      One root "product" auto-matches product/products.
#   2. FUZZY LAYER — difflib close-match against the same roots (cutoff 0.85).
#      Catches typos: prodct→product, sellr→seller, waht→what.
#
# Both layers feed a single `is_filler_token()` predicate so the orchestrator
# and the cache normalizer share one source of truth.
#
# To extend: add roots to the relevant semantic group below — that's it.
# No need to enumerate every inflection or typo manually.
# ─────────────────────────────────────────────────────────────────────────────

# Semantic root groups — English/romanized tokens only.
# Gujarati-script fillers use a separate set (Gujarati morphology differs).
#
# Each group declares:
#   inflect: True  → suffix pattern is applied (verbs, common nouns)
#   inflect: False → exact match only (grammatical primitives, Indic tokens)
#
# Why inflect is a per-group choice:
#   Applying English suffixes to grammatical primitives (prepositions, articles,
#   pronouns, conjunctions) creates catastrophic false-positives against real
#   content words.  Example: if "on" (PREP) + "ion" (suffix) both match, the
#   pattern \bon(?:ion)?\b also matches "ONION" — a real crop.  Similarly:
#     - "an" + "ion" → "anion"
#     - "to" + "es"  → "toes"
#     - "on" + "ly"  → "only" (benign here, but illustrates the class)
#   Grammatical primitives are atomic — they don't inflect — so their regex
#   must be exact match.  Suffix inflection is restricted to VERBS and NOUNS.
_ROOTS_BY_GROUP: Dict[str, Dict[str, object]] = {
    "QUESTION": {"inflect": False, "roots": {
        "how", "what", "where", "when", "why", "which", "who", "whose", "whom",
    }},
    "INTENT_VERB": {"inflect": True, "roots": {
        "sell", "buy", "purchase", "get", "find", "show", "tell", "give",
        "want", "need", "like", "help", "fetch", "display", "list", "explain",
        "describe", "know", "mean", "understand", "say", "ask", "look",
    }},
    "DETERMINER": {"inflect": False, "roots": {
        # Quantifiers/determiners — never search keywords on their own.
        "any", "some", "all", "every", "each", "no", "none", "another", "other",
        "few", "many", "much", "several", "both", "either", "neither",
        "this", "that", "these", "those",
    }},
    "ARTICLE": {"inflect": False, "roots": {"a", "an", "the"}},
    "PRONOUN": {"inflect": False, "roots": {
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
    }},
    "AUX": {"inflect": False, "roots": {
        "is", "are", "was", "were", "be", "been", "being", "am",
        "have", "has", "had", "having",
        "do", "does", "did", "doing", "done",
        "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    }},
    "PREP": {"inflect": False, "roots": {
        "in", "on", "at", "of", "for", "to", "from", "with", "by",
        "about", "into", "onto", "upon", "under", "over", "between",
        "through", "during", "before", "after", "above", "below",
    }},
    "CONJ": {"inflect": False, "roots": {
        "and", "or", "but", "so", "if", "then", "than", "as",
        "because", "while", "although", "though",
    }},
    "POLITE": {"inflect": False, "roots": {"please", "pls", "kindly", "thanks", "thank"}},
    # Category/browse nouns — app domain words that are NEVER specific items.
    # (Absorbs what used to live in knowledge_cache.BROWSE_KEYWORDS.)
    "CATEGORY": {"inflect": True, "roots": {
        "product", "item", "crop", "seed", "video", "news",
        "price", "bhav", "rate", "category", "yard", "mandi",
        "order", "listing", "post", "thing", "stuff",
        "samachar", "khabar", "kshop",
    }},
    # Romanized Gujarati / Hindi intent/filler.
    # Indic morphology doesn't follow English suffix rules (maro/mari/mara are
    # gender-inflections, not suffix-inflections), so exact match only.
    "INDIC_ROMAN": {"inflect": False, "roots": {
        "mare", "maro", "mari", "mara", "mane",
        "karvu", "karvani", "karvo", "karu", "karshu", "karva",
        "che", "chhe", "hatu", "hati", "hashe",
        "joiyu", "joiye", "jovu", "joi", "juo",
        "levu", "levo", "levi", "lidhu", "laao",
        "kharidi", "kharido", "vechan", "vecho",
        "thi", "ma", "ni", "nu", "ne", "ane", "na",
        "apo", "apso", "batao", "batavo", "batavsho",
        "pan", "pani", "kem", "shu", "kevi", "rite",
        "mujhe", "mera", "mere", "meri", "hain", "hai",
        "karo", "karna", "chahiye", "dijiye",
        "ka", "ki", "ke", "ko", "se", "mein",
    }},
}

# Derive root collections from the group declarations
_INFLECTABLE_ROOTS: Set[str] = set()
_FIXED_ROOTS:       Set[str] = set()
for _spec in _ROOTS_BY_GROUP.values():
    _bucket = _INFLECTABLE_ROOTS if _spec["inflect"] else _FIXED_ROOTS
    _bucket |= _spec["roots"]  # type: ignore[operator]

# Flat union — used for fuzzy-match pool and STRIP_WORDS materialization.
_ALL_ROOTS: Set[str] = _INFLECTABLE_ROOTS | _FIXED_ROOTS

# Inflection-suffix pattern — ONLY applied to inflectable roots (verbs, nouns).
# Irregular verbs (go/went, sell/sold) aren't covered by suffix patterns;
# if an irregular becomes relevant, add it as its own root.
_INFLECTION = r"(?:s|es|ed|ing|er|ers|est|ly|ion|ions|ive|al|ies|y)?"

# Compile master filler regex as TWO alternations (critical correctness point):
#   1. Inflectable roots + suffix  — "sell", "selling", "product", "products"
#   2. Fixed roots, exact match    — "on", "an", "the", "i", "any"
#
# Longer roots first inside each alternation so e.g. "product" beats "pro".
# The fixed-group alternation has NO trailing suffix pattern, which prevents
# "on" + "ion" from ever matching "onion".
def _alt(roots: Set[str]) -> str:
    return "|".join(re.escape(r) for r in sorted(roots, key=len, reverse=True))

_FILLER_RE = re.compile(
    r"\b(?:"
    + r"(?:" + _alt(_INFLECTABLE_ROOTS) + r")" + _INFLECTION
    + r"|"
    + r"(?:" + _alt(_FIXED_ROOTS) + r")"
    + r")\b",
    re.IGNORECASE,
)

# Gujarati-script fillers — exact match (Gujarati morphology varies per root).
_GUJARATI_FILLERS: Set[str] = {
    # Question/intent
    "કેવી", "રીતે", "કેમ", "શું", "કયા", "કયું", "કોણ", "ક્યાં",
    # Pronouns/possessives
    "મારે", "મારો", "મારી", "મને", "અમારે", "અમને", "તમે", "તમારું",
    # Be/aux
    "છે", "છો", "હશે", "હતો", "હતી",
    # Modal/volition
    "જોઈએ", "જોવું", "લેવું", "લઈ", "આપો", "બતાવો", "કહો",
    # Determiner/quantifier
    "કોઈ", "કેટલા", "કેટલું", "બધા", "આ", "તે", "કોઈપણ",
    # Category nouns
    "બીજ", "ઉત્પાદ", "વસ્તુ", "પાક", "વિડિઓ", "ભાવ",
    "સમાચાર", "શ્રેણી", "યાર્ડ", "મંડી", "ઓર્ડર",
}
_GUJARATI_FILLER_RE = re.compile("|".join(re.escape(w) for w in _GUJARATI_FILLERS))

# Fuzzy-match typo tolerance.
#
# Two independent thresholds (separating them is critical for correctness):
#
#   _FUZZY_TOKEN_MIN_LEN = 4
#     Minimum length of the USER TOKEN to be eligible for fuzzy checking.
#     Short tokens (≤3 chars) have too-high collision rates under edit distance.
#
#   _FUZZY_ROOT_MIN_LEN  = 5
#     Minimum length of ROOTS admitted to the fuzzy pool.  Short roots like
#     "what", "find", "show" (4 chars) are dangerously close to real content
#     words — "wheat" is 1 edit from "what" and would be wrongly filtered.
#     The exact-match regex already covers short roots; fuzzy adds no value
#     for them but creates collisions.
#
# Cutoff 0.85 → "prodct" matches "product" (ratio 0.92) but "kapas" does NOT
# match "papas"/"paras"/etc.  With _FUZZY_ROOT_MIN_LEN=5, there's no 4-char
# root left to collide with "wheat", "bean", "toes", etc.
import difflib

_FUZZY_CUTOFF        = 0.85
_FUZZY_TOKEN_MIN_LEN = 4
_FUZZY_ROOT_MIN_LEN  = 5
_FUZZY_POOL = [r for r in _ALL_ROOTS if len(r) >= _FUZZY_ROOT_MIN_LEN]

# Category-prefix layer — catches truncated category fragments that fuzzy
# misses because difflib's character-set ratio falls below 0.85 when the
# fragment is much shorter than the root (e.g. "prod" vs "product" → 0.73).
#
# Real-world hits this closes:
#   "tell me about prod"   → "prod" is a fragment of "product"   → filler
#   "show vide"            → "vide" is a fragment of "video"     → filler
#   "give me cate list"    → "cate" is a fragment of "category"  → filler
#   "any orde"             → "orde" is a fragment of "order"     → filler
#
# Constraints to avoid false positives on real keywords:
#   1. Only INFLECTABLE CATEGORY roots are eligible — these are app-domain
#      browse words that are NEVER specific items.  Roots from other groups
#      (verbs/prepositions/etc.) are already covered by exact + suffix regex.
#   2. Roots must be ≥5 chars so a ≥4-char prefix is unambiguous.  Shorter
#      roots like "crop", "seed", "yard" are already filler via exact match.
#   3. Token must be ≥4 chars AND strictly shorter than the root.
#
# Verified non-collisions against real keywords used in the app:
#   "kapas", "motor", "weeder", "balwan", "tractor", "thresher", "wheat",
#   "ghau", "bajra", "magfali", "onion", "tomato" — none are prefixes of
#   any CATEGORY root in this set.
_CATEGORY_PREFIX_MIN_LEN  = 4
_CATEGORY_PREFIX_ROOT_MIN = 5
_CATEGORY_PREFIX_ROOTS: Set[str] = {
    r for r in _ROOTS_BY_GROUP["CATEGORY"]["roots"]   # type: ignore[index,union-attr]
    if len(r) >= _CATEGORY_PREFIX_ROOT_MIN
}


def is_filler_token(token: str) -> bool:
    """
    True if the token is a filler/intent/category word.

    Layers (cheapest first; first hit wins):
      - Regex layer: roots + common inflection (s/es/ing/ed/er/ly/...).
      - Gujarati-script tokens: exact match against the Gujarati filler set.
      - Category-prefix layer: token is a ≥4-char prefix of an inflectable
        CATEGORY root ≥5 chars (e.g. "prod" → "product"). Catches user
        truncations the fuzzy layer misses.
      - Fuzzy layer: edit-distance tolerance for tokens ≥4 chars (typo safety).

    Used by:
      - Cache-key normalization (every token filtered individually)
      - Orchestrator exploratory-query detection
    """
    if not token:
        return True
    t = token.lower().strip()
    if len(t) <= 1:
        return True
    # Regex layer — handles English + romanized Indic inflections
    if _FILLER_RE.fullmatch(t):
        return True
    # Gujarati script exact-match
    if t in _GUJARATI_FILLERS:
        return True
    # Category-prefix layer — handles truncated browse words
    if len(t) >= _CATEGORY_PREFIX_MIN_LEN:
        for root in _CATEGORY_PREFIX_ROOTS:
            if len(t) < len(root) and root.startswith(t):
                return True
    # Fuzzy layer — typo tolerance (regex didn't match; is this a typo?)
    if len(t) >= _FUZZY_TOKEN_MIN_LEN:
        if difflib.get_close_matches(t, _FUZZY_POOL, n=1, cutoff=_FUZZY_CUTOFF):
            return True
    return False


def strip_fillers(query: str) -> List[str]:
    """
    Return the list of tokens surviving filler removal — potentially searchable
    keywords.  Empty list means the query contains no search targets
    (it's exploratory).

    Pipeline:
      1. Run regex substitutions (English + Gujarati)
      2. Tokenize, drop 1-char fragments
      3. Fuzzy pass to catch typos the regex missed
    """
    if not query:
        return []
    q = query.lower()
    q = _FILLER_RE.sub(" ", q)
    q = _GUJARATI_FILLER_RE.sub(" ", q)
    # Remove punctuation but keep Devanagari (ऀ-ॿ) and Gujarati (઀-૿)
    q = re.sub(r"[^\w\sऀ-ॿ઀-૿]", " ", q)
    tokens = [t for t in q.split() if len(t) > 1]
    # Fuzzy second pass — regex didn't match these, but they may be typos
    return [t for t in tokens if not is_filler_token(t)]


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat: STRIP_WORDS is still exported as a flat set.
#
# Built by expanding every root with its common inflections.  The cache-key
# normalizer does `if w not in _STRIP_WORDS` (exact-match semantics) so we
# materialize the regex's language as an explicit set here.
# ─────────────────────────────────────────────────────────────────────────────
def _expand_root(root: str) -> Set[str]:
    """Generate the set of inflected forms for a root (conservative)."""
    forms = {root}
    # Add common English inflections — keep it modest to avoid bogus words
    for sfx in ("s", "es", "ed", "ing", "er", "ers", "ly", "ies"):
        forms.add(root + sfx)
    # -y → -ies (categories→category is handled the other direction elsewhere)
    if root.endswith("y") and len(root) > 2:
        forms.add(root[:-1] + "ies")
    return forms

STRIP_WORDS: Set[str] = set()
for _r in _ALL_ROOTS:
    STRIP_WORDS |= _expand_root(_r)
# Also keep Gujarati-script fillers in the flat set for cache normalization
STRIP_WORDS |= _GUJARATI_FILLERS

# Internal alias kept so existing code in this file still works
_STRIP_WORDS = STRIP_WORDS


@dataclass
class CacheEntry:
    """One cached query mapping: normalized question → SQL queries."""
    queries:        List[Dict[str, str]]   # [{table_name, sql}, ...]
    selected_tools: List[str]              # ["query_kshop_products", ...]
    normalized_key: str                    # human-readable key for debugging
    created_at:     float = field(default_factory=time.time)
    hits:           int   = 0
    last_hit:       float = field(default_factory=time.time)


class QueryCache:
    """
    Persistent query cache backed by app/cache/query_cache.json.

    - In-memory dict for fast O(1) lookups.
    - JSON file for persistence across server restarts.
    - TTL eviction for stale entries.
    - LRU eviction when max_size is reached.
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 500):
        """
        Args:
            ttl_seconds: How long a cache entry lives. Default = 1 hour.
                         SQL rarely changes so this is conservative.
            max_size:    Max number of entries. Keeps the JSON file small.
                         500 entries ≈ ~200 KB on disk.
        """
        self.ttl      = ttl_seconds
        self.max_size = max_size
        self._cache:  Dict[str, CacheEntry] = {}
        self._hits    = 0
        self._misses  = 0

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    # ─────────────────────────────────────────────────────────────────────────
    # Disk I/O
    # ─────────────────────────────────────────────────────────────────────────

    def _load_from_disk(self):
        """Load persisted cache entries from JSON file at startup."""
        if not _CACHE_FILE.exists():
            logger.info(f"📂 No cache file at {_CACHE_FILE} — starting with empty cache")
            return

        try:
            with Timer() as t:
                with open(_CACHE_FILE, encoding="utf-8") as f:
                    raw: dict = json.load(f)

            now     = time.time()
            loaded  = 0
            expired = 0

            for h, d in raw.items():
                if now - d.get("created_at", 0) > self.ttl:
                    expired += 1
                    continue
                self._cache[h] = CacheEntry(
                    queries        = d["queries"],
                    selected_tools = d["selected_tools"],
                    normalized_key = d.get("normalized_key", ""),
                    created_at     = d.get("created_at", now),
                    hits           = d.get("hits", 0),
                    last_hit       = d.get("last_hit", now),
                )
                loaded += 1

            logger.info(
                f"✅ Cache loaded from {_CACHE_FILE}",
                loaded=loaded,
                expired_skipped=expired,
                elapsed_ms=f"{t.elapsed_ms:.0f}ms"
            )

        except Exception as e:
            logger.error_with_context(e, {"action": "load_cache", "file": str(_CACHE_FILE)})
            self._cache = {}

    def _save_to_disk(self):
        """Write current cache to JSON file. Called after every set()."""
        try:
            raw = {}
            for h, e in self._cache.items():
                raw[h] = {
                    "queries":        e.queries,
                    "selected_tools": e.selected_tools,
                    "normalized_key": e.normalized_key,
                    "created_at":     e.created_at,
                    "hits":           e.hits,
                    "last_hit":       e.last_hit,
                }
            _CACHE_FILE.write_text(
                json.dumps(raw, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error_with_context(e, {"action": "save_cache", "file": str(_CACHE_FILE)})

    # ─────────────────────────────────────────────────────────────────────────
    # Key normalization
    # ─────────────────────────────────────────────────────────────────────────

    def _normalize(self, question: str) -> str:
        """
        Convert a question into a stable, intent-stripped cache key.

        Steps:
          1. Lowercase
          2. Remove punctuation
          3. Split into words
          4. Remove intent/filler words
          5. Remove duplicates
          6. Sort alphabetically (so word order doesn't matter)
          7. Join with space

        Examples:
          "Balwan Power Weeder purchase karvu" → "balwan weeder"
          "balwan power weeder ni kimat jovu"  → "balwan kimat weeder"
        """
        q     = question.lower().strip()
        q     = re.sub(r"[^\w\s]", " ", q)          # remove punctuation
        words = q.split()
        kept  = sorted(set(
            w for w in words
            if w not in _STRIP_WORDS and len(w) > 1  # skip 1-char words too
        ))
        return " ".join(kept)

    def _make_hash(self, normalized: str) -> str:
        """Short MD5 hash of normalized key for dict lookup."""
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def get(self, question: str) -> Optional[Tuple[List[Dict], List[str]]]:
        """
        Look up question in cache.

        Returns:
            (queries, selected_tools) on hit — caller executes SQL directly
            None on miss — caller runs full LLM pipeline
        """
        key = self._normalize(question)
        if not key:
            return None

        h     = self._make_hash(key)
        entry = self._cache.get(h)

        # Not found
        if entry is None:
            self._misses += 1
            logger.cache_miss(question)
            return None

        # Expired
        if time.time() - entry.created_at > self.ttl:
            del self._cache[h]
            self._misses += 1
            logger.info(f"🕐 Cache EXPIRED: {key[:60]}")
            return None

        # Hit
        entry.hits    += 1
        entry.last_hit = time.time()
        self._hits    += 1

        logger.cache_hit(question, entry.selected_tools)
        logger.info(
            f"⚡ Cache HIT #{entry.hits} | key={key[:60]}",
            tools=str(entry.selected_tools),
            age_minutes=f"{(time.time()-entry.created_at)/60:.1f}min"
        )
        return entry.queries, entry.selected_tools

    def set(
        self,
        question:       str,
        queries:        List[Dict[str, str]],
        selected_tools: List[str],
    ):
        """
        Store SQL queries for this question.
        Persists to disk immediately so the entry survives server restarts.
        """
        key = self._normalize(question)
        if not key or len(key.split()) < 1:
            logger.debug(f"Cache SET skipped — empty key after normalization: {question[:60]}")
            return

        h = self._make_hash(key)

        # LRU eviction when at capacity
        if len(self._cache) >= self.max_size:
            lru_hash    = min(self._cache, key=lambda x: self._cache[x].last_hit)
            lru_key     = self._cache[lru_hash].normalized_key
            del self._cache[lru_hash]
            logger.info(f"🗑️  Cache LRU evict: '{lru_key[:50]}'")

        self._cache[h] = CacheEntry(
            queries        = queries,
            selected_tools = selected_tools,
            normalized_key = key,
        )

        logger.info(
            f"💾 Cache SET: '{key[:60]}'",
            queries=len(queries),
            tools=str(selected_tools),
            file=str(_CACHE_FILE)
        )
        self._save_to_disk()

    def invalidate(self, question: str):
        """Remove a single entry (e.g. when schema changes for one table)."""
        key = self._normalize(question)
        h   = self._make_hash(key)
        if h in self._cache:
            del self._cache[h]
            self._save_to_disk()
            logger.info(f"🗑️  Cache INVALIDATED: '{key[:60]}'")

    def clear(self):
        """
        Wipe all entries.
        Call this after regenerating tool files from full_schema.json,
        since schema changes may invalidate all cached SQL.
        """
        count = len(self._cache)
        self._cache.clear()
        self._save_to_disk()
        logger.info(f"🗑️  Cache CLEARED: {count} entries removed | file={_CACHE_FILE}")

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        total    = self._hits + self._misses
        hit_rate = round(self._hits / total * 100, 1) if total else 0.0
        return {
            "size":                  len(self._cache),
            "hits":                  self._hits,
            "misses":                self._misses,
            "hit_rate_percent":      hit_rate,
            "tokens_saved_estimate": self._hits * 2000,
            "cost_saved_usd":        round(self._hits * 2000 / 1_000_000 * 0.59, 4),
            "cache_file":            str(_CACHE_FILE),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton — one instance shared across all requests
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[QueryCache] = None


def get_query_cache() -> QueryCache:
    global _instance
    if _instance is None:
        _instance = QueryCache(ttl_seconds=3600, max_size=500)
    return _instance