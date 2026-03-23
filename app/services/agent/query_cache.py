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
from typing import Dict, List, Optional, Tuple
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
# Intent / filler words to strip before building the cache key.
# These words express HOW the user wants something, not WHAT they want.
# Stripping them means "balwan weeder joiyu" and "balwan weeder apo"
# both resolve to the same normalized key: "balwan weeder"
# ─────────────────────────────────────────────────────────────────────────────
_STRIP_WORDS = {
    # Gujarati intent/filler
    "mare", "maro", "mari", "mara", "mane",
    "karvu", "karvani", "karvo", "karu", "karshu",
    "che", "chhe", "hatu", "hati", "hashe",
    "joiyu", "joiye", "jovu", "joi", "juo",
    "levu", "levo", "levi", "lidhu", "laao",
    "kharidi", "kharido", "vechan", "vecho",
    "thi", "ma", "no", "ni", "na", "nu", "ne", "ane",
    "apo", "apso", "batao", "batavo", "batavsho",
    "pan", "pani", "ne", "kem", "shu",
    # Hindi intent/filler
    "mujhe", "mera", "mere", "meri", "hain", "hai",
    "karo", "karna", "chahiye", "dijiye", "batao",
    "ka", "ki", "ke", "ko", "se", "mein",
    # English intent/filler
    "please", "pls", "i", "want", "show", "me", "tell",
    "get", "find", "give", "fetch", "list", "display",
    "what", "is", "are", "the", "a", "an",
    "of", "in", "at", "for", "to", "my", "about",
    "purchase", "buy", "from", "kshop",
    "can", "you", "could", "would", "help",
}


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
