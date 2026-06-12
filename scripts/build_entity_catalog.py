"""
Build the entity catalog: data/entity_catalog.json (v1) + data/entity_catalog_v2.json (v2).

Run manually whenever new data lands in the DB:
    python scripts/build_entity_catalog.py            # reuses cached synonyms for known values
    python scripts/build_entity_catalog.py --force    # regenerate ALL synonyms via Groq

For each table in TABLE_COLUMNS, this script:
  1. Fetches all non-deleted (id, name) pairs. The SAME logic runs for every
     table — an empty table simply produces an empty section (never an error),
     so the build works identically against a backup DB or the live DB.
  2. Sends NEW canonical values to Groq in batches of BATCH_SIZE to generate
     variants in FOUR forms a real Indian farmer might type when searching:
        • English synonyms and translations  (cotton, groundnut, peanut)
        • Romanized Gujarati                 (kapas, magfali, ghau)
        • Hindi script / Devanagari          (कपास, मूंगफली, गेहूं)
        • Romanized Hindi                    (kapas, moongfali, gehu)
     Values already present in the existing v2 catalog reuse their stored
     variants (no Groq cost on rebuild) unless --force is passed.
  3. Writes TWO output files:

     v2 — data/entity_catalog_v2.json (used by the Stage-2 resolver):
        {
          "<table>": {
            "entries": { "<canonical>": {"ids": [42, ...]} },
            "index":   { "<normalized_variant>": "<canonical>" }
          }
        }
        • "canonical" is the EXACT string stored in the DB (including any
          internal whitespace quirks) — safe to use verbatim in SQL.
        • "ids" lists every row id sharing that name (names can repeat,
          e.g. same city name in two states).
        • "index" keys are normalized (NFC, collapsed whitespace, lowercased)
          for O(1) lookup regardless of input language or spacing.

     v1 — data/entity_catalog.json (legacy flat format, read by the current
     runtime entity_normalizer until Phase 2 switches to v2):
        { "<table>": { "<variant>": "<canonical>", ... } }

Dependencies: aiomysql, groq, python-dotenv (all already in requirements.txt).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import aiomysql
from dotenv import load_dotenv
from groq import AsyncGroq

# ── Paths / env ──
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

OUTPUT_V1_PATH = ROOT / "data" / "entity_catalog.json"
OUTPUT_V2_PATH = ROOT / "data" / "entity_catalog_v2.json"

# ── Tables & keyword columns ──
# Every table whose name values users search by. One uniform pipeline for all:
# empty table → empty section, never an error or a special case.
TABLE_COLUMNS: Dict[str, str] = {
    "categories":          "name",   # top-level crop categories (રોકડિયા પાક, અનાજ, ...)
    "sub_categories":      "name",   # crop names incl. varieties (શીંગ મગડી, તલ કાળા, ...)
    "buy_sell_categories": "name",   # used equipment + animal categories
    "kshop_categories":    "name",   # new equipment categories
    "kshop_companies":     "name",   # K-Shop company names
    "seeds":               "name",   # seed variety names (may be empty in backup DB)
    "yards":               "name",   # APMC market yard names
    "talukas":             "name",
    "cities":              "name",
    "states":              "name",
    "news_types":          "name",
}

# Tables whose values are SEMANTIC words (crops, animals, categories) — these
# have a TRUE Hindi word that differs from the Gujarati (કેરી → आम). Geographic
# tables (cities/talukas/yards/states) and company names are proper nouns where
# the Devanagari transliteration IS the correct Hindi — excluded from the
# true-Hindi fix pass.
SEMANTIC_TABLES = {
    "categories", "sub_categories", "buy_sell_categories",
    "kshop_categories", "news_types", "seeds",
}

# ── Groq ──
BATCH_SIZE       = 20
MAX_RETRY_PASSES = 4   # full passes over still-missing values
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYNONYM_PROMPT_TEMPLATE = """\
You are generating SEARCH SYNONYMS for an Indian agricultural marketplace.

Users type queries in FOUR different forms — English, Romanized Gujarati,
Hindi script (Devanagari), and Romanized Hindi. The database stores only
the Gujarati canonical value. Your job: for each Gujarati value below,
produce EVERY form a real Indian farmer might type when searching for it,
covering ALL FOUR forms wherever they exist.

For each Gujarati value, generate variants in ALL of the following forms.
Skip a form only if it has no plausible variant for that specific value.

FORM 1 — ENGLISH
  Every English word the value is known by, plus plural/singular forms.
    મગફળી    → "groundnut", "groundnuts", "peanut", "peanuts"
    કપાસ     → "cotton"
    ડુંગળી   → "onion", "onions"
    ઘઉં      → "wheat"
    ભાવનગર   → "bhavnagar"            (proper noun — transliterate)
    રાજકોટ   → "rajkot"

FORM 2 — ROMANIZED GUJARATI
  Phonetic Latin-script transliteration of the Gujarati value, including
  common spelling variations users actually type.
    મગફળી    → "magfali", "moongfali", "mungphali"
    કપાસ     → "kapas", "kapaas"
    ઘઉં      → "ghau", "gahu", "gehoo"
    ડુંગળી   → "dungli", "dungri"

FORM 3 — HINDI SCRIPT (Devanagari)
  The word HINDI SPEAKERS ACTUALLY USE for the same concept, in Devanagari.
  ⚠ NEVER just transliterate the Gujarati letters into Devanagari — the
  mechanical transliteration is added automatically by code. Give the REAL
  Hindi word, which is often a DIFFERENT word:
    કેરી     → "आम"            (NOT "केरी")
    ડુંગળી   → "प्याज"          (NOT "डुंगली")
    કોબી     → "पत्ता गोभी", "गोभी"
    ડાંગર    → "धान"
    મગફળી    → "मूंगफली", "मूँगफली"
    ઘઉં      → "गेहूं", "गेहूँ"
  When the Hindi word IS the same word (cognates), give it once:
    કપાસ     → "कपास"
  For proper nouns (cities / places / yard names), transliterate:
    ભાવનગર   → "भावनगर"
    રાજકોટ   → "राजकोट"

FORM 4 — ROMANIZED HINDI
  Phonetic Latin-script transliteration of the Hindi word, including
  spelling variations.
    मूंगफली  → "moongfali", "mungphali", "moongphali"
    प्याज    → "pyaaz", "pyaj", "pyaz", "kanda"
    गेहूं    → "gehu", "gehoon", "gehun"

STRICT RULES:
1. Output STRICT JSON only. No commentary, no markdown, no preamble.
2. DO NOT include the original Gujarati value in its own variants list.
3. DO NOT invent alternate Gujarati-script spellings. The original is canonical.
4. Lowercase ALL Latin-script variants. (Devanagari and Gujarati are case-less.)
5. Keep variants short — single words or 2-3 word phrases only.
6. Each value's variants must be a JSON array of strings, never a nested object.
7. If a value genuinely has NO plausible variants in any form (rare), return [].
8. Cover all four forms when they exist. Missing Devanagari for a real word
   is the most common gap — be sure to include it.

Input Gujarati values (JSON array):
{values_json}

Output JSON object — one key per input value, value is an OBJECT with all
four forms (use [] for a form with no plausible variants):
{{
  "<gujarati_value>": {{
    "english": ["...", "..."],
    "romanized_gujarati": ["...", "..."],
    "hindi": ["...", "..."],
    "romanized_hindi": ["...", "..."]
  }},
  ...
}}
"""

HINDI_FIX_PROMPT_TEMPLATE = """\
You are fixing Hindi search synonyms for an Indian agricultural marketplace.

For each GUJARATI term below, give the word(s) a NATIVE HINDI SPEAKER actually
uses for that concept — in Devanagari script — plus romanized spellings of
those Hindi words.

⚠ DO NOT give a letter-by-letter transliteration of the Gujarati term — we
already have those. Only the REAL Hindi word, when it differs:
    કેરી → {{"hindi": ["आम"], "romanized_hindi": ["aam"]}}
    કોબી → {{"hindi": ["पत्ता गोभी", "गोभी"], "romanized_hindi": ["patta gobhi", "gobhi"]}}
    ડાંગર → {{"hindi": ["धान"], "romanized_hindi": ["dhaan", "dhan"]}}

If the Hindi word is THE SAME word as the Gujarati (cognates like કપાસ/कपास,
ગાય/गाय, ઊંટ/ऊंट), return empty lists for that term.

STRICT RULES:
1. Output STRICT JSON only. No commentary.
2. Lowercase all romanized spellings.
3. Empty lists when transliteration already IS the correct Hindi.

Input Gujarati terms (JSON array):
{values_json}

Output JSON object:
{{
  "<gujarati_term>": {{"hindi": [...], "romanized_hindi": [...]}},
  ...
}}
"""


# ── Key normalization ────────────────────────────────────────────────────────

def norm_key(s: str) -> str:
    """Normalize a lookup key: NFC, collapse whitespace, lowercase.

    Matches what language_processor does to incoming queries, so a user-typed
    keyword and a catalog key always normalize identically. Canonical VALUES
    are never normalized — they stay byte-exact with the DB for SQL use.
    """
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def guj_to_dev(s: str) -> str:
    """Deterministic Gujarati → Devanagari transliteration.

    The Gujarati (U+0A80–0AFF) and Devanagari (U+0900–097F) Unicode blocks are
    structurally parallel — a fixed offset of 0x180 maps each Gujarati letter
    to its Devanagari counterpart. This gives the 'Gujarati-influenced Hindi
    typing' form (કેરી → केरी) for EVERY entity with zero LLM cost. The TRUE
    Hindi word (आम), when different, comes from the LLM fix pass.
    """
    return "".join(
        chr(ord(c) - 0x180) if 0x0A80 <= ord(c) <= 0x0AFF else c
        for c in s
    )


def _has_devanagari(s: str) -> bool:
    return any(0x0900 <= ord(c) <= 0x097F for c in s)


def _strip_combining_marks(s: str) -> str:
    """Drop anusvara/chandrabindu/nukta so near-identical spellings compare equal."""
    return "".join(c for c in s if ord(c) not in (0x0901, 0x0902, 0x093C))


# ── DB ──────────────────────────────────────────────────────────────────────

async def fetch_name_ids(
    pool: aiomysql.Pool, table: str, column: str
) -> Dict[str, List[int]]:
    """Fetch {exact_name: [row_ids]} for one table, skipping soft-deleted rows.

    The name key is the EXACT DB string (outer whitespace trimmed, internal
    whitespace preserved). Multiple rows sharing one name accumulate ids.
    """
    sql = (
        f"SELECT id, `{column}` AS v FROM `{table}` "
        f"WHERE `{column}` IS NOT NULL AND TRIM(`{column}`) <> '' "
        f"AND deleted_at IS NULL"
    )
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql)
            rows = await cur.fetchall()

    out: Dict[str, List[int]] = {}
    for row_id, v in rows:
        name = (v or "").strip()
        if not name:
            continue
        out.setdefault(name, []).append(int(row_id))
    return out


async def make_pool() -> aiomysql.Pool:
    return await aiomysql.create_pool(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        db=os.getenv("DB_NAME", "krushi_node"),
        charset="utf8mb4",
        autocommit=True,
        minsize=1,
        maxsize=4,
    )


# ── Synonym reuse (avoid re-paying Groq for values already cataloged) ────────

def load_existing_synonyms() -> Dict[str, Dict[str, List[str]]]:
    """Rebuild {table: {canonical: [variants]}} by inverting a previous catalog.

    Prefers the v2 index; falls back to the legacy v1 flat file so the FIRST
    v2 build reuses all previously generated synonyms instead of re-paying
    Groq for ~4,500 values. Returns {} when no previous catalog exists.
    """
    per_table_index: Dict[str, Dict[str, str]] = {}

    if OUTPUT_V2_PATH.exists():
        try:
            with OUTPUT_V2_PATH.open("r", encoding="utf-8") as f:
                old = json.load(f) or {}
            for table, payload in old.items():
                if isinstance(payload, dict) and isinstance(payload.get("index"), dict):
                    per_table_index[table] = payload["index"]
        except Exception as e:
            print(f"⚠ Could not read existing v2 catalog ({e})")

    if not per_table_index and OUTPUT_V1_PATH.exists():
        try:
            with OUTPUT_V1_PATH.open("r", encoding="utf-8") as f:
                old_v1 = json.load(f) or {}
            for table, flat in old_v1.items():
                if isinstance(flat, dict):
                    per_table_index[table] = flat
            if per_table_index:
                print("♻ Bootstrapping synonym cache from legacy v1 catalog")
        except Exception as e:
            print(f"⚠ Could not read existing v1 catalog ({e}) — full regeneration")

    out: Dict[str, Dict[str, List[str]]] = {}
    for table, index in per_table_index.items():
        per_canonical: Dict[str, List[str]] = {}
        for variant, canonical in index.items():
            if not isinstance(variant, str) or not isinstance(canonical, str):
                continue
            if norm_key(variant) == norm_key(canonical):
                continue  # self-reference, not a synonym
            per_canonical.setdefault(canonical, []).append(variant)
        out[table] = per_canonical
    return out


# ── Groq ────────────────────────────────────────────────────────────────────

async def _groq_json_call(client: AsyncGroq, prompt: str, max_tokens: int):
    """One Groq JSON-mode call with rate-limit backoff (429 → wait → retry)."""
    for attempt in range(4):
        try:
            return await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                wait = 20 * (attempt + 1)
                print(f"      ⏳ rate limited — waiting {wait}s")
                await asyncio.sleep(wait)
                continue
            raise
    raise RuntimeError("rate limit retries exhausted")


async def call_groq_for_batch(
    client: AsyncGroq, batch: List[str]
) -> Dict[str, List[str]]:
    """Single Groq call for one batch. Returns the parsed dict or {} on failure."""
    prompt = SYNONYM_PROMPT_TEMPLATE.format(
        values_json=json.dumps(batch, ensure_ascii=False)
    )
    try:
        # ~12-15 variants per value × 20 values per batch + JSON structure
        # ≈ 1500-2000 tokens of output. 3500 gives safe headroom against
        # the LLM elaborating slightly, without burning the rate limit.
        response = await _groq_json_call(client, prompt, max_tokens=3500)
    except Exception as e:
        print(f"      ⚠ Groq call failed: {e}")
        return {}

    content = (response.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"      ⚠ JSON parse failed: {e}")
        return {}
    if not isinstance(parsed, dict):
        return {}

    cleaned: Dict[str, List[str]] = {}
    for k, v in parsed.items():
        if not isinstance(k, str):
            continue
        variants: List[str] = []
        if isinstance(v, list):
            # Legacy flat shape.
            variants = [str(x).strip() for x in v if isinstance(x, str) and x.strip()]
        elif isinstance(v, dict):
            # Per-form shape — flatten all four form lists.
            for form_list in v.values():
                if isinstance(form_list, list):
                    variants.extend(
                        str(x).strip() for x in form_list
                        if isinstance(x, str) and x.strip()
                    )
        cleaned[k] = variants
    return cleaned


async def collect_synonyms(
    client: AsyncGroq, values: List[str]
) -> Dict[str, List[str]]:
    """
    Process all values in BATCH_SIZE chunks. Re-batches any value Groq dropped
    from its response. Guarantees every input value gets a key in the result
    (possibly an empty list after MAX_RETRY_PASSES).
    """
    result: Dict[str, List[str]] = {}
    pending = list(values)
    pass_no = 0

    while pending and pass_no < MAX_RETRY_PASSES:
        pass_no += 1
        print(f"    pass {pass_no} — {len(pending)} value(s) remaining")
        new_missing: List[str] = []

        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i : i + BATCH_SIZE]
            parsed = await call_groq_for_batch(client, batch)
            for v in batch:
                if v in parsed:
                    result[v] = parsed[v]
                else:
                    new_missing.append(v)
            # Courtesy spacing between batches — keeps long runs inside
            # Groq's requests-per-minute budget without hard throttling.
            await asyncio.sleep(2)

        if new_missing == pending:
            # No progress this pass — stop to avoid infinite retry.
            print(f"    ⚠ no progress on {len(new_missing)} value(s) — stopping retries")
            break
        pending = new_missing

    # Anything still missing gets an empty list so it appears in the catalog.
    for v in pending:
        result.setdefault(v, [])

    return result


# ── True-Hindi fix pass ──────────────────────────────────────────────────────

def detect_hindi_fix_candidates(
    name_ids: Dict[str, List[int]], synonyms: Dict[str, List[str]]
) -> List[str]:
    """Canonicals whose Devanagari variants are ONLY the mechanical
    transliteration (or absent) — i.e. the true Hindi word may be missing.
    Cognates (કપાસ/कपास) are also flagged here; the LLM fix pass returns []
    for those, which is correct and costs nothing extra."""
    candidates: List[str] = []
    for canonical in name_ids:
        translit = _strip_combining_marks(norm_key(guj_to_dev(canonical)))
        dev_variants = [
            v for v in synonyms.get(canonical, []) if _has_devanagari(v)
        ]
        if all(
            _strip_combining_marks(norm_key(v)) == translit
            for v in dev_variants
        ):  # vacuously true when no Devanagari variant exists at all
            candidates.append(canonical)
    return candidates


async def run_hindi_fix(
    client: AsyncGroq, candidates: List[str]
) -> Dict[str, List[str]]:
    """Ask Groq for the TRUE Hindi word (+ romanizations) per candidate.
    Returns {canonical: [extra_variants]}; cognates come back empty."""
    fixes: Dict[str, List[str]] = {}
    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i : i + BATCH_SIZE]
        prompt = HINDI_FIX_PROMPT_TEMPLATE.format(
            values_json=json.dumps(batch, ensure_ascii=False)
        )
        try:
            response = await _groq_json_call(client, prompt, max_tokens=2500)
            parsed = json.loads((response.choices[0].message.content or "").strip())
        except Exception as e:
            print(f"      ⚠ Hindi fix call failed: {e}")
            continue
        if not isinstance(parsed, dict):
            continue
        for term, forms in parsed.items():
            if not isinstance(term, str) or not isinstance(forms, dict):
                continue
            extra: List[str] = []
            for form_list in forms.values():
                if isinstance(form_list, list):
                    extra.extend(
                        str(x).strip() for x in form_list
                        if isinstance(x, str) and x.strip()
                    )
            if extra:
                fixes[term] = extra
    return fixes


# ── Catalog assembly ─────────────────────────────────────────────────────────

def build_table_catalog(
    name_ids: Dict[str, List[int]], synonyms: Dict[str, List[str]]
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Build one table's v2 payload {entries, index} and derived v1 flat dict.

    Index keys are normalized via norm_key. On variant collision across two
    canonicals, the FIRST canonical wins (stable, same rule as v1 always had).
    Every Gujarati canonical also gets its deterministic Devanagari
    transliteration as a key — guaranteed 100% coverage of the
    'Hindi-typed-like-Gujarati' form (કેરી → केरी) at zero LLM cost.
    """
    entries: Dict[str, dict] = {}
    index: Dict[str, str] = {}

    for canonical, ids in name_ids.items():
        entries[canonical] = {"ids": sorted(ids)}
        # Self-reference so direct-Gujarati input always resolves.
        index.setdefault(norm_key(canonical), canonical)
        # Deterministic Devanagari transliteration (code, not LLM).
        translit = norm_key(guj_to_dev(canonical))
        if translit and translit != norm_key(canonical):
            index.setdefault(translit, canonical)
        for syn in synonyms.get(canonical, []):
            key = norm_key(syn)
            if not key:
                continue
            index.setdefault(key, canonical)

    # v1 flat format: index plus the raw canonical self-reference the old
    # runtime normalizer expects (it lowercases keys itself at load time).
    v1_flat: Dict[str, str] = dict(index)
    for canonical in name_ids:
        v1_flat.setdefault(canonical, canonical)

    return {"entries": entries, "index": index}, v1_flat


# ── Main ────────────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(description="Build entity catalog (v1 + v2)")
    parser.add_argument(
        "--refresh", action="store_true",
        help="Regenerate ALL synonyms with the current prompt and MERGE (union) with "
             "cached variants. Use once after the generation prompt improves. "
             "Old keys are preserved — recall can only grow.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate ALL synonyms and DISCARD the cache (only when old data is known-bad)",
    )
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("✗ GROQ_API_KEY missing in .env")
        return 1

    print(f"Output v1: {OUTPUT_V1_PATH}")
    print(f"Output v2: {OUTPUT_V2_PATH}")
    OUTPUT_V2_PATH.parent.mkdir(parents=True, exist_ok=True)

    cached_synonyms = {} if args.force else load_existing_synonyms()
    if cached_synonyms:
        cached_total = sum(len(t) for t in cached_synonyms.values())
        print(f"♻ Reusing cached synonyms for {cached_total} canonical value(s) "
              f"(pass --force to regenerate all)")

    client = AsyncGroq(api_key=api_key)
    pool = await make_pool()
    catalog_v1: Dict[str, Dict[str, str]] = {}
    catalog_v2: Dict[str, dict] = {}

    try:
        for table, column in TABLE_COLUMNS.items():
            print(f"\n▶ {table}.{column}")
            try:
                name_ids = await fetch_name_ids(pool, table, column)
            except Exception as e:
                # Same shape on failure as on empty — uniform handling.
                print(f"  ✗ DB fetch failed: {e}")
                catalog_v2[table] = {"entries": {}, "index": {}}
                catalog_v1[table] = {}
                continue

            print(f"  fetched {len(name_ids)} distinct value(s)")
            if not name_ids:
                catalog_v2[table] = {"entries": {}, "index": {}}
                catalog_v1[table] = {}
                continue

            table_cache = cached_synonyms.get(table, {})
            if args.refresh:
                to_generate = list(name_ids)
            else:
                to_generate = [v for v in name_ids if v not in table_cache]

            # Start from cached variants for every value that has them.
            synonyms: Dict[str, List[str]] = {
                v: list(table_cache[v]) for v in name_ids if v in table_cache
            }
            if to_generate:
                mode = "refreshing" if args.refresh else "generating"
                print(f"  {mode} synonyms for {len(to_generate)} value(s) "
                      f"({len(synonyms)} have cached variants)")
                fresh = await collect_synonyms(client, to_generate)
                # MERGE (union) fresh + cached — dedupe on normalized key so
                # rebuilds never lose previously working variants.
                for v, fresh_variants in fresh.items():
                    merged = synonyms.get(v, []) + fresh_variants
                    seen: set = set()
                    deduped: List[str] = []
                    for s in merged:
                        k = norm_key(s)
                        if k and k not in seen:
                            seen.add(k)
                            deduped.append(s)
                    synonyms[v] = deduped
            else:
                print(f"  all {len(synonyms)} value(s) reused from cache — no Groq calls")

            # True-Hindi fix pass — semantic tables only (crops/animals/categories).
            # Geographic + company names are proper nouns where transliteration
            # IS the correct Hindi, so they are skipped.
            if table in SEMANTIC_TABLES:
                candidates = detect_hindi_fix_candidates(name_ids, synonyms)
                if candidates:
                    print(f"  hindi fix pass: {len(candidates)} candidate(s) "
                          f"(transliteration-only Devanagari)")
                    fixes = await run_hindi_fix(client, candidates)
                    for term, extra in fixes.items():
                        synonyms.setdefault(term, []).extend(extra)
                    print(f"  hindi fix pass: true-Hindi words added for "
                          f"{len(fixes)} value(s)")

            v2_payload, v1_flat = build_table_catalog(name_ids, synonyms)
            catalog_v2[table] = v2_payload
            catalog_v1[table] = v1_flat
            print(f"  index entries: {len(v2_payload['index'])} | "
                  f"canonicals: {len(v2_payload['entries'])}")
    finally:
        pool.close()
        await pool.wait_closed()

    with OUTPUT_V2_PATH.open("w", encoding="utf-8") as f:
        json.dump(catalog_v2, f, ensure_ascii=False, indent=2, sort_keys=True)
    with OUTPUT_V1_PATH.open("w", encoding="utf-8") as f:
        json.dump(catalog_v1, f, ensure_ascii=False, indent=2, sort_keys=True)

    total_idx = sum(len(t.get("index", {})) for t in catalog_v2.values())
    total_can = sum(len(t.get("entries", {})) for t in catalog_v2.values())
    print(f"\n✅ Wrote v2 ({total_can} canonicals, {total_idx} index entries, "
          f"{len(catalog_v2)} tables) and v1 (legacy runtime format)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
