"""
Build data/entity_catalog.json.

Run manually whenever new data lands in the DB:
    python scripts/build_entity_catalog.py

For each table in TABLE_COLUMNS, this script:
  1. Fetches all DISTINCT non-null values from the configured column.
  2. Sends them to Groq in batches of BATCH_SIZE values to generate variants
     in FOUR forms a real Indian farmer might type when searching:
        • English synonyms and translations  (cotton, groundnut, peanut)
        • Romanized Gujarati                 (kapas, magfali, ghau)
        • Hindi script / Devanagari          (कपास, मूंगफली, गेहूं)
        • Romanized Hindi                    (kapas, moongfali, gehu)
  3. Builds a flat inverted index per table:
        {variant_lower: canonical_gujarati,
         canonical_gujarati: canonical_gujarati,
         hindi_script_variant: canonical_gujarati,
         ...}
     Flat dict — industry-standard for entity-alias lookup (Lucene, Solr,
     Wikidata). O(1) lookup regardless of which language the user typed in,
     so the runtime normalizer doesn't need to know the user's language.
  4. Writes the catalog to data/entity_catalog.json (UTF-8).

The runtime entity_normalizer reads this file once at startup and uses it
to map user keywords to the canonical Gujarati value the DB actually stores,
eliminating LLM hallucination of multi-language variants in SQL.

Dependencies: aiomysql, groq, python-dotenv (all already in requirements.txt).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import aiomysql
from dotenv import load_dotenv
from groq import AsyncGroq

# ── Paths / env ──
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

OUTPUT_PATH = ROOT / "data" / "entity_catalog.json"

# ── Tables & keyword columns ──
# These are the EXACT tables/columns that hold the keyword values
# users search by. Match the spec from the project lead.
TABLE_COLUMNS: Dict[str, str] = {
    "buy_sell_categories": "name",
    "kshop_categories":    "name",
    "seeds":               "name",
    "sub_categories":      "name",
    "yards":               "name",
    "talukas":             "name",
    "states":              "name",
    "news_types":          "name",
    "cities":              "name",
}

# ── Groq ──
BATCH_SIZE      = 20
MAX_RETRY_PASSES = 4   # full passes over still-missing values
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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
  The Hindi word for the same concept written in Devanagari. For proper
  nouns (cities / places / yard names), transliterate to Devanagari.
    મગફળી    → "मूंगफली", "मूँगफली"
    કપાસ     → "कपास"
    ડુંગળી   → "प्याज"
    ઘઉં      → "गेहूं", "गेहूँ"
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

Output JSON object — one key per input value, value is an array of variant strings:
{{
  "<gujarati_value>": ["variant1", "variant2", ...],
  ...
}}
"""


# ── DB ──────────────────────────────────────────────────────────────────────

async def fetch_distinct_values(pool: aiomysql.Pool, table: str, column: str) -> List[str]:
    """Fetch distinct, non-null, non-empty values from one table column.

    All tables in TABLE_COLUMNS have a deleted_at column — filter soft-deleted rows.
    """
    sql = (
        f"SELECT DISTINCT `{column}` AS v FROM `{table}` "
        f"WHERE `{column}` IS NOT NULL AND TRIM(`{column}`) <> '' "
        f"AND deleted_at IS NULL"
    )
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql)
            rows = await cur.fetchall()

    out: List[str] = []
    seen: set = set()
    for row in rows:
        v = (row[0] or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
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


# ── Groq ────────────────────────────────────────────────────────────────────

async def call_groq_for_batch(
    client: AsyncGroq, batch: List[str]
) -> Dict[str, List[str]]:
    """Single Groq call for one batch. Returns the parsed dict or {} on failure."""
    prompt = SYNONYM_PROMPT_TEMPLATE.format(
        values_json=json.dumps(batch, ensure_ascii=False)
    )
    try:
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            # ~12-15 variants per value × 20 values per batch + JSON structure
            # ≈ 1500-2000 tokens of output. 3500 gives safe headroom against
            # the LLM elaborating slightly, without burning the rate limit.
            max_tokens=3500,
            response_format={"type": "json_object"},
        )
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
        if isinstance(v, list):
            variants = [str(x).strip() for x in v if isinstance(x, str) and x.strip()]
        else:
            variants = []
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

        if new_missing == pending:
            # No progress this pass — stop to avoid infinite retry.
            print(f"    ⚠ no progress on {len(new_missing)} value(s) — stopping retries")
            break
        pending = new_missing

    # Anything still missing gets an empty list so it appears in the catalog.
    for v in pending:
        result.setdefault(v, [])

    return result


# ── Inverted index ──────────────────────────────────────────────────────────

def build_inverted_index(
    values: List[str], synonyms: Dict[str, List[str]]
) -> Dict[str, str]:
    """
    Build: {english_variant_lower: canonical_gujarati,
            canonical_gujarati:    canonical_gujarati}
    On synonym collision across two canonicals, FIRST canonical wins (stable).
    """
    out: Dict[str, str] = {}
    for canonical in values:
        # Self-reference so direct-Gujarati input always resolves.
        out.setdefault(canonical, canonical)
        for syn in synonyms.get(canonical, []):
            key = syn.strip().lower()
            if not key:
                continue
            out.setdefault(key, canonical)
    return out


# ── Main ────────────────────────────────────────────────────────────────────

async def main() -> int:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("✗ GROQ_API_KEY missing in .env")
        return 1

    print(f"Output: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = AsyncGroq(api_key=api_key)
    pool   = await make_pool()
    catalog: Dict[str, Dict[str, str]] = {}

    try:
        for table, column in TABLE_COLUMNS.items():
            print(f"\n▶ {table}.{column}")
            try:
                values = await fetch_distinct_values(pool, table, column)
            except Exception as e:
                print(f"  ✗ DB fetch failed: {e}")
                catalog[table] = {}
                continue

            print(f"  fetched {len(values)} distinct value(s)")
            if not values:
                catalog[table] = {}
                continue

            synonyms = await collect_synonyms(client, values)
            inverted = build_inverted_index(values, synonyms)
            catalog[table] = inverted
            print(f"  index entries: {len(inverted)}")
    finally:
        pool.close()
        await pool.wait_closed()

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2, sort_keys=True)

    total = sum(len(t) for t in catalog.values())
    print(f"\n✅ Wrote {OUTPUT_PATH} ({total} total entries across {len(catalog)} tables)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
