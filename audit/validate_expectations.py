"""
Validate audit/expected.json entities against the entity catalog.

WHY THIS EXISTS
---------------
audit/questions.md was AI-generated, so some questions may name crops, yards,
or categories that don't exist in the database. For those questions an honest
"no data available" answer is CORRECT behavior — not a pipeline bug. This
script checks every catalog-checkable expected entity against the entity
catalog (which is built directly from the DB) and writes a review report, so
nobody has to manually cross-check 114 questions against the database.

Usage:
    python audit/validate_expectations.py
        reads:  audit/expected.json
                data/entity_catalog_v2.json   (preferred — has ids)
                data/entity_catalog.json      (fallback — legacy flat format)
                data/entity_catalog_manual.json (manual overrides, if present)
        writes: audit/expectations_report.md

Entity statuses:
    EXACT    — normalized value found in the catalog index (entity exists in DB)
    PARTIAL  — value is a substring of a canonical (or vice versa); lists hits.
               e.g. "તલ" → તલ કાળા, તલ સફેદ — fine for LIKE/IN searches.
    UNKNOWN  — no catalog match. Either the question invented it (fix the
               expectation) or the catalog is missing a synonym (add to
               data/entity_catalog_manual.json).
    SKIP     — free-text entity type (topic / identifier) — not checkable.

Exit code is always 0 — this is a report generator, not a gate.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

EXPECTED_PATH = SCRIPT_DIR / "expected.json"
CATALOG_V2_PATH = ROOT / "data" / "entity_catalog_v2.json"
CATALOG_V1_PATH = ROOT / "data" / "entity_catalog.json"
CATALOG_MANUAL_PATH = ROOT / "data" / "entity_catalog_manual.json"
REPORT_PATH = SCRIPT_DIR / "expectations_report.md"

# Entity type → catalog tables to search. "category" is scoped by module
# (see _tables_for) because each domain has its own category table.
ENTITY_TABLES: Dict[str, List[str]] = {
    "crop":      ["sub_categories"],
    "category":  ["categories", "kshop_categories", "buy_sell_categories"],
    "location":  ["cities", "talukas", "yards", "states"],
    "equipment": ["kshop_categories", "buy_sell_categories"],
    "animal":    ["buy_sell_categories"],
    "news_type": ["news_types"],
}
FREE_TEXT_TYPES = {"topic", "identifier"}

MODULE_CATEGORY_SCOPE = {
    "4A": ["categories"],
    "4B": ["kshop_categories"],
    "4C": ["buy_sell_categories"],
}


def norm_key(s: str) -> str:
    """Same normalization as scripts/build_entity_catalog.py — keep in sync."""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        print(f"⚠ Could not read {path.name}: {e}")
        return {}


def load_catalog() -> Tuple[Dict[str, Dict[str, str]], Dict[str, Set[str]], str]:
    """Load the catalog into uniform lookup structures.

    Returns:
        index:      {table: {normalized_key: canonical}}
        canonicals: {table: {canonical, ...}}
        source:     which file was used (for the report header)
    """
    index: Dict[str, Dict[str, str]] = {}
    canonicals: Dict[str, Set[str]] = {}

    v2 = _load_json(CATALOG_V2_PATH)
    if v2:
        source = CATALOG_V2_PATH.name
        for table, payload in v2.items():
            if not isinstance(payload, dict):
                continue
            idx = payload.get("index", {})
            index[table] = {norm_key(k): v for k, v in idx.items()
                            if isinstance(k, str) and isinstance(v, str)}
            canonicals[table] = {c for c in payload.get("entries", {})
                                 if isinstance(c, str)}
    else:
        v1 = _load_json(CATALOG_V1_PATH)
        source = CATALOG_V1_PATH.name if v1 else "(no catalog found)"
        for table, flat in (v1 or {}).items():
            if not isinstance(flat, dict):
                continue
            index[table] = {norm_key(k): v for k, v in flat.items()
                            if isinstance(k, str) and isinstance(v, str)}
            canonicals[table] = {v for v in flat.values() if isinstance(v, str)}

    # Manual overrides (flat v1 format) merge on top — they always win.
    manual = _load_json(CATALOG_MANUAL_PATH)
    for table, flat in (manual or {}).items():
        if not isinstance(flat, dict):
            continue
        index.setdefault(table, {})
        canonicals.setdefault(table, set())
        for k, v in flat.items():
            if isinstance(k, str) and isinstance(v, str):
                index[table][norm_key(k)] = v
                canonicals[table].add(v)

    return index, canonicals, source


def _tables_for(entity_type: str, module: str) -> List[str]:
    tables = ENTITY_TABLES.get(entity_type, [])
    if entity_type == "category":
        for prefix, scoped in MODULE_CATEGORY_SCOPE.items():
            if module.startswith(prefix):
                return scoped
    return tables


def check_entity(
    value: str,
    tables: List[str],
    index: Dict[str, Dict[str, str]],
    canonicals: Dict[str, Set[str]],
) -> Tuple[str, str]:
    """Return (status, detail) for one entity value against the given tables."""
    key = norm_key(value)

    # 1. Exact index hit (covers canonical names AND all generated synonyms).
    for table in tables:
        hit = index.get(table, {}).get(key)
        if hit:
            return "EXACT", f"{table} → {hit}"

    # 2. Substring containment against canonicals (both directions).
    partial_hits: List[str] = []
    for table in tables:
        for canonical in canonicals.get(table, set()):
            ck = norm_key(canonical)
            if key in ck or ck in key:
                partial_hits.append(f"{table} → {canonical}")
                if len(partial_hits) >= 6:
                    break
        if len(partial_hits) >= 6:
            break
    if partial_hits:
        return "PARTIAL", "; ".join(partial_hits)

    return "UNKNOWN", f"searched tables: {', '.join(tables) or '(none)'}"


def main() -> int:
    expected = _load_json(EXPECTED_PATH)
    expectations = expected.get("expectations", [])
    if not expectations:
        print(f"✗ No expectations found in {EXPECTED_PATH}")
        return 0

    index, canonicals, source = load_catalog()
    if not index:
        print("✗ No catalog available — run scripts/build_entity_catalog.py first.")
        return 0

    lines: List[str] = [
        "# Expectations vs Catalog — Validation Report",
        "",
        f"Catalog source: `{source}`  |  Expectations: {len(expectations)} triplets "
        f"(questions 1–114)",
        "",
        "| Status | Meaning |",
        "|---|---|",
        "| ✅ EXACT | entity exists in DB (catalog hit) — data CAN exist |",
        "| 🟡 PARTIAL | substring match — LIKE/IN search will find these canonicals |",
        "| ❌ UNKNOWN | not in catalog — likely AI-invented question keyword OR missing synonym |",
        "| ➖ SKIP | free-text (topic/identifier) — not catalog-checkable |",
        "",
        "## Per-question results",
        "",
        "| Q# | Intent | Entity (type) | Status | Match |",
        "|---|---|---|---|---|",
    ]

    counts = {"EXACT": 0, "PARTIAL": 0, "UNKNOWN": 0, "SKIP": 0}
    unknowns: List[str] = []
    review_rows: List[str] = []

    for exp in expectations:
        qids = exp.get("qids", [])
        qlabel = f"{qids[0]}–{qids[-1]}" if qids else "?"
        module = exp.get("module", "")
        intent = exp.get("expected_intent", "")
        entities: Dict[str, List[str]] = exp.get("expected_entities", {}) or {}

        if not entities:
            lines.append(f"| {qlabel} | {intent} | — | — | no entities expected |")
        for etype, values in entities.items():
            for value in values:
                if etype in FREE_TEXT_TYPES:
                    counts["SKIP"] += 1
                    lines.append(f"| {qlabel} | {intent} | {value} ({etype}) | ➖ SKIP | free text |")
                    continue
                tables = _tables_for(etype, module)
                status, detail = check_entity(value, tables, index, canonicals)
                counts[status] += 1
                icon = {"EXACT": "✅", "PARTIAL": "🟡", "UNKNOWN": "❌"}[status]
                lines.append(f"| {qlabel} | {intent} | {value} ({etype}) | {icon} {status} | {detail} |")
                if status == "UNKNOWN":
                    unknowns.append(f"- Q{qlabel} [{etype}] **{value}** — fix the expectation "
                                    f"if AI-invented, or add a synonym to entity_catalog_manual.json")

        note = (exp.get("review_note") or "").strip()
        if note:
            review_rows.append(f"- Q{qlabel} ({intent}): {note}")

    lines += [
        "",
        "## Summary",
        "",
        f"- ✅ EXACT: {counts['EXACT']}   🟡 PARTIAL: {counts['PARTIAL']}   "
        f"❌ UNKNOWN: {counts['UNKNOWN']}   ➖ SKIP: {counts['SKIP']}",
        "",
    ]
    if unknowns:
        lines += ["## ❌ Unknown entities — needs your decision", ""] + unknowns + [""]
    if review_rows:
        lines += ["## Review notes (label decisions awaiting confirmation)", ""] + review_rows + [""]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Report written: {REPORT_PATH}")
    print(f"   EXACT={counts['EXACT']} PARTIAL={counts['PARTIAL']} "
          f"UNKNOWN={counts['UNKNOWN']} SKIP={counts['SKIP']}")
    if counts["UNKNOWN"]:
        print(f"   ⚠ {counts['UNKNOWN']} unknown entit(ies) — see the report's decision list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
