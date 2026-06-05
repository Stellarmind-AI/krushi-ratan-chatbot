"""Resolve raw image filenames returned from SQL queries into full S3 URLs.

The DB stores image *names* only (e.g. `tractor.jpg`).  The frontend renders
images, which requires a URL.  S3 layout is `<BASE>/<folder>/<filename>`
where the folder depends on which table the image originated from.

──────────────────────────────────────────────────────────────────────────
HOW WE DETERMINE THE FOLDER (in order):

  1. SQL-parsed source table.
       We parse the SQL with sqlglot, build an `output_alias → source_table`
       map by walking each top-level projection back to its first
       `<table_alias>.<column>` and resolving `<table_alias>` against the
       outer FROM/JOIN clauses.  If the source table is in our folder map,
       that's the answer.  This makes folder resolution INDEPENDENT of how
       the LLM happens to alias the column (subcategory_img, crop_img,
       cat_img, sc_img — all resolve correctly).

  2. Explicit field-name overrides.
       For `product_image`/`product_images` the SQL-parsed source is an
       auxiliary table (`media` for kshop, `buy_sell_products` itself for
       the JSON_EXTRACT case) whose folder isn't its own name — the file
       logically belongs to a *product* folder.  An explicit alias-→-folder
       map covers these.

  3. Suffix-pattern fallback (`<table>_img|image|images`).
       Used only when sqlglot fails to parse or the alias falls outside
       both prior cases.  Keeps backwards compatibility.

──────────────────────────────────────────────────────────────────────────
VALUE-SHAPE HANDLING — image fields are not consistent across sources:

    None / empty            → pass through unchanged
    plain string            → single URL string  ("a.jpg" → ".../a.jpg")
    Python list             → list of URL strings
    JSON-array string       → parsed via json.loads, then list of URL strings
                                e.g. '["a.jpg", "b.jpg"]'
    JSON-quoted string      → parsed via json.loads (handles JSON_EXTRACT on
                                a single value, which returns '"a.jpg"')
    already-a-URL           → pass through unchanged (idempotent)
    anything else           → pass through unchanged (never mutate junk)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

from app.core.config import settings
from app.core.logger import get_logger

try:
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover — sqlglot is a hard dep elsewhere too
    sqlglot = None
    exp = None

logger = get_logger("image_resolver")


# Field-name → S3 folder map for cases where the SQL-parsed source table
# is NOT the right folder (e.g. `product_image` is built from `media` rows
# but the files live under `kshopProducts/` — `media` itself has no folder).
_EXPLICIT_FIELD_FOLDER: Mapping[str, str] = {
    "product_image":   "kshopProducts",
    "product_images":  "buy_sell_products",
}

# Real-table-name → S3 folder map.  Used by the SQL-parsed-source path
# (primary) and the suffix-pattern fallback path.
_TABLE_FOLDER: Mapping[str, str] = {
    "categories":           "categories",
    "sub_categories":       "sub-categories",
    "seeds":                "seeds",
    "kshop_categories":     "k_shop_categories",
    "kshop_companies":      "k_shop_companies",
    "kshop_products":       "kshopProducts",
    "news":                 "news",
    "buy_sell_products":    "buy_sell_products",
    "buy_sell_categories":  "buy_sell_categories",
}

# Recognise an output field as an image field by its name shape so we don't
# accidentally URL-ify, say, a `description` column whose value happens to
# be the literal text "image.jpg".
_IMAGE_SUFFIX_RE = re.compile(r".+_(?:img|image|images)$|^(?:img|image|images)$")

# Suffix-pattern fallback: extract `<table>` prefix from `<table>_img`.
_TABLE_FIELD_RE = re.compile(r"^(?P<table>[a-z][a-z0-9_]+?)_(?:img|image|images)$")


# ── SQL parsing helpers ──────────────────────────────────────────────────

def _is_in_subquery(node: Any, outer: Any) -> bool:
    """Return True if `node` lives inside a Subquery relative to `outer`."""
    if exp is None:
        return False
    parent = getattr(node, "parent", None)
    while parent is not None and parent is not outer:
        if isinstance(parent, exp.Subquery):
            return True
        parent = getattr(parent, "parent", None)
    return False


def _build_alias_to_source_table(sql: Optional[str]) -> Dict[str, str]:
    """Map each top-level output alias to the real table its first column came from.

    Empty dict if sqlglot is unavailable or the SQL doesn't parse.  We only
    consider the OUTER query's projections — subquery columns are not
    user-facing aliases.
    """
    if sqlglot is None or exp is None or not sql:
        return {}
    try:
        parsed = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return {}
    if not isinstance(parsed, exp.Select):
        return {}

    # Resolve table aliases used in the outer FROM/JOIN clauses.
    table_alias_to_name: Dict[str, str] = {}
    for table in parsed.find_all(exp.Table):
        if _is_in_subquery(table, parsed):
            continue
        name = (table.name or "").lower()
        alias = (table.alias_or_name or name).lower()
        if name:
            table_alias_to_name[alias] = name

    # For each top-level projection, find the first Column reference and
    # resolve its table alias back to the real table name.
    out: Dict[str, str] = {}
    for projection in parsed.expressions:
        out_alias = (projection.alias_or_name or "").strip()
        if not out_alias:
            continue
        cols = list(projection.find_all(exp.Column))
        if not cols:
            continue
        col_table_alias = (cols[0].table or "").lower()
        if not col_table_alias:
            continue
        src_table = table_alias_to_name.get(col_table_alias)
        if src_table:
            out[out_alias] = src_table
    return out


# ── Folder lookup ────────────────────────────────────────────────────────

def _looks_like_image_field(field_name: str) -> bool:
    """Cheap pre-filter so we don't URL-ify arbitrary scalar columns."""
    if field_name in _EXPLICIT_FIELD_FOLDER:
        return True
    return bool(_IMAGE_SUFFIX_RE.match(field_name))


def _folder_for_field(
    field_name: str,
    alias_to_table: Mapping[str, str],
) -> Optional[str]:
    """Resolve an output field name to its S3 sub-folder.

    Order:
      1. SQL-parsed source table (when known and mapped).
      2. Explicit alias overrides (for product_image / product_images
         where the SQL source is an auxiliary table whose folder differs).
      3. Suffix-pattern fallback (`<table>_img` → table's folder).
    """
    # 1. SQL-parsed source — the deterministic path.
    src = alias_to_table.get(field_name)
    if src and src in _TABLE_FOLDER:
        return _TABLE_FOLDER[src]

    # 2. Explicit override (covers product_image / product_images, where
    #    the parsed source is `media` or `buy_sell_products` but the
    #    files live under a product folder).
    if field_name in _EXPLICIT_FIELD_FOLDER:
        return _EXPLICIT_FIELD_FOLDER[field_name]

    # 3. Suffix-pattern fallback when SQL parsing was unavailable.
    m = _TABLE_FIELD_RE.match(field_name)
    if m:
        return _TABLE_FOLDER.get(m.group("table"))

    return None


# ── Value-shape handling ─────────────────────────────────────────────────

def _build_url(base: str, folder: str, name: str) -> str:
    return f"{base.rstrip('/')}/{folder.strip('/')}/{name.lstrip('/')}"


def _resolve_one(name: Any, base: str, folder: str) -> Any:
    if not isinstance(name, str):
        return name
    n = name.strip()
    if not n:
        return name
    if "://" in n:           # already a URL — idempotent
        return n
    return _build_url(base, folder, n)


def _resolve_value(value: Any, base: str, folder: str) -> Any:
    if value is None:
        return value

    if isinstance(value, list):
        return [_resolve_one(item, base, folder) for item in value]

    if not isinstance(value, str):
        return value

    s = value.strip()
    if not s:
        return value

    # JSON shapes: `[...]` array, `"..."` quoted single string.
    if s[0] in ("[", '"'):
        try:
            parsed = json.loads(s)
        except (ValueError, TypeError):
            return value          # malformed JSON — leave untouched
        if isinstance(parsed, list):
            return [_resolve_one(item, base, folder) for item in parsed]
        if isinstance(parsed, str):
            return _resolve_one(parsed, base, folder)
        return value              # dict / number / bool — pass through

    # Plain image-name string.
    return _resolve_one(s, base, folder)


# ── Public entry point ───────────────────────────────────────────────────

def resolve_images_in_rows(
    rows: Iterable[Dict[str, Any]],
    sql: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Walk every row and rewrite image-bearing fields to S3 URLs in place.

    `sql` is the executed query — passed in so we can determine each output
    column's true source table regardless of the LLM's alias choice.  When
    omitted (or unparseable), we fall back to explicit alias and suffix-
    pattern matching only.

    Returns the same list (mutated) for ergonomic chaining.
    """
    base = (settings.IMAGE_BASE_URL or "").strip()
    if not base:
        logger.debug("IMAGE_BASE_URL not set; skipping URL resolution")
        return list(rows or [])

    alias_to_table = _build_alias_to_source_table(sql)

    out: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            out.append(row)
            continue
        for field in list(row.keys()):
            if not _looks_like_image_field(field):
                continue
            folder = _folder_for_field(field, alias_to_table)
            if folder is None:
                continue
            row[field] = _resolve_value(row[field], base, folder)
        out.append(row)
    return out
