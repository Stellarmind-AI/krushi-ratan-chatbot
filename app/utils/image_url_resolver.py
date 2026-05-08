"""Resolve raw image filenames returned from SQL queries into full S3 URLs.

The DB stores image *names* only (e.g. `tractor.jpg`).  The frontend renders
images, which requires a URL.  S3 layout is `<BASE>/<folder>/<filename>`
where the folder depends on which table the image originated from.

We disambiguate the folder by *field name* in the result row — the SQL
alias carries enough information to identify the source table.  For example:

    product_image    came from kshop_products via media     → kshopProducts/
    product_images   came from buy_sell_products.form_data  → buy_sell_products/
    category_img     came from kshop_categories.img         → k_shop_categories/
    category_image   came from buy_sell_categories.image    → buy_sell_categories/
    crop_img         came from sub_categories.img           → sub-categories/

In addition, the orchestrator's `_build_compiled_schemas` may auto-alias
joined-table image columns as `<table>_img|image|images` (e.g.
`sub_categories_img`, `kshop_categories_image`).  Those are resolved via
the table → folder map below.

Value-shape handling — image fields are not consistent across sources:

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

logger = get_logger("image_resolver")


# Field-name → S3 folder map for explicit aliases produced by hand-written
# example queries / the SQL prompt.  These are the canonical aliases the
# LLM is told to use.
_EXPLICIT_FIELD_FOLDER: Mapping[str, str] = {
    "product_image":   "kshopProducts",
    "product_images":  "buy_sell_products",
    "category_img":    "k_shop_categories",
    "category_image":  "buy_sell_categories",
    "crop_img":        "sub-categories",
}

# Table → S3 folder map used for auto-generated aliases of the shape
# `<table>_img | <table>_image | <table>_images` produced by the
# exploratory-SQL builder when it projects a joined table's image column.
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

# Match auto-aliases like `sub_categories_img`, `kshop_companies_image`.
_TABLE_FIELD_RE = re.compile(r"^(?P<table>[a-z][a-z0-9_]+?)_(?:img|image|images)$")


def _folder_for_field(field_name: str) -> Optional[str]:
    """Return the S3 sub-folder for a result field, or None if unknown."""
    if field_name in _EXPLICIT_FIELD_FOLDER:
        return _EXPLICIT_FIELD_FOLDER[field_name]
    m = _TABLE_FIELD_RE.match(field_name)
    if not m:
        return None
    return _TABLE_FOLDER.get(m.group("table"))


def _build_url(base: str, folder: str, name: str) -> str:
    return f"{base.rstrip('/')}/{folder.strip('/')}/{name.lstrip('/')}"


def _resolve_one(name: Any, base: str, folder: str) -> Any:
    """Convert a single image-name string to a URL. Pass through if unusable."""
    if not isinstance(name, str):
        return name
    n = name.strip()
    if not n:
        return name
    if "://" in n:           # already a URL — idempotent
        return n
    return _build_url(base, folder, n)


def _resolve_value(value: Any, base: str, folder: str) -> Any:
    """Resolve any of: None, str, list, JSON-array-string, JSON-quoted-string."""
    if value is None:
        return value

    if isinstance(value, list):
        return [_resolve_one(item, base, folder) for item in value]

    if not isinstance(value, str):
        return value

    s = value.strip()
    if not s:
        return value

    # JSON shapes: `[...]` (array), `"..."` (quoted single string).
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


def resolve_images_in_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Walk every row and rewrite image-bearing fields to S3 URLs in place.

    Returns the same list (mutated) for ergonomic chaining.  Fields whose
    folder we don't recognise are left untouched, as are values that don't
    look like image names.  This is the single transform site between SQL
    execution and any downstream consumer (LLM context, status filter,
    cache replay, frontend WebSocket payload).
    """
    base = (settings.IMAGE_BASE_URL or "").strip()
    if not base:
        # No base configured — return rows untouched. Logged once at
        # debug level so a missing setting is visible without spamming.
        logger.debug("IMAGE_BASE_URL not set; skipping URL resolution")
        return list(rows or [])

    out: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            out.append(row)
            continue
        for field in list(row.keys()):
            folder = _folder_for_field(field)
            if folder is None:
                continue
            row[field] = _resolve_value(row[field], base, folder)
        out.append(row)
    return out
