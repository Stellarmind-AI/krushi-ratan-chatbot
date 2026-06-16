"""
Stage 3 — Deterministic SQL Builder (NO LLM).

Compiles a resolved NLU frame into SQL by COMPOSITION, not templates: each
frame dimension (intent → tables/joins, resolved entities → id filters,
constraints → clauses, query_type → shape) maps to a SQL fragment, and the
fragments combine. This covers unlimited question phrasings because Stage 1
already collapsed them into a finite frame.

Everything the old LLM SQL prompt did is reproduced here, validated against the
live DB:
  • Joins + JOIN/LEFT-JOIN types  → from schema_generator.VERIFIED_FK_MAP
  • Soft delete (deleted_at)        → SOFT_DELETE_TABLES
  • Status: kshop status=1 in SQL; buy_sell/video/seed SELECT the status (+
    is_sold) column so the post-retrieval status_filter removes sold_out/draft;
    products & news have no status column.
  • Images: buy_sell form_data '$.Images' + bc.image; kshop mediables→media +
    kc.img; crop sc.img — exact aliases the image_url_resolver expects.
  • Filtering is ID-BASED (subcategory_id IN, category_id IN, y.city_id =) from
    resolved row ids; LIKE-on-name is a FALLBACK only for unresolved entities.
  • query_type: count → COUNT, list_all → DISTINCT names (no keyword filter),
    specific_search → full WHERE.
  • Constraints (price/date/sort/group_by) emitted ONLY when present in the
    frame — never invented.

The output is the same shape the executor expects: [{"table_name", "sql"}].
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.models.nlu_frame import NLUFrame
from app.services.agent.resolver import ResolvedFrame, ResolvedEntity
from app.core.logger import get_logger

logger = get_logger("sql_builder")

_LIMIT = 50
_LIST_LIMIT = 100


def _esc(s: str) -> str:
    """Escape a string literal for safe inlining (read-only, validated SELECT)."""
    return (s or "").replace("\\", "\\\\").replace("'", "''")


def _in_list(ids: List[int]) -> str:
    return ", ".join(str(int(i)) for i in ids)


# ── Location: resolved {table,id} → the right column per intent ──────────────
# For crop_price the geography is reached through yards; for news it is direct
# columns on the news row. Returns a single OR-group SQL string, or "".

def _loc_filter_via_yards(locations: List[ResolvedEntity]) -> str:
    """products→yards→{cities,talukas,states}. Resolved → id equality; the
    matched catalog table tells us the column. Unresolved → 3-way name LIKE."""
    clauses: List[str] = []
    for loc in locations:
        if loc.resolved and loc.ids:
            col = {
                "cities":  "y.city_id",
                "talukas": "y.taluka_id",
                "yards":   "p.yard_id",
                "states":  "c.state_id",
            }.get(loc.table)
            if col:
                clauses.append(f"{col} IN ({_in_list(loc.ids)})")
                continue
        # Fallback: unresolved location → name LIKE across city/taluka/yard.
        kw = _esc(loc.surface)
        clauses.append(
            f"(c.name LIKE '%{kw}%' OR t.name LIKE '%{kw}%' OR y.name LIKE '%{kw}%')"
        )
    return " AND ".join(f"({c})" for c in clauses) if clauses else ""


def _loc_filter_direct(locations: List[ResolvedEntity], alias: str) -> str:
    """news has direct state_id/city_id/taluka_id columns on <alias>."""
    clauses: List[str] = []
    for loc in locations:
        if loc.resolved and loc.ids:
            col = {
                "cities":  f"{alias}.city_id",
                "talukas": f"{alias}.taluka_id",
                "states":  f"{alias}.state_id",
            }.get(loc.table)
            if col:
                clauses.append(f"{col} IN ({_in_list(loc.ids)})")
                continue
            if loc.table == "yards":
                # news has no yard fk; fall back to taluka/city of that yard name
                kw = _esc(loc.surface)
                clauses.append(
                    f"({alias}.city_id IN (SELECT id FROM cities WHERE name LIKE '%{kw}%') "
                    f"OR {alias}.taluka_id IN (SELECT id FROM talukas WHERE name LIKE '%{kw}%'))"
                )
                continue
        kw = _esc(loc.surface)
        clauses.append(
            f"({alias}.city_id IN (SELECT id FROM cities WHERE name LIKE '%{kw}%') "
            f"OR {alias}.taluka_id IN (SELECT id FROM talukas WHERE name LIKE '%{kw}%') "
            f"OR {alias}.state_id IN (SELECT id FROM states WHERE name LIKE '%{kw}%'))"
        )
    return " AND ".join(clauses) if clauses else ""


def _entity_id_or_like(
    entities: List[ResolvedEntity], id_col: str, name_cols: List[str]
) -> str:
    """Build one OR-group: resolved entities → <id_col> IN (ids); unresolved →
    name LIKE across name_cols. Multiple entities are OR'd together."""
    ids: List[int] = []
    likes: List[str] = []
    for e in entities:
        if e.resolved and e.ids:
            ids.extend(e.ids)
        elif e.surface:
            kw = _esc(e.surface)
            likes.extend(f"{nc} LIKE '%{kw}%'" for nc in name_cols)
    parts: List[str] = []
    if ids:
        parts.append(f"{id_col} IN ({_in_list(sorted(set(ids)))})")
    parts.extend(likes)
    return "(" + " OR ".join(parts) + ")" if parts else ""


# ── Constraint clauses (only what the frame explicitly carries) ──────────────

def _price_constraints(c, min_col: str, max_col: str) -> List[str]:
    out = []
    if c.price_above is not None:
        out.append(f"{max_col} >= {float(c.price_above)}")
    if c.price_below is not None:
        out.append(f"{min_col} <= {float(c.price_below)}")
    return out


def _date_constraint(c, date_col: str) -> Optional[str]:
    if c.date == "today":
        return f"{date_col} = CURDATE()"
    if c.date == "this_week":
        return f"{date_col} >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)"
    if c.date == "this_month":
        return f"{date_col} >= DATE_FORMAT(CURDATE(), '%Y-%m-01')"
    return None  # "latest" handled by ORDER BY


# ── crop_price (products) ────────────────────────────────────────────────────

_CROP_JOINS = (
    "JOIN sub_categories sc ON p.subcategory_id = sc.id "
    "JOIN yards y ON p.yard_id = y.id "
    "LEFT JOIN cities c ON y.city_id = c.id "
    "LEFT JOIN talukas t ON y.taluka_id = t.id "
    "LEFT JOIN states s ON c.state_id = s.id "
    "LEFT JOIN weights w ON p.weight_id = w.id"
)
_CROP_SELECT = (
    "sc.name AS crop, sc.img AS crop_img, p.min_price, p.max_price, "
    "p.price_date, y.name AS yard, c.name AS city, t.name AS taluka, w.name AS weight"
)


def _build_crop_price(frame: NLUFrame, rf: ResolvedFrame) -> List[Dict]:
    where = ["p.deleted_at IS NULL"]

    if frame.query_type == "count":
        return _build_crop_count(frame, rf, where)

    if frame.query_type == "list_all":
        # Browse: distinct crop names, NO keyword filter. Optional location.
        loc = _loc_filter_via_yards(rf.locations)
        if loc:
            where.append(loc)
        sql = (f"SELECT DISTINCT sc.name AS crop, sc.img AS crop_img "
               f"FROM products p {_CROP_JOINS} WHERE {' AND '.join(where)} "
               f"ORDER BY sc.name ASC LIMIT {_LIST_LIMIT}")
        return [{"table_name": "products", "sql": sql}]

    # specific_search
    crop_f = _entity_id_or_like(rf.crops, "p.subcategory_id",
                                ["sc.name", "p.subcategory_name"])
    if crop_f:
        where.append(crop_f)
    # category-scoped (e.g. "cash crops"): sc.category_id IN (resolved cat ids)
    cat_f = _entity_id_or_like(rf.categories, "sc.category_id", ["sc.name"])
    if cat_f and rf.categories:
        where.append(cat_f)
    loc = _loc_filter_via_yards(rf.locations)
    if loc:
        where.append(loc)
    where += _price_constraints(frame.constraints, "p.min_price", "p.max_price")
    dc = _date_constraint(frame.constraints, "p.price_date")
    if dc:
        where.append(dc)

    order = _crop_order(frame.constraints)
    sql = (f"SELECT {_CROP_SELECT} FROM products p {_CROP_JOINS} "
           f"WHERE {' AND '.join(where)} {order} LIMIT {_LIMIT}")
    return [{"table_name": "products", "sql": sql}]


def _crop_order(c) -> str:
    if c.sort == "cheapest":
        return "ORDER BY p.min_price ASC"
    if c.sort == "most_expensive":
        return "ORDER BY p.max_price DESC"
    return "ORDER BY p.price_date DESC"  # latest/newest default


def _build_crop_count(frame: NLUFrame, rf: ResolvedFrame, where: List[str]) -> List[Dict]:
    target = frame.count_target or "crop"
    loc = _loc_filter_via_yards(rf.locations)
    # Geography counts hit their own table; crop/product count hits products.
    if target == "yard":
        w = ["y.deleted_at IS NULL"]
        if rf.locations:
            w.append(_loc_count_geo(rf.locations, "y"))
        return [{"table_name": "yards",
                 "sql": f"SELECT COUNT(DISTINCT y.id) AS count FROM yards y "
                        f"LEFT JOIN cities c ON y.city_id=c.id LEFT JOIN talukas t ON y.taluka_id=t.id "
                        f"WHERE {' AND '.join(x for x in w if x)}"}]
    if target in ("city", "taluka", "state"):
        return [_geo_count(target, rf.locations)]
    # crop / product → from products
    if loc:
        where.append(loc)
    col = "p.subcategory_id" if target == "crop" else "p.id"
    sql = (f"SELECT COUNT(DISTINCT {col}) AS count FROM products p {_CROP_JOINS} "
           f"WHERE {' AND '.join(where)}")
    return [{"table_name": "products", "sql": sql}]


def _loc_count_geo(locations: List[ResolvedEntity], alias: str) -> str:
    """Filter a geography-count by a parent location (e.g. talukas in Bhavnagar)."""
    clauses = []
    for loc in locations:
        if loc.resolved and loc.ids:
            if loc.table == "cities":
                clauses.append(f"{alias}.city_id IN ({_in_list(loc.ids)})")
            elif loc.table == "states":
                clauses.append(f"c.state_id IN ({_in_list(loc.ids)})")
            elif loc.table == "talukas":
                clauses.append(f"{alias}.taluka_id IN ({_in_list(loc.ids)})")
        else:
            kw = _esc(loc.surface)
            clauses.append(f"(c.name LIKE '%{kw}%')")
    return " AND ".join(clauses)


def _geo_count(target: str, locations: List[ResolvedEntity]) -> Dict:
    table = {"city": "cities", "taluka": "talukas", "state": "states"}[target]
    if target == "city":
        w = ["c.deleted_at IS NULL"]
        for loc in locations:
            if loc.resolved and loc.table == "states" and loc.ids:
                w.append(f"c.state_id IN ({_in_list(loc.ids)})")
            elif not loc.resolved and loc.surface:
                w.append(f"c.state_id IN (SELECT id FROM states WHERE name LIKE '%{_esc(loc.surface)}%')")
        return {"table_name": "cities",
                "sql": f"SELECT COUNT(*) AS count FROM cities c WHERE {' AND '.join(w)}"}
    if target == "taluka":
        w = ["t.deleted_at IS NULL"]
        for loc in locations:
            if loc.resolved and loc.table == "cities" and loc.ids:
                w.append(f"t.city_id IN ({_in_list(loc.ids)})")
            elif not loc.resolved and loc.surface:
                w.append(f"t.city_id IN (SELECT id FROM cities WHERE name LIKE '%{_esc(loc.surface)}%')")
        return {"table_name": "talukas",
                "sql": f"SELECT COUNT(*) AS count FROM talukas t WHERE {' AND '.join(w)}"}
    return {"table_name": "states",
            "sql": "SELECT COUNT(*) AS count FROM states s WHERE s.deleted_at IS NULL"}


# ── kshop (equipment_kshop / kshop_product) ─────────────────────────────────

_KSHOP_JOINS = (
    "JOIN kshop_companies kco ON kp.kshop_company_id = kco.id "
    "LEFT JOIN kshop_categories kc ON kp.kshop_category_id = kc.id AND kc.deleted_at IS NULL "
    "LEFT JOIN kshop_weights kw ON kp.kshop_weight_id = kw.id "
    "LEFT JOIN mediables mb ON mb.mediable_id = kp.id AND mb.mediable_type LIKE 'App%KshopProduct' "
    "LEFT JOIN media m ON m.id = mb.media_id"
)
_KSHOP_SELECT = (
    "kp.id, kp.name, kp.price, kp.discount_price, kp.status, "
    "kco.name AS company, kc.name AS category, kc.img AS category_img, "
    "CONCAT(m.filename, '.', m.extension) AS product_image"
)


def _build_kshop(frame: NLUFrame, rf: ResolvedFrame) -> List[Dict]:
    if frame.query_type == "count":
        target = frame.count_target or "product"
        if target == "company":
            return [{"table_name": "kshop_companies",
                     "sql": "SELECT COUNT(*) AS count FROM kshop_companies kco WHERE kco.deleted_at IS NULL"}]
        if target == "category":
            return [{"table_name": "kshop_categories",
                     "sql": "SELECT COUNT(*) AS count FROM kshop_categories kc WHERE kc.deleted_at IS NULL"}]
        return [{"table_name": "kshop_products",
                 "sql": "SELECT COUNT(*) AS count FROM kshop_products kp WHERE kp.deleted_at IS NULL AND kp.status = 1"}]

    where = ["kp.deleted_at IS NULL", "kp.status = 1"]
    if frame.query_type == "list_all":
        sql = (f"SELECT {_KSHOP_SELECT} FROM kshop_products kp {_KSHOP_JOINS} "
               f"WHERE {' AND '.join(where)} ORDER BY kp.updated_at DESC LIMIT {_LIST_LIMIT}")
        return [{"table_name": "kshop_products", "sql": sql}]

    cat_f = _entity_id_or_like(rf.equipment, "kp.kshop_category_id", ["kc.name", "kp.name"])
    if cat_f:
        where.append(cat_f)
    where += _price_constraints(frame.constraints, "kp.price", "kp.price")
    order = ("ORDER BY kp.price ASC" if frame.constraints.sort == "cheapest"
             else "ORDER BY kp.price DESC" if frame.constraints.sort == "most_expensive"
             else "ORDER BY kp.updated_at DESC")
    sql = (f"SELECT {_KSHOP_SELECT} FROM kshop_products kp {_KSHOP_JOINS} "
           f"WHERE {' AND '.join(where)} {order} LIMIT {_LIMIT}")
    return [{"table_name": "kshop_products", "sql": sql}]


# ── buy_sell (equipment_used / buy_sell_product) ────────────────────────────

_BUYSELL_JOINS = "LEFT JOIN buy_sell_categories bc ON bp.category_id = bc.id AND bc.deleted_at IS NULL"
# status + is_sold SELECTED so the post-retrieval status_filter can drop sold_out.
_BUYSELL_SELECT = (
    "bp.id, bp.product_name, bp.price, bp.quantity_available, bp.status, bp.is_sold, "
    "JSON_EXTRACT(bp.form_data, '$.Images') AS product_images, "
    "bc.name AS category, bc.image AS category_image"
)


def _build_buysell(frame: NLUFrame, rf: ResolvedFrame) -> List[Dict]:
    if frame.query_type == "count":
        target = frame.count_target or "product"
        if target == "category":
            return [{"table_name": "buy_sell_categories",
                     "sql": "SELECT COUNT(*) AS count FROM buy_sell_categories bc WHERE bc.deleted_at IS NULL"}]
        return [{"table_name": "buy_sell_products",
                 "sql": "SELECT COUNT(*) AS count FROM buy_sell_products bp WHERE bp.deleted_at IS NULL"}]

    where = ["bp.deleted_at IS NULL"]

    # Listing-id lookup ("owner of tractor 1778..."). The id is embedded in the
    # listing NAME ("ટ્રેક્ટર - 1778741216208"); product_code is unused (NULL).
    if frame.identifier:
        where.append(
            f"(bp.product_name LIKE '%{_esc(frame.identifier)}%' "
            f"OR bp.product_code = '{_esc(frame.identifier)}')"
        )
        sql = (f"SELECT {_BUYSELL_SELECT} FROM buy_sell_products bp {_BUYSELL_JOINS} "
               f"WHERE {' AND '.join(where)} LIMIT {_LIMIT}")
        return [{"table_name": "buy_sell_products", "sql": sql}]

    if frame.query_type == "list_all":
        sql = (f"SELECT {_BUYSELL_SELECT} FROM buy_sell_products bp {_BUYSELL_JOINS} "
               f"WHERE {' AND '.join(where)} ORDER BY bp.created_at DESC LIMIT {_LIST_LIMIT}")
        return [{"table_name": "buy_sell_products", "sql": sql}]

    # animals + used equipment both resolve to buy_sell_categories.
    subjects = rf.animals + rf.equipment
    cat_f = _entity_id_or_like(subjects, "bp.category_id", ["bc.name", "bp.product_name"])
    if cat_f:
        where.append(cat_f)
    where += _price_constraints(frame.constraints, "bp.price", "bp.price")
    order = ("ORDER BY bp.price ASC" if frame.constraints.sort == "cheapest"
             else "ORDER BY bp.price DESC" if frame.constraints.sort == "most_expensive"
             else "ORDER BY bp.created_at DESC")
    sql = (f"SELECT {_BUYSELL_SELECT} FROM buy_sell_products bp {_BUYSELL_JOINS} "
           f"WHERE {' AND '.join(where)} {order} LIMIT {_LIMIT}")
    return [{"table_name": "buy_sell_products", "sql": sql}]


# ── seed_info (seeds) ────────────────────────────────────────────────────────

def _build_seed(frame: NLUFrame, rf: ResolvedFrame) -> List[Dict]:
    if frame.query_type == "count":
        return [{"table_name": "seeds",
                 "sql": "SELECT COUNT(*) AS count FROM seeds s WHERE s.deleted_at IS NULL"}]
    where = ["s.deleted_at IS NULL"]
    sel = "s.id, s.name, s.status, sc.name AS subcategory, sc.img AS crop_img"
    joins = "LEFT JOIN sub_categories sc ON s.subcategory_id = sc.id"
    if frame.query_type != "list_all":
        crop_f = _entity_id_or_like(rf.crops, "s.subcategory_id", ["s.name", "sc.name"])
        if crop_f:
            where.append(crop_f)
    sql = (f"SELECT {sel} FROM seeds s {joins} WHERE {' AND '.join(where)} "
           f"ORDER BY s.name ASC LIMIT {_LIST_LIMIT}")
    return [{"table_name": "seeds", "sql": sql}]


# ── news (news) ──────────────────────────────────────────────────────────────

_NEWS_JOINS = (
    "LEFT JOIN states s ON n.state_id = s.id "
    "LEFT JOIN cities c ON n.city_id = c.id "
    "LEFT JOIN talukas t ON n.taluka_id = t.id"
)
_NEWS_SELECT = (
    "n.id, n.title, n.description, n.news_type, n.created_at, "
    "s.name AS state, c.name AS city, t.name AS taluka"
)


def _build_news(frame: NLUFrame, rf: ResolvedFrame) -> List[Dict]:
    if frame.query_type == "count":
        return [{"table_name": "news",
                 "sql": "SELECT COUNT(*) AS count FROM news n WHERE n.deleted_at IS NULL"}]
    where = ["n.deleted_at IS NULL"]
    # STRUCTURED filters (reliable) — always AND.
    # news.news_type is a STRING == news_types.name (resolved canonical).
    has_type = False
    for nt in rf.news_types:
        if nt.resolved:
            where.append(f"n.news_type = '{_esc(nt.canonical)}'")
            has_type = True
        elif nt.surface:
            where.append(f"n.news_type LIKE '%{_esc(nt.surface)}%'")
            has_type = True
    has_loc = False
    if frame.query_type != "list_all":
        loc = _loc_filter_direct(rf.locations, "n")
        if loc:
            where.append(loc)
            has_loc = bool(rf.locations)
    dc = _date_constraint(frame.constraints, "n.created_at")
    if dc:
        where.append(dc)
    # FREE-TEXT topic (fragile LIKE): AND it only when it is the SOLE content
    # signal. When a reliable structured filter (type/location) is present,
    # AND-ing a sparse free-text LIKE causes false negatives (over-filtering),
    # so we let the structured scope stand and the answer layer rank by topic.
    if frame.query_type != "list_all" and frame.news.topics and not (has_type or has_loc):
        ors = []
        for tp in frame.news.topics:
            kw = _esc(tp)
            ors.append(f"n.title LIKE '%{kw}%' OR n.description LIKE '%{kw}%'")
        where.append("(" + " OR ".join(ors) + ")")
    sql = (f"SELECT {_NEWS_SELECT} FROM news n {_NEWS_JOINS} "
           f"WHERE {' AND '.join(where)} ORDER BY n.created_at DESC LIMIT {_LIST_LIMIT}")
    return [{"table_name": "news", "sql": sql}]


# ── video (video_posts) ──────────────────────────────────────────────────────

def _build_video(frame: NLUFrame, rf: ResolvedFrame) -> List[Dict]:
    if frame.query_type == "count":
        return [{"table_name": "video_posts",
                 "sql": "SELECT COUNT(*) AS count FROM video_posts vp WHERE vp.deleted_at IS NULL"}]
    where = ["vp.deleted_at IS NULL"]
    sel = ("vp.id, vp.title, vp.description, vp.video_url, vp.thumbnail_url, "
           "vp.views_count, vp.status, vc.name AS category")
    joins = "LEFT JOIN video_categories vc ON vp.video_category_id = vc.id"
    if frame.query_type != "list_all":
        topics = frame.video.topics
        if topics:
            ors = []
            for tp in topics:
                kw = _esc(tp)
                ors.append(f"vp.title LIKE '%{kw}%' OR vp.description LIKE '%{kw}%'")
            where.append("(" + " OR ".join(ors) + ")")
    sql = (f"SELECT {sel} FROM video_posts vp {joins} WHERE {' AND '.join(where)} "
           f"ORDER BY vp.views_count DESC LIMIT {_LIST_LIMIT}")
    return [{"table_name": "video_posts", "sql": sql}]


# ── Dispatch ─────────────────────────────────────────────────────────────────

_BUILDERS = {
    "crop_price":       _build_crop_price,
    "equipment_kshop":  _build_kshop,
    "kshop_product":    _build_kshop,
    "equipment_used":   _build_buysell,
    "buy_sell_product": _build_buysell,
    "seed_info":        _build_seed,
    "news":             _build_news,
    "video":            _build_video,
}


def build_sql(frame: NLUFrame, resolved: ResolvedFrame) -> List[Dict]:
    """Compile a resolved frame to SQL. Returns [{"table_name","sql"}] or []."""
    builder = _BUILDERS.get(frame.intent)
    if not builder:
        logger.warning(f"sql_builder: no builder for intent={frame.intent}")
        return []
    try:
        queries = builder(frame, resolved)
        for q in queries:
            logger.info(f"🏗️  BUILT SQL | intent={frame.intent} qtype={frame.query_type} "
                        f"| {q['sql'][:160]}")
        return queries
    except Exception as e:
        logger.error_with_context(e, {"action": "build_sql", "intent": frame.intent})
        return []
