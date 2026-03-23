"""
Query Generator — Production-Level SQL Generation
Generates accurate MySQL JOIN queries from natural language questions.

Key design decisions:
1. Full FK relationship map from real tool files
2. Soft-delete filter (deleted_at IS NULL) always applied
3. Status filter applied for tables that have it
4. LEFT JOIN used where FK can be NULL/0 (e.g. kshop_category_id = 0)
5. Keyword extraction strips Gujarati/Hindi intent words before search
6. Each keyword searched independently with OR — never full-phrase match
7. Both Gujarati script and English/transliteration searched always
"""

import json
from typing import List, Dict, Any
from app.services.llm.manager import get_llm_manager
from app.models.chat_models import LLMMessage, QueryGenerationResponse
from app.utils.json_parser import json_parser
from app.core.logger import get_agent_logger
from app.core.config import settings

logger = get_agent_logger()


# ══════════════════════════════════════════════════════════════════════════════
# COMPLETE FK RELATIONSHIP MAP — built from actual tool JSON files
# Format: table -> list of {col, ref_table, ref_col, join_type}
# join_type = "JOIN" for strict FK, "LEFT JOIN" for nullable/optional FK
# ══════════════════════════════════════════════════════════════════════════════
FK_MAP: Dict[str, List[Dict]] = {
    "buy_sell_category_fields": [
        {"col": "category_id", "ref_table": "buy_sell_categories", "ref_col": "id", "join": "JOIN"},
    ],
    "buy_sell_category_steps": [
        {"col": "category_id", "ref_table": "buy_sell_categories", "ref_col": "id", "join": "JOIN"},
    ],
    "buy_sell_orders": [
        {"col": "product_id",  "ref_table": "buy_sell_products",   "ref_col": "id", "join": "JOIN"},
        {"col": "buyer_id",    "ref_table": "users",               "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "seller_id",   "ref_table": "users",               "ref_col": "id", "join": "LEFT JOIN"},
    ],
    "buy_sell_products": [
        {"col": "category_id", "ref_table": "buy_sell_categories", "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "seller_id",   "ref_table": "users",               "ref_col": "id", "join": "LEFT JOIN"},
    ],
    "cities": [
        {"col": "state_id",    "ref_table": "states",              "ref_col": "id", "join": "LEFT JOIN"},
    ],
    "company_orders": [
        {"col": "farmer_order_id",  "ref_table": "farmer_orders",  "ref_col": "id", "join": "JOIN"},
        {"col": "user_id",          "ref_table": "users",          "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "farmer_id",        "ref_table": "users",          "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "order_status_id",  "ref_table": "order_statuses", "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "subcategory_id",   "ref_table": "sub_categories", "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "weight_id",        "ref_table": "weights",        "ref_col": "id", "join": "LEFT JOIN"},
    ],
    "farmer_orders": [
        {"col": "user_id",         "ref_table": "users",           "ref_col": "id", "join": "LEFT JOIN"},
        # NOTE: company_id references kshop_companies (not 'companys' which is a typo in tool file)
        {"col": "company_id",      "ref_table": "kshop_companies", "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "order_status_id", "ref_table": "order_statuses",  "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "subcategory_id",  "ref_table": "sub_categories",  "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "weight_id",       "ref_table": "weights",         "ref_col": "id", "join": "LEFT JOIN"},
    ],
    "kshop_category_company": [
        {"col": "kshop_company_id",  "ref_table": "kshop_companies",  "ref_col": "id", "join": "JOIN"},
        {"col": "kshop_category_id", "ref_table": "kshop_categories", "ref_col": "id", "join": "JOIN"},
    ],
    "kshop_orders": [
        {"col": "kshop_product_id",  "ref_table": "kshop_products",   "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "kshop_category_id", "ref_table": "kshop_categories", "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "kshop_company_id",  "ref_table": "kshop_companies",  "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "order_status_id",   "ref_table": "order_statuses",   "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "user_id",           "ref_table": "users",            "ref_col": "id", "join": "LEFT JOIN"},
    ],
    "kshop_products": [
        # kshop_company_id is always set — use JOIN
        {"col": "kshop_company_id",  "ref_table": "kshop_companies",  "ref_col": "id", "join": "JOIN"},
        # kshop_category_id can be 0 (invalid FK) — MUST use LEFT JOIN
        {"col": "kshop_category_id", "ref_table": "kshop_categories", "ref_col": "id", "join": "LEFT JOIN"},
        # kshop_weight_id can be NULL — use LEFT JOIN
        {"col": "kshop_weight_id",   "ref_table": "kshop_weights",    "ref_col": "id", "join": "LEFT JOIN"},
    ],
    "news": [
        {"col": "state_id",  "ref_table": "states",  "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "city_id",   "ref_table": "cities",  "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "taluka_id", "ref_table": "talukas", "ref_col": "id", "join": "LEFT JOIN"},
    ],
    "products": [
        {"col": "subcategory_id", "ref_table": "sub_categories", "ref_col": "id", "join": "JOIN"},
        {"col": "weight_id",      "ref_table": "weights",         "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "yard_id",        "ref_table": "yards",           "ref_col": "id", "join": "JOIN"},
    ],
    "seeds": [
        {"col": "subcategory_id", "ref_table": "sub_categories", "ref_col": "id", "join": "LEFT JOIN"},
    ],
    # sub_categories.category_id -> categories.id (tool says 'categorys' which is typo)
    "sub_categories": [
        {"col": "category_id", "ref_table": "categories", "ref_col": "id", "join": "LEFT JOIN"},
    ],
    "talukas": [
        {"col": "city_id", "ref_table": "cities", "ref_col": "id", "join": "JOIN"},
    ],
    "user_products": [
        {"col": "subcategory_id", "ref_table": "sub_categories", "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "user_id",        "ref_table": "users",           "ref_col": "id", "join": "JOIN"},
    ],
    "user_subcategories": [
        {"col": "subcategory_id", "ref_table": "sub_categories", "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "user_id",        "ref_table": "users",           "ref_col": "id", "join": "JOIN"},
    ],
    "user_talukas": [
        {"col": "user_id",   "ref_table": "users",   "ref_col": "id", "join": "JOIN"},
        {"col": "taluka_id", "ref_table": "talukas", "ref_col": "id", "join": "JOIN"},
    ],
    "user_video_categories": [
        {"col": "user_id",           "ref_table": "users",            "ref_col": "id", "join": "JOIN"},
        {"col": "video_category_id", "ref_table": "video_categories", "ref_col": "id", "join": "JOIN"},
    ],
    "users": [
        {"col": "state_id", "ref_table": "states", "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "city_id",  "ref_table": "cities", "ref_col": "id", "join": "LEFT JOIN"},
    ],
    "video_comment_likes": [
        # tool says 'comments.id' — actual table is video_comments
        {"col": "comment_id", "ref_table": "video_comments", "ref_col": "id", "join": "JOIN"},
        {"col": "user_id",    "ref_table": "users",           "ref_col": "id", "join": "JOIN"},
    ],
    "video_comments": [
        {"col": "video_post_id",     "ref_table": "video_posts",    "ref_col": "id", "join": "JOIN"},
        {"col": "user_id",           "ref_table": "users",          "ref_col": "id", "join": "JOIN"},
        {"col": "parent_comment_id", "ref_table": "video_comments", "ref_col": "id", "join": "LEFT JOIN"},
    ],
    "video_likes": [
        {"col": "video_post_id", "ref_table": "video_posts", "ref_col": "id", "join": "JOIN"},
        {"col": "user_id",       "ref_table": "users",        "ref_col": "id", "join": "JOIN"},
    ],
    "video_posts": [
        {"col": "user_id",           "ref_table": "users",            "ref_col": "id", "join": "JOIN"},
        {"col": "video_category_id", "ref_table": "video_categories", "ref_col": "id", "join": "LEFT JOIN"},
    ],
    "video_saves": [
        {"col": "video_post_id", "ref_table": "video_posts", "ref_col": "id", "join": "JOIN"},
        {"col": "user_id",       "ref_table": "users",        "ref_col": "id", "join": "JOIN"},
    ],
    "video_shares": [
        {"col": "video_post_id", "ref_table": "video_posts", "ref_col": "id", "join": "JOIN"},
        {"col": "user_id",       "ref_table": "users",        "ref_col": "id", "join": "JOIN"},
    ],
    "video_views": [
        {"col": "video_post_id", "ref_table": "video_posts", "ref_col": "id", "join": "JOIN"},
        {"col": "user_id",       "ref_table": "users",        "ref_col": "id", "join": "JOIN"},
    ],
    "yards": [
        {"col": "state_id",  "ref_table": "states",  "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "city_id",   "ref_table": "cities",  "ref_col": "id", "join": "LEFT JOIN"},
        {"col": "taluka_id", "ref_table": "talukas", "ref_col": "id", "join": "LEFT JOIN"},
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# TABLES WITH SOFT DELETE — always add WHERE deleted_at IS NULL
# ══════════════════════════════════════════════════════════════════════════════
SOFT_DELETE_TABLES = {
    "buy_sell_categories", "buy_sell_category_fields", "buy_sell_category_steps",
    "buy_sell_orders", "buy_sell_products", "categories", "cities", "company_orders",
    "farmer_orders", "kshop_categories", "kshop_companies", "kshop_orders",
    "kshop_products", "kshop_weights", "navigation_flow", "news", "news_types",
    "products", "seeds", "states", "sub_categories", "talukas", "user_otps",
    "user_products", "user_subcategories", "user_talukas", "user_video_categories",
    "users", "video_categories", "video_comments", "video_posts", "weights", "yards",
}

# Tables where status=1 means active (add to WHERE for product searches)
STATUS_FILTERED_TABLES = {
    "kshop_products": 1,
    "buy_sell_products": "active",  # status='active'
    "kshop_categories": 1,
    "kshop_companies": 1,
    "sub_categories": 1,
    "video_posts": 1,
    "users": 1,
}

# ══════════════════════════════════════════════════════════════════════════════
# GUJARATI INTENT WORDS TO STRIP — these are not product keywords
# ══════════════════════════════════════════════════════════════════════════════
INTENT_WORDS = {
    # Gujarati intent words
    "mare", "maro", "mari", "mara", "mane", "maja",
    "karvu", "karvu", "karvani", "karvo", "karu", "karvu", "karshu",
    "che", "chhe", "hatu", "hati", "hathu",
    "joiyu", "joiye", "jovu", "joi",
    "levu", "levo", "levi", "lidhu",
    "purchase", "kharidi", "kharido", "vekhat",
    "from", "thi", "ma", "no", "ni", "na", "nu", "ne",
    "kshop", "buysell", "buy", "sell",
    "info", "mahiti", "jankarij", "jankari", "vishhe", "vishe",
    "apo", "apso", "batao", "batavo",
    "please", "pls",
}


class QueryGenerator:
    """Production-level SQL query generator with full relationship awareness.
    
    Uses OpenAI (GPT-4o) for SQL generation if configured — better accuracy
    on complex JOINs and mixed Gujarati/English queries than Groq.
    Falls back to Groq automatically if OpenAI is not configured.
    """

    def __init__(self):
        self.llm_manager = get_llm_manager()
        # Use OpenAI for SQL generation if available (better accuracy)
        # Falls back to Groq if OpenAI key not configured
        self._sql_provider = self._resolve_sql_provider()

    def _resolve_sql_provider(self) -> str:
        """
        Determine which LLM provider to use for SQL generation.
        
        Priority:
          1. SQL_PROVIDER env var: "openai" | "groq" | "auto"
          2. "auto" = use OpenAI if key configured, else Groq
        
        Set SQL_PROVIDER=openai in .env to force OpenAI for SQL.
        Set SQL_PROVIDER=groq  in .env to force Groq for SQL.
        """
        available = self.llm_manager.get_available_providers()
        config_choice = getattr(settings, "SQL_PROVIDER", "auto").lower().strip()

        if config_choice == "openai":
            if "openai" in available:
                logger.info("🧠 SQL provider: OpenAI (forced by SQL_PROVIDER=openai)")
                return "openai"
            else:
                logger.warning("⚠️  SQL_PROVIDER=openai but OpenAI not configured — falling back to Groq")
                return "groq"

        if config_choice == "groq":
            logger.info("🧠 SQL provider: Groq (forced by SQL_PROVIDER=groq)")
            return "groq"

        # "auto" — prefer OpenAI if available
        if "openai" in available:
            logger.info("🧠 SQL provider: OpenAI GPT-4o (auto — better SQL accuracy)")
            return "openai"

        logger.info("🧠 SQL provider: Groq (auto — OpenAI not configured)")
        return "groq"

    async def generate_queries(
        self,
        user_query: str,
        tool_schemas: List[Dict[str, Any]]
    ) -> QueryGenerationResponse:
        """Generate precise SQL queries for the selected tools."""
        try:
            logger.info("📊 QUERY GENERATION STARTED",
                        query=user_query[:100],
                        tool_count=len(tool_schemas))

            system_prompt = self._build_system_prompt(tool_schemas)
            user_message  = self._build_user_message(user_query)

            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user",   content=user_message),
            ]

            response = await self.llm_manager.generate(
                messages=messages,
                temperature=0.0,   # Zero — deterministic SQL
                max_tokens=3000,
                # Use OpenAI if available — significantly better SQL accuracy
                # especially for complex JOINs and Gujarati/English mixed queries
                provider_name=self._sql_provider,
            )

            queries = self._parse_query_response(response.content, tool_schemas)

            for q in queries:
                logger.sql_generation(query=q["sql"], table=q["table_name"])

            return QueryGenerationResponse(queries=queries)

        except Exception as e:
            logger.error_with_context(e, {"action": "generate_queries",
                                          "query": user_query[:200]})
            raise

    # ── System Prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self, tool_schemas: List[Dict[str, Any]]) -> str:
        schema_block = self._build_schema_block(tool_schemas)
        fk_block     = self._build_fk_block(tool_schemas)

        return f"""You are a production MySQL query generator for "Krushi Ratn" — a Gujarati agricultural marketplace app.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELECTED TABLE SCHEMAS (columns + types)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{schema_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOREIGN KEY RELATIONSHIPS (exact JOIN syntax to use)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{fk_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES — FOLLOW EXACTLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — KEYWORD EXTRACTION (most important rule):
The user query contains INTENT WORDS that are NOT product names.
Strip these before searching: mare, maro, mane, karvu, karu, che, chhe, purchase,
kharidi, levu, levo, joiyu, from, kshop, info, mahiti, please, apo, batao,
thi, ma, no, ni, na, nu, ne, vishe, vishhe

EXAMPLE:
  User: "mare balwan power weeder purchase karvu che from kshop"
  Product keywords = ["balwan", "power", "weeder"] ← ONLY these for search
  Intent words stripped = ["mare", "purchase", "karvu", "che", "from", "kshop"]

RULE 2 — SEARCH EACH KEYWORD INDEPENDENTLY WITH OR:
WRONG: WHERE name LIKE '%balwan power weeder%'   ← full phrase, too strict
RIGHT: WHERE (name LIKE '%balwan%' OR name LIKE '%બલવાન%')
         AND (name LIKE '%weeder%' OR name LIKE '%વીડર%' OR name LIKE '%power%' OR name LIKE '%પાવર%')

For 1-2 keywords use OR between them all.
For 3+ keywords: first keyword as AND, rest as OR group.

RULE 3 — ALWAYS SEARCH BOTH SCRIPTS:
Database has text in GUJARATI SCRIPT and English. Always search both:
  balwan  ↔  બલવાન
  power   ↔  પાવર
  weeder  ↔  વીડર
  thresher ↔ થ્રેશર
  pump    ↔  પંપ
  tractor ↔  ટ્રેક્ટર
  machine ↔  મશીન

RULE 4 — SOFT DELETE FILTER:
Tables with deleted_at column MUST always include: AND main_table.deleted_at IS NULL
These tables have deleted_at: kshop_products, buy_sell_products, products, users,
kshop_companies, kshop_categories, sub_categories, yards, cities, news, video_posts

RULE 5 — ACTIVE STATUS FILTER:
For kshop_products: add AND kp.status = 1
For buy_sell_products: add AND bp.status = 'active'
For video_posts: add AND vp.status = 1

RULE 6 — USE LEFT JOIN FOR OPTIONAL FKs:
kshop_products.kshop_category_id can be 0 (invalid) → MUST use LEFT JOIN kshop_categories
kshop_products.kshop_weight_id can be NULL → MUST use LEFT JOIN kshop_weights
buy_sell_products.category_id can be NULL → MUST use LEFT JOIN buy_sell_categories
Use JOIN only when FK is guaranteed to exist (kshop_products.kshop_company_id)

RULE 7 — ALWAYS RESOLVE IDs TO NAMES VIA JOIN:
NEVER return raw IDs to user. Always JOIN to get human-readable names:
  kshop_company_id → JOIN kshop_companies to get company name
  kshop_category_id → LEFT JOIN kshop_categories to get category name

RULE 8 — ONE QUERY ONLY, NO FALLBACK:
Generate exactly ONE precise query per table. NO broad SELECT * without WHERE.
If you cannot form a meaningful WHERE clause, return empty array [].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE QUERIES FOR COMMON PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

-- KShop product search (e.g. "balwan power weeder from kshop"):
SELECT kp.id, kp.name, kp.price, kp.discount_price, kp.description,
       kp.weight_value, kco.name AS company_name,
       COALESCE(kc.name, 'N/A') AS category_name,
       COALESCE(kw.display_name, '') AS weight_unit
FROM kshop_products kp
JOIN kshop_companies kco ON kp.kshop_company_id = kco.id
LEFT JOIN kshop_categories kc ON kp.kshop_category_id = kc.id AND kc.deleted_at IS NULL
LEFT JOIN kshop_weights kw ON kp.kshop_weight_id = kw.id
WHERE kp.deleted_at IS NULL AND kp.status = 1
AND (kp.name LIKE '%balwan%' OR kp.name LIKE '%બલવાન%')
AND (kp.name LIKE '%weeder%' OR kp.name LIKE '%વીડર%'
     OR kp.name LIKE '%power%' OR kp.name LIKE '%પાવર%')
ORDER BY kp.updated_at DESC
LIMIT 50

-- Crop price in a city (e.g. "kapas price in bhavnagar"):
SELECT sc.name AS crop, p.min_price, p.max_price, p.price_date,
       y.name AS yard, c.name AS city, w.display_name AS unit
FROM products p
JOIN sub_categories sc ON p.subcategory_id = sc.id
JOIN yards y ON p.yard_id = y.id
JOIN cities c ON y.city_id = c.id
LEFT JOIN weights w ON p.weight_id = w.id
WHERE p.deleted_at IS NULL
AND (c.name LIKE '%Bhavnagar%' OR c.name LIKE '%ભાવનગર%'
     OR y.name LIKE '%Bhavnagar%' OR y.name LIKE '%ભાવનગર%')
AND (sc.name LIKE '%kapas%' OR sc.name LIKE '%કપાસ%'
     OR p.subcategory_name LIKE '%kapas%' OR p.subcategory_name LIKE '%કપાસ%')
ORDER BY p.price_date DESC LIMIT 20

-- Buy/sell product search (e.g. "tractor for sale"):
SELECT bp.id, bp.product_name, bp.price, bp.quantity_available,
       bp.description, bp.status, bc.name AS category,
       u.name AS seller_name, u.mobile_number AS seller_mobile
FROM buy_sell_products bp
LEFT JOIN buy_sell_categories bc ON bp.category_id = bc.id AND bc.deleted_at IS NULL
LEFT JOIN users u ON bp.seller_id = u.id
WHERE bp.deleted_at IS NULL AND bp.status = 'active'
AND (bp.product_name LIKE '%tractor%' OR bp.product_name LIKE '%ટ્રેક્ટર%')
ORDER BY bp.created_at DESC LIMIT 50

-- Video search (e.g. "farming video"):
SELECT vp.id, vp.title, vp.description, vp.video_url, vp.views_count,
       vp.likes_count, u.name AS creator, vc.name AS category
FROM video_posts vp
JOIN users u ON vp.user_id = u.id
LEFT JOIN video_categories vc ON vp.video_category_id = vc.id
WHERE vp.deleted_at IS NULL AND vp.status = 1
AND (vp.title LIKE '%farming%' OR vp.title LIKE '%ખेती%'
     OR vp.description LIKE '%farming%')
ORDER BY vp.views_count DESC LIMIT 20

-- News by location (e.g. "news in surat"):
SELECT n.title, n.description, n.tags, n.created_at,
       c.name AS city, s.name AS state
FROM news n
LEFT JOIN cities c ON n.city_id = c.id
LEFT JOIN states s ON n.state_id = s.id
WHERE n.deleted_at IS NULL
AND (c.name LIKE '%Surat%' OR c.name LIKE '%સુરત%')
ORDER BY n.created_at DESC LIMIT 20

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a valid JSON array. No markdown, no explanation, nothing else.
[
  {{
    "table_name": "primary_table_name",
    "sql": "SELECT ... FROM ... JOIN ... WHERE ..."
  }}
]
"""

    def _build_schema_block(self, tool_schemas: List[Dict[str, Any]]) -> str:
        """Build full schema info for selected tables + all JOIN targets."""
        # Collect all tables needed (selected + their FK targets)
        selected = {s["table_name"] for s in tool_schemas}
        fk_targets = set()
        for tname in selected:
            for fk in FK_MAP.get(tname, []):
                fk_targets.add(fk["ref_table"])

        lines = []
        # Primary tables (full schema)
        for schema in tool_schemas:
            tname = schema["table_name"]
            soft  = "YES — always add WHERE deleted_at IS NULL" if tname in SOFT_DELETE_TABLES else "NO"
            status_note = ""
            if tname in STATUS_FILTERED_TABLES:
                v = STATUS_FILTERED_TABLES[tname]
                status_note = f"  ⚡ Active filter: WHERE status = {repr(v)}\n"

            lines.append(f"TABLE: {tname}  (soft-delete: {soft})")
            lines.append(status_note if status_note else "")
            for c in schema.get("columns", []):
                nullable = "NULL" if c.get("nullable") else "NOT NULL"
                lines.append(f"  {c['name']}  {c['type']}  {nullable}")
            lines.append("")

        # FK target tables (just names so LLM knows they exist for JOINs)
        extra = fk_targets - selected
        if extra:
            lines.append("JOIN TARGET TABLES (available for JOINs, not primary query tables):")
            for t in sorted(extra):
                lines.append(f"  {t}")
            lines.append("")

        return "\n".join(lines)

    def _build_fk_block(self, tool_schemas: List[Dict[str, Any]]) -> str:
        """Build FK relationship block showing exact JOIN syntax."""
        selected = {s["table_name"] for s in tool_schemas}
        lines = []
        for tname in sorted(selected):
            fks = FK_MAP.get(tname, [])
            if fks:
                lines.append(f"{tname}:")
                for fk in fks:
                    lines.append(
                        f"  {fk['join']} {fk['ref_table']} ON {tname}.{fk['col']} = {fk['ref_table']}.{fk['ref_col']}"
                    )
                lines.append("")
        return "\n".join(lines) if lines else "No relationships for selected tables."

    def _build_user_message(self, user_query: str) -> str:
        return f"""User question: "{user_query}"

Step 1: Identify PRODUCT/TOPIC keywords (strip intent words like mare/karvu/purchase/che/from/kshop)
Step 2: Generate precise SQL with correct JOINs, WHERE filters, soft-delete, and status filters
Step 3: Return ONLY the JSON array"""

    # ── Response Parsing ──────────────────────────────────────────────────────

    def _parse_query_response(
        self,
        response_content: str,
        tool_schemas: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Parse LLM response and validate SQL queries."""
        try:
            queries = json_parser.extract_queries_from_text(response_content)

            if not queries:
                parsed = json_parser.safe_parse(response_content, default=[], expected_type=list)
                if parsed and isinstance(parsed, list):
                    queries = parsed

            import re
            validated = []
            for q in queries:
                if not isinstance(q, dict) or "sql" not in q:
                    continue

                sql = q["sql"].strip()

                # Skip broad fallback queries — SELECT * without WHERE
                if re.match(r"SELECT\s+\*\s+FROM\s+\w+\s*(?:LIMIT|$)", sql, re.IGNORECASE):
                    logger.info(f"⏭️ Skipping broad fallback query: {sql[:80]}")
                    continue

                # Must have a WHERE clause to be considered specific
                if "WHERE" not in sql.upper():
                    logger.info(f"⏭️ Skipping query without WHERE: {sql[:80]}")
                    continue

                if "table_name" not in q:
                    m = re.search(r"FROM\s+([a-zA-Z0-9_]+)", sql, re.IGNORECASE)
                    q["table_name"] = m.group(1) if m else "unknown"

                validated.append(q)

            return validated

        except Exception as e:
            logger.error_with_context(e, {
                "action": "parse_query_response",
                "response": response_content[:200],
            })
            return []


# Global instance
query_generator = QueryGenerator()


def get_query_generator() -> QueryGenerator:
    return query_generator