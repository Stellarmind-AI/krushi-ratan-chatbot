"""
Schema Generator Utility
Generates condensed schema and individual tool files from full_schema.json.

KEY FIX: Uses VERIFIED_FK_MAP (hardcoded, correct) instead of naive column-name inference.
This ensures regenerated tool files always have correct relationships and JOIN types.
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from app.core.logger import get_logger
from app.utils.privacy_policy import PrivacyPolicy, get_privacy_policy

logger = get_logger("schema_generator")


# ══════════════════════════════════════════════════════════════════════════════
# VERIFIED FK RELATIONSHIP MAP
# Built by reading full_schema.json + cross-checking actual DB.
# join_type: JOIN = FK always set, LEFT JOIN = FK can be NULL or 0
#
# CORRECTIONS vs original naive inference:
#   farmer_orders.company_id    → kshop_companies     (was 'companys')
#   sub_categories.category_id  → categories          (was 'categorys')
#   video_comment_likes.comment_id → video_comments   (was 'comments')
#   video_comments.parent_comment_id → video_comments (was 'parent_comments', self-ref)
#   buy_sell_category_fields.step_id → buy_sell_category_steps (was 'steps')
#   navigation_flow.step_id → buy_sell_category_steps (was 'steps')
#   navigation_flow.category_id → buy_sell_categories (was 'categorys')
#   buy_sell_orders.product_id  → buy_sell_products   (was 'products')
#   kshop_products.kshop_category_id → LEFT JOIN      (can be 0 in real data)
# ══════════════════════════════════════════════════════════════════════════════
VERIFIED_FK_MAP: Dict[str, List[Dict]] = {
    "buy_sell_category_fields": [
        {"column": "category_id", "references": "buy_sell_categories.id",    "join_type": "JOIN",      "description": "Category this field belongs to"},
        {"column": "step_id",     "references": "buy_sell_category_steps.id","join_type": "LEFT JOIN", "description": "Step this field belongs to"},
    ],
    "buy_sell_category_steps": [
        {"column": "category_id", "references": "buy_sell_categories.id", "join_type": "JOIN", "description": "Category this step belongs to"},
    ],
    "buy_sell_orders": [
        {"column": "product_id", "references": "buy_sell_products.id", "join_type": "JOIN",      "description": "Product being ordered"},
    ],
    "buy_sell_products": [
        {"column": "category_id", "references": "buy_sell_categories.id", "join_type": "LEFT JOIN", "description": "Product category"},
    ],
    "cities": [
        {"column": "state_id", "references": "states.id", "join_type": "LEFT JOIN", "description": "State this city belongs to"},
    ],
    "company_orders": [
        {"column": "farmer_order_id", "references": "farmer_orders.id",  "join_type": "JOIN",      "description": "Related farmer order"},
        {"column": "order_status_id", "references": "order_statuses.id", "join_type": "LEFT JOIN", "description": "Current order status"},
        {"column": "subcategory_id",  "references": "sub_categories.id", "join_type": "LEFT JOIN", "description": "Product subcategory"},
        {"column": "weight_id",       "references": "weights.id",        "join_type": "LEFT JOIN", "description": "Weight unit"},
    ],
    "farmer_orders": [
        {"column": "company_id",      "references": "kshop_companies.id", "join_type": "LEFT JOIN", "description": "Company receiving the order"},
        {"column": "order_status_id", "references": "order_statuses.id",  "join_type": "LEFT JOIN", "description": "Order status"},
        {"column": "subcategory_id",  "references": "sub_categories.id",  "join_type": "LEFT JOIN", "description": "Product subcategory"},
        {"column": "weight_id",       "references": "weights.id",         "join_type": "LEFT JOIN", "description": "Weight unit"},
    ],
    "kshop_category_company": [
        {"column": "kshop_company_id",  "references": "kshop_companies.id",  "join_type": "JOIN", "description": "K-Shop company"},
        {"column": "kshop_category_id", "references": "kshop_categories.id", "join_type": "JOIN", "description": "K-Shop category"},
    ],
    "kshop_orders": [
        {"column": "kshop_product_id",  "references": "kshop_products.id",   "join_type": "LEFT JOIN", "description": "K-Shop product ordered"},
        {"column": "kshop_category_id", "references": "kshop_categories.id", "join_type": "LEFT JOIN", "description": "Product category"},
        {"column": "kshop_company_id",  "references": "kshop_companies.id",  "join_type": "LEFT JOIN", "description": "Company supplying product"},
        {"column": "order_status_id",   "references": "order_statuses.id",   "join_type": "LEFT JOIN", "description": "Order status"},
    ],
    "kshop_products": [
        {"column": "kshop_company_id",  "references": "kshop_companies.id",  "join_type": "JOIN",      "description": "Company that makes/sells this product (always set)"},
        {"column": "kshop_category_id", "references": "kshop_categories.id", "join_type": "LEFT JOIN", "description": "Product category — CAN BE 0 (uncategorized), MUST use LEFT JOIN"},
        {"column": "kshop_weight_id",   "references": "kshop_weights.id",    "join_type": "LEFT JOIN", "description": "Weight unit (nullable)"},
    ],
    "navigation_flow": [
        {"column": "category_id", "references": "buy_sell_categories.id",    "join_type": "LEFT JOIN", "description": "Category for navigation"},
        {"column": "step_id",     "references": "buy_sell_category_steps.id","join_type": "LEFT JOIN", "description": "Step in navigation flow"},
    ],
    "news": [
        {"column": "state_id",  "references": "states.id",  "join_type": "LEFT JOIN", "description": "State this news is about"},
        {"column": "city_id",   "references": "cities.id",  "join_type": "LEFT JOIN", "description": "City this news is about"},
        {"column": "taluka_id", "references": "talukas.id", "join_type": "LEFT JOIN", "description": "Taluka this news is about"},
    ],
    "products": [
        {"column": "subcategory_id", "references": "sub_categories.id", "join_type": "JOIN",      "description": "Crop/product type (e.g. Kapas, Wheat, Rice)"},
        {"column": "weight_id",      "references": "weights.id",        "join_type": "LEFT JOIN", "description": "Weight unit (kg, quintal)"},
        {"column": "yard_id",        "references": "yards.id",          "join_type": "JOIN",      "description": "Market yard where price is recorded"},
    ],
    "seeds": [
        {"column": "subcategory_id", "references": "sub_categories.id", "join_type": "LEFT JOIN", "description": "Seed subcategory"},
    ],
    "sub_categories": [
        {"column": "category_id", "references": "categories.id", "join_type": "LEFT JOIN", "description": "Parent category"},
    ],
    "talukas": [
        {"column": "city_id", "references": "cities.id", "join_type": "JOIN", "description": "City this taluka belongs to"},
    ],
    "user_products": [
        {"column": "subcategory_id", "references": "sub_categories.id", "join_type": "LEFT JOIN", "description": "Product subcategory"},
    ],
    "user_subcategories": [
        {"column": "subcategory_id", "references": "sub_categories.id", "join_type": "LEFT JOIN", "description": "Preferred subcategory"},
    ],
    "user_video_categories": [
        {"column": "video_category_id", "references": "video_categories.id", "join_type": "JOIN", "description": "Preferred video category"},
    ],
    "video_comment_likes": [
        {"column": "comment_id", "references": "video_comments.id", "join_type": "JOIN", "description": "Comment that was liked"},
    ],
    "video_comments": [
        {"column": "video_post_id",     "references": "video_posts.id",    "join_type": "JOIN",      "description": "Video being commented on"},
        {"column": "parent_comment_id", "references": "video_comments.id", "join_type": "LEFT JOIN", "description": "Parent comment if reply (self-reference)"},
    ],
    "video_likes": [
        {"column": "video_post_id", "references": "video_posts.id", "join_type": "JOIN", "description": "Video that was liked"},
    ],
    "video_posts": [
        {"column": "video_category_id", "references": "video_categories.id", "join_type": "LEFT JOIN", "description": "Video category"},
    ],
    "video_saves": [
        {"column": "video_post_id", "references": "video_posts.id", "join_type": "JOIN", "description": "Saved video"},
    ],
    "video_shares": [
        {"column": "video_post_id", "references": "video_posts.id", "join_type": "JOIN", "description": "Shared video"},
    ],
    "video_views": [
        {"column": "video_post_id", "references": "video_posts.id", "join_type": "JOIN", "description": "Viewed video"},
    ],
    "yards": [
        {"column": "state_id",  "references": "states.id",  "join_type": "LEFT JOIN", "description": "State of this yard"},
        {"column": "city_id",   "references": "cities.id",  "join_type": "LEFT JOIN", "description": "City of this yard"},
        {"column": "taluka_id", "references": "talukas.id", "join_type": "LEFT JOIN", "description": "Taluka of this yard"},
    ],
}

SOFT_DELETE_TABLES = {
    "buy_sell_categories","buy_sell_category_fields","buy_sell_category_steps",
    "buy_sell_orders","buy_sell_products","categories","cities","company_orders",
    "farmer_orders","kshop_categories","kshop_companies","kshop_orders",
    "kshop_products","kshop_weights","navigation_flow","news","news_types",
    "products","seeds","states","sub_categories","talukas","user_otps",
    "user_products","user_subcategories","user_talukas","user_video_categories",
    "users","video_categories","video_comments","video_posts","weights","yards",
}

# kshop_products is the ONLY table where the SQL layer adds a status = 1 filter.
# All other tables' status handling is done by the post-retrieval status_filter
# layer — NOT in SQL. Adding status conditions in SQL for other tables silently
# drops rows with valid non-'active' states (e.g. buy_sell 'sold_out') that the
# user may legitimately want to see. This aligns with orchestrator Rule #12.
STATUS_NOTES = {
    "kshop_products": "Active only: WHERE kshop_products.status = 1 AND kshop_products.deleted_at IS NULL",
}

TABLE_CONTEXTS = {
    "buy_sell_categories":     "Categories for buy/sell marketplace (animals, equipment, crops). Referenced as category_id in buy_sell_products",
    "buy_sell_products":       "Marketplace product listings by farmers — product_name, price, quantity, status='active'/'sold_out'.",
    "buy_sell_orders":         "Purchase transactions in buy/sell marketplace — buyer_id and seller_id both reference users",
    "buy_sell_category_fields":"Custom form fields per buy/sell product category",
    "buy_sell_category_steps": "Multi-step form wizard steps for buy/sell product listing",
    "categories":              "Main product categories — referenced by sub_categories",
    "sub_categories":          "Sub-categories under main categories — referenced as subcategory_id in products, seeds",
    "products":                "Crop/commodity market prices — subcategory_id (crop type), yard_id (market), min_price, max_price, price_date",
    "seeds":                   "Seed products with subcategory_id and variety info",
    "user_products":           "Products owned/listed by users with price range and subcategory",
    "user_subcategories":      "User preferences for product subcategories",
    "weights":                 "Weight measurement units (kg, quintal, ton) — referenced as weight_id",
    "yards":                   "Market yard locations — name, city_id, state_id, taluka_id. Referenced as yard_id in products",
    "kshop_companies":         "Companies selling in K-Shop — referenced as kshop_company_id",
    "kshop_categories":        "Product categories in K-Shop — referenced as kshop_category_id",
    "kshop_products":          "Products in K-Shop — name (Gujarati), price, discount_price, description, kshop_company_id. status=1 means active",
    "kshop_orders":            "Orders in K-Shop —  kshop_product_id, kshop_company_id, order_status_id",
    "kshop_weights":           "Weight units for K-Shop products",
    "kshop_category_company":  "Junction table: K-Shop companies ↔ categories",
    "company_orders":          "Orders from company perspective — linked to farmer_orders",
    "farmer_orders":           "Orders from farmer perspective —  company_id→kshop_companies",
    "order_statuses":          "Order status lookup (pending, processing, completed, cancelled)",
    "video_posts":             "Educational agricultural videos — title, video_url, views_count, video_category_id",
    "video_categories":        "Categories for educational videos — referenced as video_category_id",
    "video_likes":             "User likes on videos — video_post_id",
    "video_comments":          "Comments on videos — video_post_id, parent_comment_id (for replies)",
    "video_comment_likes":     "Likes on video comments — comment_id",
    "video_saves":             "Videos bookmarked/saved by users",
    "video_shares":            "Video sharing tracking — includes platform field",
    "video_views":             "Video view tracking — ip_address",
    "user_video_categories":   "User preferences for video categories they follow",
    "news":                    "Agricultural news articles — title, description, state_id, city_id, taluka_id for location filtering",
    "news_types":              "Types/categories of news articles",
    "states":                  "Indian states list — referenced as state_id",
    "cities":                  "Cities within states — has state_id. Referenced as city_id",
    "talukas":                 "Talukas (sub-districts) within cities — has city_id. Referenced as taluka_id",
    "media":                   "Media file storage references (images, documents)",
    "mediables":               "Polymorphic link table connecting media files to various models",
    "settings":                "Application configuration key-value pairs",
    "navigation_flow":         "UI navigation flow config for buy/sell category forms",
}


class SchemaGenerator:
    """Generates condensed schema and tool files from full_schema.json."""

    def __init__(
        self,
        full_schema_path: str,
        schemas_dir: str,
        tools_dir: str,
        privacy_policy_path: Optional[str] = None,
    ):
        # Resolve to ABSOLUTE paths at construction time so later operations
        # work regardless of cwd changes (uvicorn reloader, threading, etc.)
        self.full_schema_path      = os.path.abspath(full_schema_path)
        self.schemas_dir           = os.path.abspath(schemas_dir)
        self.tools_dir             = os.path.abspath(tools_dir)
        self.condensed_schema_path = os.path.join(self.schemas_dir, "condensed_schema.json")
        self.privacy_policy: PrivacyPolicy = get_privacy_policy(privacy_policy_path)
        Path(self.schemas_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tools_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"🗂️  SchemaGenerator paths resolved | "
            f"tools_dir={self.tools_dir} | schemas_dir={self.schemas_dir}"
        )
        

    def load_full_schema(self) -> Dict[str, Any]:
        try:
            with open(self.full_schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            logger.info("📖 Full schema loaded", tables=len(schema.get('tables', [])))
            return schema
        except FileNotFoundError:
            logger.error(f"❌ Full schema file not found: {self.full_schema_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in full schema: {e}")
            raise

    def generate_condensed_schema(self, full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate minimal sanitized schema: public table names + contexts."""
        logger.info("Generating sanitized condensed schema...")
        condensed = {
            "database_name": full_schema.get("database_name", "unknown"),
            "description": "Agricultural marketplace database - Krushi Ratn",
            "privacy_policy_version": self.privacy_policy.version,
            "privacy_policy_hash": self.privacy_policy.policy_hash,
            "privacy_mode": self.privacy_policy.mode,
            "total_tables": 0,
            "tables": [],
        }
        for table in full_schema.get("tables", []):
            table_name = table.get("table_name")
            if not self.privacy_policy.is_queryable_table(table_name):
                logger.info(f"Privacy policy excluded table from condensed schema: {table_name}")
                continue
            context = TABLE_CONTEXTS.get(table_name, f"Data related to {table_name.replace('_', ' ')}")
            if table_name not in TABLE_CONTEXTS:
                logger.warning(f"No context defined for table: {table_name}")
            condensed["tables"].append({"name": table_name, "context": context})
        condensed["total_tables"] = len(condensed["tables"])
        logger.info("Sanitized condensed schema generated", tables=len(condensed["tables"]))
        return condensed

    def _safe_columns_for_table(self, table_name: str, columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            col for col in columns
            if self.privacy_policy.is_safe_tool_column(table_name, col.get("name", ""))
        ]

    def _safe_relationships_for_table(self, table_name: str) -> List[Dict[str, Any]]:
        relationships = []
        for rel in VERIFIED_FK_MAP.get(table_name, []):
            ref = rel.get("references", "")
            if not ref or "." not in ref:
                continue
            ref_table = ref.split(".", 1)[0]
            if self.privacy_policy.is_sql_visible_table(ref_table):
                relationships.append(dict(rel))
            else:
                logger.info(
                    f"Privacy policy removed relationship: "
                    f"{table_name}.{rel.get('column')} -> {ref}"
                )
        return relationships

    def generate_tool_for_table(self, table: Dict[str, Any], database_name: str) -> Dict[str, Any]:
        """
        Generate tool definition for a table.
        Uses VERIFIED_FK_MAP — no naive inference.
        """
        table_name = table.get("table_name")
        columns    = self._safe_columns_for_table(table_name, table.get("columns", []))

        column_details = []
        for col in columns:
            col_detail = {
                "name":     col.get("name"),
                "type":     col.get("type"),
                "nullable": col.get("nullable", True),
            }
            if col.get("default") is not None:
                col_detail["default"] = col.get("default")
            if col.get("comment"):
                col_detail["comment"] = col.get("comment")
            column_details.append(col_detail)

        # Verified relationships, filtered through privacy policy. Join-only
        # tables can remain as FK metadata without getting their own query tool.
        relationships = self._safe_relationships_for_table(table_name)

        notes = [
            "Only SELECT queries allowed (READ-ONLY)",
            "Use proper WHERE clauses — never SELECT * without WHERE",
            "Use LIMIT to avoid large results",
            "Use JOIN type from relationships field: JOIN for required FK, LEFT JOIN for optional/nullable FK",
        ]
        if table_name in SOFT_DELETE_TABLES:
            notes.append(f"SOFT DELETE — always add: WHERE {table_name}.deleted_at IS NULL")
        if table_name in STATUS_NOTES:
            notes.append(STATUS_NOTES[table_name])
        if table_name in ("kshop_products", "buy_sell_products"):
            notes.append("Product names in GUJARATI SCRIPT — search both scripts: WHERE name LIKE '%keyword%' OR name LIKE '%gujarati%'")
            notes.append("STRIP intent words before search (mare=I want, karvu=to do, che=is, purchase, from, kshop) — these are NOT product keywords")
            notes.append("Search each keyword INDEPENDENTLY with OR — never full phrase LIKE")
        if table_name == "buy_sell_products":
            notes.append(
                "IMAGE COLUMN — product photos are stored INSIDE the form_data JSON column under key 'Images' (a JSON array). "
                "To return product images you MUST select: JSON_EXTRACT(bp.form_data, '$.Images') AS product_images. "
                "The standalone `images` column is legacy and should NOT be projected."
            )
            notes.append("Always also include bc.image AS category_image when JOINing buy_sell_categories (category thumbnail).")
        if table_name == "kshop_products":
            notes.append(
                "IMAGE COLUMN — kshop_products has NO direct image column. Product images live in the `media` table, "
                "linked through the polymorphic `mediables` table. To return product images you MUST add: "
                "LEFT JOIN mediables mb ON mb.mediable_id = kp.id AND mb.mediable_type LIKE 'App%KshopProduct' "
                "LEFT JOIN media m ON m.id = mb.media_id, then SELECT CONCAT(m.filename, '.', m.extension) AS product_image. "
                "Use LIKE 'App%KshopProduct' (NOT a backslash equality literal) — backslash escaping through JSON/MySQL is unreliable."
            )
            notes.append("Always also include kc.img AS category_img when JOINing kshop_categories (category thumbnail).")
        if table_name == "products":
            notes.append("Crop price table — JOIN yards→cities AND yards→talukas to filter by location, ORDER BY price_date DESC for latest")
            notes.append(
                "LOCATION FILTER RULE: A user-supplied location name may be a city, a taluka, or a yard name. "
                "ALWAYS match against ALL THREE name columns with OR — cities.name, talukas.name, AND yards.name — "
                "in BOTH English and Gujarati script. Example: '%Mahuva%' OR '%મહુવા%' OR '%mahuva%' against c.name AND t.name AND y.name. "
                "NEVER filter only on cities.name — talukas like મહુવા are not cities and would be missed."
            )
            notes.append(
                "LOCATION FK CHAIN: products.yard_id → yards (has city_id, taluka_id, state_id). "
                "To filter by taluka, JOIN talukas via y.taluka_id. To filter by city, JOIN cities via y.city_id. "
                "Both joins should be LEFT JOIN (some yards may have nullable city_id/taluka_id)."
            )
            notes.append(
                "MULTI-TALUKA RULE: When the user names a CITY (e.g. Bhavnagar), include all yards in that city — "
                "which means all yards across all talukas under that city_id. Do NOT additionally filter by "
                "taluka.name = city.name; just match c.name (and let the FK chain include every yard with that "
                "city_id automatically)."
            )

        return {
            "tool_name":      f"query_{table_name}",
            "description":    TABLE_CONTEXTS.get(table_name, f"Query the {table_name} table"),
            "table_name":     table_name,
            "database":       database_name,
            "privacy_policy_version": self.privacy_policy.version,
            "privacy_policy_hash": self.privacy_policy.policy_hash,
            "join_only_tables": sorted(self.privacy_policy.join_only_tables.keys()),
            "engine":         table.get("engine", "InnoDB"),
            "columns":        column_details,
            "column_count":   len(column_details),
            "relationships":  relationships,
            "example_queries": self._build_example_queries(table_name, [c["name"] for c in column_details]),
            "notes":          notes,
        }

    def _build_example_queries(self, table_name: str, safe_columns: Optional[List[str]] = None) -> List[str]:
        # Standard count-pattern templates per "entity table". Each template tells
        # the SQL generator (and the tool selector) that this table is the canonical
        # home for counting that entity. COUNT(*) is used because each row in
        # these tables is one unique entity (PK guarantees uniqueness).
        _COUNT_EXAMPLES: Dict[str, str] = {
            "kshop_products":      "SELECT COUNT(*) AS count FROM kshop_products kp WHERE kp.deleted_at IS NULL AND kp.status = 1",
            "buy_sell_products":   "SELECT COUNT(*) AS count FROM buy_sell_products bp WHERE bp.deleted_at IS NULL",
            "buy_sell_categories": "SELECT COUNT(*) AS count FROM buy_sell_categories bc WHERE bc.deleted_at IS NULL",
            "kshop_categories":    "SELECT COUNT(*) AS count FROM kshop_categories kc WHERE kc.deleted_at IS NULL",
            "kshop_companies":     "SELECT COUNT(*) AS count FROM kshop_companies kco WHERE kco.deleted_at IS NULL",
            "categories":          "SELECT COUNT(*) AS count FROM categories c WHERE c.deleted_at IS NULL",
            "sub_categories":      "SELECT COUNT(*) AS count FROM sub_categories sc WHERE sc.deleted_at IS NULL",
            "products":            "SELECT COUNT(*) AS count FROM products p WHERE p.deleted_at IS NULL",
            "seeds":               "SELECT COUNT(*) AS count FROM seeds s WHERE s.deleted_at IS NULL",
            "yards":               "SELECT COUNT(*) AS count FROM yards y WHERE y.deleted_at IS NULL",
            # Geography: count with optional location filter — show the three-way OR
            # pattern so the LLM can apply it when the user filters by location.
            "talukas":             "SELECT COUNT(*) AS count FROM talukas t LEFT JOIN cities c ON t.city_id = c.id WHERE t.deleted_at IS NULL AND (c.name LIKE '%Bhavnagar%' OR c.name LIKE '%ભાવનગર%')",
            "cities":              "SELECT COUNT(*) AS count FROM cities c LEFT JOIN states s ON c.state_id = s.id WHERE c.deleted_at IS NULL",
            "states":              "SELECT COUNT(*) AS count FROM states s WHERE s.deleted_at IS NULL",
            "news":                "SELECT COUNT(*) AS count FROM news n WHERE n.deleted_at IS NULL",
            "video_posts":         "SELECT COUNT(*) AS count FROM video_posts vp WHERE vp.deleted_at IS NULL",
            "video_categories":    "SELECT COUNT(*) AS count FROM video_categories vc WHERE vc.deleted_at IS NULL",
        }

        examples = {
            "kshop_products": [
                # IMAGE: kshop_products has no direct image column — resolve via
                # mediables (polymorphic, mediable_type LIKE 'App%KshopProduct')
                # → media (filename + extension). LIKE pattern avoids the
                # backslash-escape fragility of equality with 'App\\Models\\X'.
                "SELECT kp.id, kp.name, kp.price, kp.discount_price, kco.name AS company, kc.name AS category, kc.img AS category_img, "
                "CONCAT(m.filename, '.', m.extension) AS product_image "
                "FROM kshop_products kp "
                "JOIN kshop_companies kco ON kp.kshop_company_id = kco.id "
                "LEFT JOIN kshop_categories kc ON kp.kshop_category_id = kc.id AND kc.deleted_at IS NULL "
                "LEFT JOIN kshop_weights kw ON kp.kshop_weight_id = kw.id "
                "LEFT JOIN mediables mb ON mb.mediable_id = kp.id AND mb.mediable_type LIKE 'App%KshopProduct' "
                "LEFT JOIN media m ON m.id = mb.media_id "
                "WHERE kp.deleted_at IS NULL AND kp.status = 1 "
                "AND kp.kshop_category_id IN (SELECT id FROM kshop_categories WHERE (name LIKE '%weeder%' OR name LIKE '%વીડર%') AND deleted_at IS NULL) "
                "ORDER BY kp.updated_at DESC LIMIT 50",
                # COUNT: "how many products in kshop" — use the entity table directly.
                _COUNT_EXAMPLES["kshop_products"],
            ],
            "buy_sell_products": [
                # NOTE: No status = 'active' filter here — per SQL generation Rule #12,
                # status filtering is handled by the post-retrieval status_filter layer.
                # Adding it in SQL would silently exclude 'sold_out' and other valid states.
                # IMAGE: product photos live INSIDE form_data JSON column under key 'Images'.
                # The standalone `images` column is legacy — do not project it.
                "SELECT bp.id, bp.product_name, bp.price, bp.quantity_available, "
                "JSON_EXTRACT(bp.form_data, '$.Images') AS product_images, "
                "bc.name AS category, bc.image AS category_image "
                "FROM buy_sell_products bp "
                "LEFT JOIN buy_sell_categories bc ON bp.category_id = bc.id AND bc.deleted_at IS NULL "
                "WHERE bp.deleted_at IS NULL "
                "AND bp.category_id IN (SELECT id FROM buy_sell_categories WHERE (name LIKE '%tractor%' OR name LIKE '%ટ્રેક્ટર%') AND deleted_at IS NULL) "
                "ORDER BY bp.created_at DESC LIMIT 50",
                # COUNT: "how many listings in buy/sell" — use the entity table directly.
                _COUNT_EXAMPLES["buy_sell_products"],
            ],
            "products": [
                # Crop + city — must JOIN talukas too AND match against city/taluka/yard
                # name columns with OR. Krushi Ratn yards live in talukas; a yard whose
                # taluka equals the user-typed city would otherwise be missed.
                "SELECT sc.name AS crop, sc.img AS crop_img, p.min_price, p.max_price, p.price_date, y.name AS yard, c.name AS city, t.name AS taluka "
                "FROM products p "
                "JOIN sub_categories sc ON p.subcategory_id = sc.id "
                "JOIN yards y ON p.yard_id = y.id "
                "LEFT JOIN cities c ON y.city_id = c.id "
                "LEFT JOIN talukas t ON y.taluka_id = t.id "
                "LEFT JOIN weights w ON p.weight_id = w.id "
                "WHERE p.deleted_at IS NULL "
                "AND (sc.name LIKE '%kapas%' OR sc.name LIKE '%કપાસ%' OR p.subcategory_name LIKE '%kapas%') "
                "AND (c.name LIKE '%Bhavnagar%' OR c.name LIKE '%ભાવનગર%' OR t.name LIKE '%Bhavnagar%' OR t.name LIKE '%ભાવનગર%' OR y.name LIKE '%Bhavnagar%' OR y.name LIKE '%ભાવનગર%') "
                "ORDER BY p.price_date DESC LIMIT 50",
                # Crop + taluka (e.g. onion in મહુવા) — taluka name must be matched on
                # talukas.name through yards.taluka_id; matching only cities.name would
                # silently miss every yard whose city.name differs from the taluka name.
                "SELECT sc.name AS crop, sc.img AS crop_img, p.min_price, p.max_price, p.price_date, y.name AS yard, c.name AS city, t.name AS taluka "
                "FROM products p "
                "JOIN sub_categories sc ON p.subcategory_id = sc.id "
                "JOIN yards y ON p.yard_id = y.id "
                "LEFT JOIN cities c ON y.city_id = c.id "
                "LEFT JOIN talukas t ON y.taluka_id = t.id "
                "WHERE p.deleted_at IS NULL "
                "AND (sc.name LIKE '%onion%' OR sc.name LIKE '%ડુંગળી%' OR sc.name LIKE '%dungli%' OR p.subcategory_name LIKE '%onion%') "
                "AND (t.name LIKE '%Mahuva%' OR t.name LIKE '%મહુવા%' OR c.name LIKE '%Mahuva%' OR c.name LIKE '%મહુવા%' OR y.name LIKE '%Mahuva%' OR y.name LIKE '%મહુવા%') "
                "ORDER BY p.price_date DESC LIMIT 50",
                # COUNT: "how many price records exist" — entity = product, table = products.
                _COUNT_EXAMPLES["products"],
                # COUNT(DISTINCT): "how many different crops have price data" — entity (crop)
                # lives in sub_categories, but uniqueness is across the products transactional
                # table, so COUNT(DISTINCT subcategory_id) is the right pattern.
                "SELECT COUNT(DISTINCT p.subcategory_id) AS count FROM products p WHERE p.deleted_at IS NULL",
            ],
        }
        if table_name in examples:
            return examples[table_name]

        # Tables without an explicit rich example get a minimal data-fetch example
        # PLUS the count-pattern example (when applicable). The count example
        # ensures the tool selector sees this table as relevant for count
        # questions about its entity, and the SQL generator has a matching
        # template to reference.
        result: List[str] = []
        safe_columns = safe_columns or ["id"]
        select_cols = ", ".join(safe_columns[:6])
        if table_name in SOFT_DELETE_TABLES:
            result.append(f"SELECT {select_cols} FROM {table_name} WHERE deleted_at IS NULL LIMIT 10")
        else:
            result.append(f"SELECT {select_cols} FROM {table_name} WHERE 1=1 LIMIT 10")
        # Append count example if curated; otherwise generate a generic one for
        # soft-delete tables so "how many <entity>" queries always have a template.
        if table_name in _COUNT_EXAMPLES:
            result.append(_COUNT_EXAMPLES[table_name])
        elif table_name in SOFT_DELETE_TABLES:
            result.append(f"SELECT COUNT(*) AS count FROM {table_name} WHERE deleted_at IS NULL")
        return result

    def save_condensed_schema(self, condensed_schema: Dict[str, Any]):
        with open(self.condensed_schema_path, 'w', encoding='utf-8') as f:
            json.dump(condensed_schema, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Condensed schema saved: {self.condensed_schema_path}")

    def save_tool(self, tool: Dict[str, Any], tool_name: str):
        tool_path = os.path.join(self.tools_dir, f"{tool_name}.json")
        with open(tool_path, 'w', encoding='utf-8') as f:
            json.dump(tool, f, indent=2, ensure_ascii=False)

    def _artifacts_current(self) -> bool:
        if not os.path.exists(self.condensed_schema_path):
            return False
        try:
            with open(self.condensed_schema_path, "r", encoding="utf-8") as f:
                condensed = json.load(f)
            return condensed.get("privacy_policy_hash") == self.privacy_policy.policy_hash
        except Exception:
            return False

    def _delete_stale_tools(self, allowed_table_names: set) -> int:
        deleted = 0
        for tool_file in Path(self.tools_dir).glob("*_tool.json"):
            table_name = tool_file.stem.replace("_tool", "")
            if table_name not in allowed_table_names:
                try:
                    tool_file.unlink()
                    deleted += 1
                    logger.info(f"Deleted stale/private tool file: {tool_file.name}")
                except OSError as e:
                    logger.warning(f"Could not delete stale/private tool file {tool_file}: {e}")
        return deleted

    def generate_all(self, force: bool = False) -> dict:
        """Generate sanitized schema and tool files."""
        stats = {
            "condensed_schema": "skipped",
            "tools_generated": 0,
            "tools_skipped": 0,
            "tools_deleted": 0,
            "total_tables": 0,
            "policy_hash": self.privacy_policy.policy_hash,
        }

        try:
            full_schema = self.load_full_schema()
        except FileNotFoundError:
            logger.warning("⚠️  full_schema.json not found — skipping generation.")
            if os.path.exists(self.condensed_schema_path):
                stats["condensed_schema"] = "exists"
            return stats

        allowed_tables = [
            table for table in full_schema.get("tables", [])
            if self.privacy_policy.is_queryable_table(table.get("table_name", ""))
        ]
        allowed_names = {table.get("table_name") for table in allowed_tables}
        stats["total_tables"] = len(allowed_tables)
        database_name = full_schema.get("database_name", "unknown")

        existing_tools = list(Path(self.tools_dir).glob("*_tool.json"))
        if not force and self._artifacts_current() and len(existing_tools) >= len(allowed_tables):
            logger.info(f"⏭️  Condensed schema exists: {self.condensed_schema_path}")
            stats["tools_deleted"] = self._delete_stale_tools(allowed_names)
            stats["condensed_schema"] = "exists"
            stats["tools_skipped"] = len(list(Path(self.tools_dir).glob("*_tool.json")))
            return stats

        condensed = self.generate_condensed_schema(full_schema)
        self.save_condensed_schema(condensed)
        stats["condensed_schema"] = "generated"

        logger.info("Generating sanitized individual tool files...")
        stats["tools_deleted"] = self._delete_stale_tools(allowed_names)
        for table in allowed_tables:
            table_name = table.get("table_name")
            tool = self.generate_tool_for_table(table, database_name)
            self.save_tool(tool, f"{table_name}_tool")
            stats["tools_generated"] += 1
        logger.info(f"Generated {stats['tools_generated']} sanitized tool files")

        logger.info("📊 GENERATION SUMMARY:",
                    condensed=stats["condensed_schema"],
                    tools_generated=stats["tools_generated"],
                    tools_skipped=stats["tools_skipped"])
        return stats

    def load_condensed_schema(self) -> Dict[str, Any]:
        if not os.path.exists(self.condensed_schema_path):
            raise FileNotFoundError(f"Condensed schema not found: {self.condensed_schema_path}")
        with open(self.condensed_schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_tool(self, table_name: str) -> Dict[str, Any]:
        tool_path = os.path.join(self.tools_dir, f"{table_name}_tool.json")
        if not os.path.exists(tool_path):
            raise FileNotFoundError(f"Tool not found: {tool_path}")
        with open(tool_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_all_tools(self) -> Dict[str, Dict[str, Any]]:
        tools = {}
        for tool_file in Path(self.tools_dir).glob("*_tool.json"):
            table_name = tool_file.stem.replace("_tool", "")
            if not self.privacy_policy.is_queryable_table(table_name):
                logger.info(f"Privacy policy ignored tool file at load time: {tool_file.name}")
                continue
            with open(tool_file, 'r', encoding='utf-8') as f:
                tools[table_name] = json.load(f)

        # Defensive diagnostic: if we loaded zero but the directory exists,
        # log loudly so the path mismatch is immediately visible.
        if not tools:
            logger.warning(
                f"⚠️  load_all_tools() loaded 0 tools | "
                f"tools_dir={self.tools_dir} | "
                f"cwd={os.getcwd()} | "
                f"dir_exists={os.path.isdir(self.tools_dir)} | "
                f"files_in_dir={len(list(Path(self.tools_dir).glob('*'))) if os.path.isdir(self.tools_dir) else 0}"
            )
        else:
            logger.info(f"📖 Loaded {len(tools)} tools from {self.tools_dir}")

        return tools

    def get_available_tool_names(self) -> List[str]:
        tools = []
        for tool_file in Path(self.tools_dir).glob("*_tool.json"):
            table_name = tool_file.stem.replace("_tool", "")
            if not self.privacy_policy.is_queryable_table(table_name):
                continue
            tools.append(f"query_{table_name}")
        if not tools:
            logger.warning(
                f"⚠️  get_available_tool_names() returned empty | "
                f"tools_dir={self.tools_dir} | cwd={os.getcwd()}"
            )
        return sorted(tools)


def initialize_schemas(
    schemas_dir: str = "app/schemas",
    tools_dir: str   = "app/schemas/tools",
    privacy_policy_path: Optional[str] = None,
) -> SchemaGenerator:
    """Initialize and generate schemas on application startup."""
    full_schema_path = os.path.join(schemas_dir, "full_schema.json")
    generator = SchemaGenerator(full_schema_path, schemas_dir, tools_dir, privacy_policy_path)
    logger.info("🚀 Initializing schema generator...")
    stats = generator.generate_all(force=False)
    logger.info("📊 GENERATION SUMMARY:")
    logger.info(f"   Condensed: {stats['condensed_schema']}")
    logger.info(f"   Tools Generated: {stats['tools_generated']}")
    logger.info(f"   Tools Skipped: {stats['tools_skipped']}")
    logger.info(f"   Tools Deleted: {stats.get('tools_deleted', 0)}")
    logger.info(f"   Privacy Policy: {stats.get('policy_hash', '')}")
    return generator


if __name__ == "__main__":
    generator = initialize_schemas()
    tool_names = generator.get_available_tool_names()
    print(f"\n✅ Available Tools ({len(tool_names)}):")
    for i, tool in enumerate(tool_names, 1):
        print(f"   {i}. {tool}")