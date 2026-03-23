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
        {"column": "buyer_id",   "references": "users.id",             "join_type": "LEFT JOIN", "description": "User who is buying"},
        {"column": "seller_id",  "references": "users.id",             "join_type": "LEFT JOIN", "description": "User who is selling"},
    ],
    "buy_sell_products": [
        {"column": "category_id", "references": "buy_sell_categories.id", "join_type": "LEFT JOIN", "description": "Product category"},
        {"column": "seller_id",   "references": "users.id",               "join_type": "LEFT JOIN", "description": "Farmer/user selling this product"},
    ],
    "cities": [
        {"column": "state_id", "references": "states.id", "join_type": "LEFT JOIN", "description": "State this city belongs to"},
    ],
    "company_orders": [
        {"column": "farmer_order_id", "references": "farmer_orders.id",  "join_type": "JOIN",      "description": "Related farmer order"},
        {"column": "user_id",         "references": "users.id",          "join_type": "LEFT JOIN", "description": "User placing the order"},
        {"column": "farmer_id",       "references": "users.id",          "join_type": "LEFT JOIN", "description": "Farmer in the order"},
        {"column": "order_status_id", "references": "order_statuses.id", "join_type": "LEFT JOIN", "description": "Current order status"},
        {"column": "subcategory_id",  "references": "sub_categories.id", "join_type": "LEFT JOIN", "description": "Product subcategory"},
        {"column": "weight_id",       "references": "weights.id",        "join_type": "LEFT JOIN", "description": "Weight unit"},
    ],
    "farmer_orders": [
        {"column": "user_id",         "references": "users.id",           "join_type": "LEFT JOIN", "description": "Farmer placing the order"},
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
        {"column": "user_id",           "references": "users.id",            "join_type": "LEFT JOIN", "description": "User who placed order"},
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
        {"column": "user_id",        "references": "users.id",           "join_type": "JOIN",      "description": "Owner farmer/user"},
    ],
    "user_subcategories": [
        {"column": "subcategory_id", "references": "sub_categories.id", "join_type": "LEFT JOIN", "description": "Preferred subcategory"},
        {"column": "user_id",        "references": "users.id",           "join_type": "JOIN",      "description": "User"},
    ],
    "user_talukas": [
        {"column": "user_id",   "references": "users.id",   "join_type": "JOIN", "description": "User"},
        {"column": "taluka_id", "references": "talukas.id", "join_type": "JOIN", "description": "Taluka user belongs to"},
    ],
    "user_video_categories": [
        {"column": "user_id",           "references": "users.id",            "join_type": "JOIN", "description": "User"},
        {"column": "video_category_id", "references": "video_categories.id", "join_type": "JOIN", "description": "Preferred video category"},
    ],
    "users": [
        {"column": "state_id", "references": "states.id", "join_type": "LEFT JOIN", "description": "State of user"},
        {"column": "city_id",  "references": "cities.id", "join_type": "LEFT JOIN", "description": "City of user"},
    ],
    "video_comment_likes": [
        {"column": "comment_id", "references": "video_comments.id", "join_type": "JOIN", "description": "Comment that was liked"},
        {"column": "user_id",    "references": "users.id",          "join_type": "JOIN", "description": "User who liked"},
    ],
    "video_comments": [
        {"column": "video_post_id",     "references": "video_posts.id",    "join_type": "JOIN",      "description": "Video being commented on"},
        {"column": "user_id",           "references": "users.id",          "join_type": "JOIN",      "description": "User who commented"},
        {"column": "parent_comment_id", "references": "video_comments.id", "join_type": "LEFT JOIN", "description": "Parent comment if reply (self-reference)"},
    ],
    "video_likes": [
        {"column": "video_post_id", "references": "video_posts.id", "join_type": "JOIN", "description": "Video that was liked"},
        {"column": "user_id",       "references": "users.id",       "join_type": "JOIN", "description": "User who liked"},
    ],
    "video_posts": [
        {"column": "user_id",           "references": "users.id",            "join_type": "JOIN",      "description": "User who posted video"},
        {"column": "video_category_id", "references": "video_categories.id", "join_type": "LEFT JOIN", "description": "Video category"},
    ],
    "video_saves": [
        {"column": "video_post_id", "references": "video_posts.id", "join_type": "JOIN", "description": "Saved video"},
        {"column": "user_id",       "references": "users.id",       "join_type": "JOIN", "description": "User who saved"},
    ],
    "video_shares": [
        {"column": "video_post_id", "references": "video_posts.id", "join_type": "JOIN", "description": "Shared video"},
        {"column": "user_id",       "references": "users.id",       "join_type": "JOIN", "description": "User who shared"},
    ],
    "video_views": [
        {"column": "video_post_id", "references": "video_posts.id", "join_type": "JOIN", "description": "Viewed video"},
        {"column": "user_id",       "references": "users.id",       "join_type": "JOIN", "description": "User who viewed"},
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

STATUS_NOTES = {
    "kshop_products":   "Active only: WHERE status = 1 AND deleted_at IS NULL",
    "buy_sell_products":"Active only: WHERE status = 'active' AND deleted_at IS NULL",
    "video_posts":      "Published only: WHERE status = 1 AND deleted_at IS NULL",
    "kshop_companies":  "Active only: WHERE status = 1 AND deleted_at IS NULL",
    "kshop_categories": "Active only: WHERE status = 1 AND deleted_at IS NULL",
    "sub_categories":   "Active only: WHERE status = 1 AND deleted_at IS NULL",
    "users":            "Active only: WHERE status = 1 AND deleted_at IS NULL",
}

TABLE_CONTEXTS = {
    "users":                   "Farmer and user accounts — profile, contact, location. Referenced as user_id, seller_id, buyer_id, farmer_id",
    "buy_sell_categories":     "Categories for buy/sell marketplace (animals, equipment, crops). Referenced as category_id in buy_sell_products",
    "buy_sell_products":       "Marketplace product listings by farmers — product_name, price, quantity, status='active'/'sold_out'. seller_id→users",
    "buy_sell_orders":         "Purchase transactions in buy/sell marketplace — buyer_id and seller_id both reference users",
    "buy_sell_category_fields":"Custom form fields per buy/sell product category",
    "buy_sell_category_steps": "Multi-step form wizard steps for buy/sell product listing",
    "categories":              "Main product categories — referenced by sub_categories",
    "sub_categories":          "Sub-categories under main categories — referenced as subcategory_id in products, seeds, user_products",
    "products":                "Crop/commodity market prices — subcategory_id (crop type), yard_id (market), min_price, max_price, price_date",
    "seeds":                   "Seed products with subcategory_id and variety info",
    "user_products":           "Products owned/listed by users with price range and subcategory",
    "user_subcategories":      "User preferences for product subcategories",
    "weights":                 "Weight measurement units (kg, quintal, ton) — referenced as weight_id",
    "yards":                   "Market yard locations — name, city_id, state_id, taluka_id. Referenced as yard_id in products",
    "kshop_companies":         "Companies selling in K-Shop — referenced as kshop_company_id",
    "kshop_categories":        "Product categories in K-Shop — referenced as kshop_category_id",
    "kshop_products":          "Products in K-Shop — name (Gujarati), price, discount_price, description, kshop_company_id. status=1 means active",
    "kshop_orders":            "Orders in K-Shop — user_id, kshop_product_id, kshop_company_id, order_status_id",
    "kshop_weights":           "Weight units for K-Shop products",
    "kshop_category_company":  "Junction table: K-Shop companies ↔ categories",
    "company_orders":          "Orders from company perspective — linked to farmer_orders",
    "farmer_orders":           "Orders from farmer perspective — user_id=farmer, company_id→kshop_companies",
    "order_statuses":          "Order status lookup (pending, processing, completed, cancelled)",
    "video_posts":             "Educational agricultural videos — title, video_url, views_count, user_id (creator), video_category_id",
    "video_categories":        "Categories for educational videos — referenced as video_category_id",
    "video_likes":             "User likes on videos — video_post_id and user_id",
    "video_comments":          "Comments on videos — video_post_id, user_id, parent_comment_id (for replies)",
    "video_comment_likes":     "Likes on video comments — comment_id and user_id",
    "video_saves":             "Videos bookmarked/saved by users",
    "video_shares":            "Video sharing tracking — includes platform field",
    "video_views":             "Video view tracking — user_id and ip_address",
    "user_video_categories":   "User preferences for video categories they follow",
    "news":                    "Agricultural news articles — title, description, state_id, city_id, taluka_id for location filtering",
    "news_types":              "Types/categories of news articles",
    "states":                  "Indian states list — referenced as state_id",
    "cities":                  "Cities within states — has state_id. Referenced as city_id",
    "talukas":                 "Talukas (sub-districts) within cities — has city_id. Referenced as taluka_id",
    "user_talukas":            "User location preferences (which talukas they belong to)",
    "notifications":           "User push notifications",
    "user_otps":               "OTP codes for user mobile authentication",
    "media":                   "Media file storage references (images, documents)",
    "mediables":               "Polymorphic link table connecting media files to various models",
    "settings":                "Application configuration key-value pairs",
    "navigation_flow":         "UI navigation flow config for buy/sell category forms",
}


class SchemaGenerator:
    """Generates condensed schema and tool files from full_schema.json."""

    def __init__(self, full_schema_path: str, schemas_dir: str, tools_dir: str):
        self.full_schema_path    = full_schema_path
        self.schemas_dir         = schemas_dir
        self.tools_dir           = tools_dir
        self.condensed_schema_path = os.path.join(schemas_dir, "condensed_schema.json")
        Path(schemas_dir).mkdir(parents=True, exist_ok=True)
        Path(tools_dir).mkdir(parents=True, exist_ok=True)

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
        """Generate minimal condensed schema — table names + context descriptions."""
        logger.info("🔧 Generating condensed schema...")
        condensed = {
            "database_name": full_schema.get("database_name", "unknown"),
            "description":   "Agricultural marketplace database — Krushi Ratn",
            "total_tables":  full_schema.get("total_tables", 0),
            "tables":        [],
        }
        for table in full_schema.get("tables", []):
            table_name = table.get("table_name")
            context = TABLE_CONTEXTS.get(table_name, f"Data related to {table_name.replace('_', ' ')}")
            if table_name not in TABLE_CONTEXTS:
                logger.warning(f"⚠️  No context defined for table: {table_name}")
            condensed["tables"].append({"name": table_name, "context": context})
        logger.info("✅ Condensed schema generated", tables=len(condensed["tables"]))
        return condensed

    def generate_tool_for_table(self, table: Dict[str, Any], database_name: str) -> Dict[str, Any]:
        """
        Generate tool definition for a table.
        Uses VERIFIED_FK_MAP — no naive inference.
        """
        table_name = table.get("table_name")
        columns    = table.get("columns", [])

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

        # Verified relationships — no guessing
        relationships = VERIFIED_FK_MAP.get(table_name, [])

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
        if table_name == "products":
            notes.append("Crop price table — JOIN yards→cities to filter by location, ORDER BY price_date DESC for latest")

        return {
            "tool_name":      f"query_{table_name}",
            "description":    TABLE_CONTEXTS.get(table_name, f"Query the {table_name} table"),
            "table_name":     table_name,
            "database":       database_name,
            "engine":         table.get("engine", "InnoDB"),
            "columns":        column_details,
            "column_count":   len(column_details),
            "relationships":  relationships,
            "example_queries": self._build_example_queries(table_name),
            "notes":          notes,
        }

    def _build_example_queries(self, table_name: str) -> List[str]:
        examples = {
            "kshop_products": [
                "SELECT kp.name, kp.price, kp.discount_price, kco.name AS company, COALESCE(kc.name,'N/A') AS category "
                "FROM kshop_products kp "
                "JOIN kshop_companies kco ON kp.kshop_company_id = kco.id "
                "LEFT JOIN kshop_categories kc ON kp.kshop_category_id = kc.id AND kc.deleted_at IS NULL "
                "LEFT JOIN kshop_weights kw ON kp.kshop_weight_id = kw.id "
                "WHERE kp.deleted_at IS NULL AND kp.status = 1 "
                "AND (kp.name LIKE '%balwan%' OR kp.name LIKE '%બલવાન%') "
                "AND (kp.name LIKE '%weeder%' OR kp.name LIKE '%વીડર%') "
                "ORDER BY kp.updated_at DESC LIMIT 50",
            ],
            "buy_sell_products": [
                "SELECT bp.product_name, bp.price, bp.quantity_available, bc.name AS category, u.name AS seller "
                "FROM buy_sell_products bp "
                "LEFT JOIN buy_sell_categories bc ON bp.category_id = bc.id AND bc.deleted_at IS NULL "
                "LEFT JOIN users u ON bp.seller_id = u.id "
                "WHERE bp.deleted_at IS NULL AND bp.status = 'active' "
                "AND (bp.product_name LIKE '%tractor%' OR bp.product_name LIKE '%ટ્રેક્ટર%') "
                "ORDER BY bp.created_at DESC LIMIT 50",
            ],
            "products": [
                "SELECT sc.name AS crop, p.min_price, p.max_price, p.price_date, y.name AS yard, c.name AS city "
                "FROM products p "
                "JOIN sub_categories sc ON p.subcategory_id = sc.id "
                "JOIN yards y ON p.yard_id = y.id "
                "JOIN cities c ON y.city_id = c.id "
                "LEFT JOIN weights w ON p.weight_id = w.id "
                "WHERE p.deleted_at IS NULL "
                "AND (sc.name LIKE '%kapas%' OR sc.name LIKE '%કપાસ%' OR p.subcategory_name LIKE '%kapas%') "
                "AND (c.name LIKE '%Bhavnagar%' OR c.name LIKE '%ભાવનગર%') "
                "ORDER BY p.price_date DESC LIMIT 20",
            ],
        }
        if table_name in examples:
            return examples[table_name]
        if table_name in SOFT_DELETE_TABLES:
            return [f"SELECT * FROM {table_name} WHERE deleted_at IS NULL LIMIT 10"]
        return [f"SELECT * FROM {table_name} LIMIT 10"]

    def save_condensed_schema(self, condensed_schema: Dict[str, Any]):
        with open(self.condensed_schema_path, 'w', encoding='utf-8') as f:
            json.dump(condensed_schema, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Condensed schema saved: {self.condensed_schema_path}")

    def save_tool(self, tool: Dict[str, Any], tool_name: str):
        tool_path = os.path.join(self.tools_dir, f"{tool_name}.json")
        with open(tool_path, 'w', encoding='utf-8') as f:
            json.dump(tool, f, indent=2, ensure_ascii=False)

    def generate_all(self, force: bool = False) -> dict:
        """Generate condensed schema and all tool files. force=True regenerates even if files exist."""
        stats = {"condensed_schema": "skipped", "tools_generated": 0, "tools_skipped": 0, "total_tables": 0}

        condensed_exists = os.path.exists(self.condensed_schema_path) and not force
        existing_tools   = list(Path(self.tools_dir).glob("*_tool.json"))
        tools_exist      = len(existing_tools) > 0 and not force

        if condensed_exists and tools_exist:
            logger.info(f"⏭️  Condensed schema exists: {self.condensed_schema_path}")
            logger.info(f"⏭️  Tools exist: {len(existing_tools)} files found")
            stats["condensed_schema"] = "exists"
            stats["tools_skipped"]    = len(existing_tools)
            return stats

        try:
            full_schema = self.load_full_schema()
        except FileNotFoundError:
            logger.warning("⚠️  full_schema.json not found — skipping generation.")
            if condensed_exists:
                stats["condensed_schema"] = "exists"
            return stats

        stats["total_tables"] = full_schema.get("total_tables", 0)
        database_name = full_schema.get("database_name", "unknown")

        if not condensed_exists:
            condensed = self.generate_condensed_schema(full_schema)
            self.save_condensed_schema(condensed)
            stats["condensed_schema"] = "generated"
        else:
            logger.info(f"⏭️  Condensed schema exists: {self.condensed_schema_path}")
            stats["condensed_schema"] = "exists"

        if not tools_exist:
            logger.info("🔧 Generating individual tool files...")
            for table in full_schema.get("tables", []):
                table_name = table.get("table_name")
                tool_path  = os.path.join(self.tools_dir, f"{table_name}_tool.json")
                if os.path.exists(tool_path) and not force:
                    stats["tools_skipped"] += 1
                else:
                    tool = self.generate_tool_for_table(table, database_name)
                    self.save_tool(tool, f"{table_name}_tool")
                    stats["tools_generated"] += 1
            logger.info(f"✅ Generated {stats['tools_generated']} tool files")
        else:
            stats["tools_skipped"] = len(existing_tools)
            logger.info(f"⏭️  Tools exist: {len(existing_tools)} files skipped")

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
            with open(tool_file, 'r', encoding='utf-8') as f:
                tools[table_name] = json.load(f)
        logger.info(f"📖 Loaded {len(tools)} tools")
        return tools

    def get_available_tool_names(self) -> List[str]:
        tools = []
        for tool_file in Path(self.tools_dir).glob("*_tool.json"):
            table_name = tool_file.stem.replace("_tool", "")
            tools.append(f"query_{table_name}")
        return sorted(tools)


def initialize_schemas(
    schemas_dir: str = "app/schemas",
    tools_dir: str   = "app/schemas/tools",
) -> SchemaGenerator:
    """Initialize and generate schemas on application startup."""
    full_schema_path = os.path.join(schemas_dir, "full_schema.json")
    generator = SchemaGenerator(full_schema_path, schemas_dir, tools_dir)
    logger.info("🚀 Initializing schema generator...")
    stats = generator.generate_all(force=False)
    logger.info("📊 GENERATION SUMMARY:")
    logger.info(f"   Condensed: {stats['condensed_schema']}")
    logger.info(f"   Tools Generated: {stats['tools_generated']}")
    logger.info(f"   Tools Skipped: {stats['tools_skipped']}")
    return generator


if __name__ == "__main__":
    generator = initialize_schemas()
    tool_names = generator.get_available_tool_names()
    print(f"\n✅ Available Tools ({len(tool_names)}):")
    for i, tool in enumerate(tool_names, 1):
        print(f"   {i}. {tool}")