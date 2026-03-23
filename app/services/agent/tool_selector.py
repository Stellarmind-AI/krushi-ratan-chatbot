"""
Tool Selector — Intelligent table selection with FK dependency resolution.
Selects the primary table(s) + all JOIN dependencies in one shot.
"""

import json
from typing import List, Dict, Any
from app.services.llm.manager import get_llm_manager
from app.models.chat_models import LLMMessage, ToolSelectionResponse
from app.utils.json_parser import json_parser
from app.core.logger import get_agent_logger

logger = get_agent_logger()


# ══════════════════════════════════════════════════════════════════════════════
# FK DEPENDENCY CHAINS
# When you pick a table, you MUST also pick these tables for JOINs.
# Built from the real tool files.
# ══════════════════════════════════════════════════════════════════════════════
FK_DEPS: Dict[str, List[str]] = {
    "kshop_products":         ["kshop_companies", "kshop_categories", "kshop_weights"],
    "kshop_orders":           ["kshop_products", "kshop_companies", "kshop_categories", "order_statuses", "users"],
    "kshop_category_company": ["kshop_companies", "kshop_categories"],
    "buy_sell_products":      ["buy_sell_categories", "users"],
    "buy_sell_orders":        ["buy_sell_products", "users"],
    "buy_sell_category_fields": ["buy_sell_categories"],
    "buy_sell_category_steps":  ["buy_sell_categories"],
    "products":               ["sub_categories", "yards", "weights"],
    "yards":                  ["cities", "states", "talukas"],
    "cities":                 ["states"],
    "talukas":                ["cities"],
    "news":                   ["cities", "states", "talukas"],
    "seeds":                  ["sub_categories"],
    "sub_categories":         ["categories"],
    "farmer_orders":          ["users", "kshop_companies", "sub_categories", "weights", "order_statuses"],
    "company_orders":         ["farmer_orders", "users", "sub_categories", "weights", "order_statuses"],
    "video_posts":            ["users", "video_categories"],
    "video_comments":         ["video_posts", "users"],
    "video_likes":            ["video_posts", "users"],
    "video_saves":            ["video_posts", "users"],
    "video_shares":           ["video_posts", "users"],
    "video_views":            ["video_posts", "users"],
    "video_comment_likes":    ["video_comments", "users"],
    "user_products":          ["sub_categories", "users"],
    "user_subcategories":     ["sub_categories", "users"],
    "user_talukas":           ["users", "talukas"],
    "user_video_categories":  ["users", "video_categories"],
    "users":                  ["states", "cities"],
}

# Intent words to strip when doing keyword-based topic detection
INTENT_WORDS = {
    "mare", "maro", "mari", "mara", "mane",
    "karvu", "karvani", "karvo", "karu", "karshu",
    "che", "chhe", "hatu", "hati",
    "joiyu", "joiye", "jovu",
    "levu", "levo", "levi", "lidhu",
    "purchase", "kharidi", "kharido",
    "from", "thi", "ma", "no", "ni", "na", "nu", "ne",
    "info", "mahiti", "jankari", "vishhe", "vishe",
    "apo", "apso", "batao", "batavo", "please", "pls",
    "i", "want", "show", "me", "tell", "get", "find", "give",
}

# Source-specific keywords — when user mentions these, go directly to that source
SOURCE_KEYWORDS = {
    "kshop":     ["kshop", "k-shop", "k shop"],
    "buy_sell":  ["buy sell", "buysell", "buy/sell", "marketplace"],
    "video":     ["video", "watch", "creator"],
    "news":      ["news", "samachar", "update"],
    "price":     ["price", "bhav", "ભાવ", "rate", "mandi", "yard", "yard price"],
    "order":     ["order", "orders", "purchase history", "my order"],
    "user":      ["user", "farmer", "profile", "account"],
}


class ToolSelector:
    """Selects tables needed to answer a query, including all JOIN dependencies."""

    def __init__(self):
        self.llm_manager = get_llm_manager()

    async def select_tools(
        self,
        user_query: str,
        condensed_schema: dict,
        available_tools: List[str]
    ) -> ToolSelectionResponse:
        try:
            logger.info("🔧 TOOL SELECTION STARTED", query=user_query[:100])

            system_prompt = self._build_system_prompt(condensed_schema, available_tools)
            user_message  = self._build_user_message(user_query)

            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user",   content=user_message),
            ]

            response = await self.llm_manager.generate(
                messages=messages,
                temperature=0.0,
                max_tokens=500,
            )

            selected = self._parse_and_expand(response.content, available_tools)
            logger.tool_selection(tools=selected, query=user_query)

            return ToolSelectionResponse(
                selected_tools=selected,
                reasoning=response.content,
            )

        except Exception as e:
            logger.error_with_context(e, {"action": "select_tools",
                                          "query": user_query[:200]})
            raise

    def _build_system_prompt(self, condensed_schema: dict, available_tools: List[str]) -> str:
        table_ctx = self._format_schema_info(condensed_schema)
        tools_str = "\n".join(f"  {t}" for t in sorted(available_tools))

        # Format FK dependency chains for the prompt
        dep_lines = []
        for table, deps in FK_DEPS.items():
            tool_name = f"query_{table}"
            if tool_name in available_tools:
                dep_str = ", ".join(f"query_{d}" for d in deps if f"query_{d}" in available_tools)
                if dep_str:
                    dep_lines.append(f"  {tool_name} → also needs: {dep_str}")
        deps_block = "\n".join(dep_lines)

        return f"""You are a database table selector for "Krushi Ratn" — a Gujarati agricultural marketplace.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABLE CONTEXTS (what each table contains)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{table_ctx}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FK DEPENDENCIES (if you pick a table, you MUST also pick its dependencies)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{deps_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOL NAMES (return ONLY names from this list)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{tools_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELECTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. STRIP intent words first: mare, karvu, che, purchase, from, kshop — these are not product keywords
2. Find the PRIMARY table for the actual topic (product, price, video, news, order)
3. Add ALL FK dependency tables listed above for that primary table
4. CRITICAL TABLE DISAMBIGUATION — ALWAYS apply before selecting:
   ⚠️  "seeds" table = CROP SEEDS (wheat seed, cotton seed) — for farming/planting
   ⚠️  "kshop_products" table = FARMING EQUIPMENT & TOOLS (seeder machine, weeder, tractor, pump, sprayer, thresher, harvester) — for K-Shop
   - User mentions seeder/સીડર/weeder/વીડર/tractor/pump/sprayer/thresher/harvester/balwan/auger → ALWAYS kshop_products (NOT seeds)
   - User asks about crop seeds (wheat seed, bajra seed, cotton seed) → seeds table

5. Source-specific rules:
   - User says "kshop" → query_kshop_products + its dependencies
   - User says "buy sell" → query_buy_sell_products + its dependencies
   - User asks about price/bhav/mandi → query_products + query_sub_categories + query_yards + query_cities
   - General product/equipment query (no source) → BOTH query_kshop_products AND query_buy_sell_products + their deps
   - Video query → query_video_posts + its dependencies
   - News query → query_news + its dependencies
5. Return ONLY a JSON array of tool names

OUTPUT: Return ONLY a JSON array. Example:
["query_kshop_products", "query_kshop_companies", "query_kshop_categories", "query_kshop_weights"]
"""

    def _format_schema_info(self, condensed_schema: dict) -> str:
        tables = condensed_schema.get("tables", [])
        return "\n".join(
            f"  {t.get('name', '')}: {t.get('context', '')}"
            for t in tables
        )

    def _build_user_message(self, user_query: str) -> str:
        return f"""User question: "{user_query}"

Strip intent words, identify what the user actually wants, then return ALL needed tool names (primary + FK dependencies).
Return ONLY the JSON array."""

    def _parse_and_expand(self, response_content: str, available_tools: List[str]) -> List[str]:
        """Parse tool names, validate against available list, expand FK dependencies."""
        try:
            tools = json_parser.extract_tools_from_text(response_content)
            available_set = set(available_tools)

            # Normalize and validate
            validated = []
            for tool in tools:
                tool = tool.strip().strip('"\'')
                if not tool.startswith("query_"):
                    tool = f"query_{tool}"
                if tool in available_set and tool not in validated:
                    validated.append(tool)
                elif tool not in available_set:
                    logger.warning(f"⚠️ Unknown tool ignored: {tool}")

            # Expand FK dependencies for each selected table
            expanded = list(validated)
            for tool in validated:
                table = tool.replace("query_", "")
                for dep_table in FK_DEPS.get(table, []):
                    dep_tool = f"query_{dep_table}"
                    if dep_tool in available_set and dep_tool not in expanded:
                        expanded.append(dep_tool)

            logger.info(f"🔗 Tools after FK expansion: {expanded}")
            return expanded

        except Exception as e:
            logger.error_with_context(e, {"action": "parse_and_expand",
                                          "response": response_content[:200]})
            return []


# Global instance
tool_selector = ToolSelector()


def get_tool_selector() -> ToolSelector:
    return tool_selector