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
# Derived at startup from each tool JSON's `relationships` field.
# Orchestrator calls set_fk_deps() after loading tools.
# Source of truth: app/schemas/tools/*.json
# ══════════════════════════════════════════════════════════════════════════════
def build_fk_deps_from_tools(tools: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Compute direct FK dependency map from loaded tool JSON files.
    For each table, lists the referenced tables (deduped, order preserved)
    from its `relationships` field. Self-references are kept.
    """
    deps: Dict[str, List[str]] = {}
    for tname, tool in (tools or {}).items():
        refs: List[str] = []
        for rel in tool.get("relationships", []):
            ref = rel.get("references", "")
            if not ref or "." not in ref:
                continue
            ref_table = ref.split(".", 1)[0]
            if ref_table and ref_table not in refs:
                refs.append(ref_table)
        if refs:
            deps[tname] = refs
    return deps

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
        # Populated by orchestrator at startup via set_fk_deps() once tool
        # JSON files are loaded. Empty until then — selection still works,
        # just without FK dependency auto-expansion.
        self.fk_deps: Dict[str, List[str]] = {}

    def set_fk_deps(self, fk_deps: Dict[str, List[str]]) -> None:
        """Install the FK dependency map (built from tool JSON relationships)."""
        self.fk_deps = dict(fk_deps or {})
        logger.info(f"🔗 FK deps installed for {len(self.fk_deps)} tables")

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
        for table, deps in self.fk_deps.items():
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
   - User asks about price/bhav/mandi/cost/₹/rupees → query_products + query_sub_categories + query_yards + query_cities (NEVER query_news)
   - General product/equipment query (no source) → BOTH query_kshop_products AND query_buy_sell_products + their deps
   - Video query → query_video_posts + its dependencies
   - News query → query_news + its dependencies ONLY when user asks "what's the news", "latest news", "samachar", "khabar" — NEVER for price/availability questions

6. CRITICAL — "PRICE OF X" IS ALWAYS A PRODUCT QUERY, NEVER NEWS:
   - "What is the price of X?" → X is a product/crop, NOT a news article
   - "How much does X cost?" → X is a product, NOT news
   - Even if X sounds like a news title (e.g. "Fertilizer Update"), the words "price", "cost",
     "bhav", "કિંમત", "ભાવ", "₹" mean the user wants product data from kshop_products or products table
   - The news table has NO price column — it's impossible to get a price from news
   - If you see ANY price indicator + unknown noun → use query_kshop_products + query_products, NEVER query_news

7. Return ONLY a JSON array of tool names

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
                for dep_table in self.fk_deps.get(table, []):
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