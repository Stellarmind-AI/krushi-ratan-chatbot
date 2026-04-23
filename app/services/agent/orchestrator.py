"""
Main Orchestrator — Production-grade, fully wired.

Pipeline per query:
  [WebSocket] ──► Route Agent ──► SQL Flow
                             ──► Navigation Flow
                             ──► General Flow
                             ──► Greeting Flow

Every step is logged with timing via StructuredLogger.
LLM always answers in ENGLISH — Sarvam translates to user language after.

Optimizations:
  • Query cache      → skip LLM entirely for repeated SQL questions
  • Intent router    → skip tool-selection LLM for obvious SQL questions
  • Pre-compiled schemas → compact per-table strings built once at startup
"""

import re
import time
from typing import List, Dict, Any, Optional

from app.services.agent.confirmation_layer import get_confirmation_layer
from app.services.agent.route_agent    import get_route_agent, FlowType
from app.services.agent.query_cache    import get_query_cache, STRIP_WORDS
from app.services.knowledge_cache      import BROWSE_KEYWORDS
from app.services.agent.intent_router  import get_intent_router
from app.services.agent.tool_selector  import get_tool_selector, build_fk_deps_from_tools
from app.services.agent.answer_generator import get_answer_generator
from app.services.agent.knowledge_handler import get_knowledge_handler
from app.services.agent.status_filter     import filter_query_results
from app.services.database.query_executor import get_query_executor
from app.services.database.query_validator import query_validator
from app.utils.schema_generator        import SchemaGenerator
from app.utils.privacy_policy          import get_privacy_policy
from app.services.llm.manager          import get_llm_manager
from app.models.chat_models            import LLMMessage
from app.core.logger                   import get_agent_logger, Timer
from app.core.config                   import settings

logger = get_agent_logger()


# ── Keyword variant map (built once, used by SQL generation) ──────────────────
# Maps a keyword (EN, Romanized, or Gujarati) → all search variants.
# Used when F1 Confirmation Layer provides a keyword_hint.
_KV_BASE: Dict[str, list] = {
    "kapas":      ["kapas", "cotton", "કપાસ"],
    "wheat":      ["wheat", "ghau", "gahu", "ઘઉં", "ઘઉ"],
    "bajra":      ["bajra", "bajri", "bajro", "બાજરો", "બાજરી"],
    "magfali":    ["magfali", "groundnut", "moongfali", "મગફળી"],
    "onion":      ["onion", "dungli", "kanda", "ડુંગળી"],
    "tomato":     ["tomato", "tameta", "ટામેટા", "ટામેટું"],
    "potato":     ["potato", "bataka", "bateta", "બટાકા", "બટેટા"],
    "garlic":     ["garlic", "lasan", "લસણ"],
    "chana":      ["chana", "ghana", "channa", "ચણા"],
    "mung":       ["mung", "moong", "મગ"],
    "jowar":      ["jowar", "jwari", "jwar", "જુવાર"],
    "corn":       ["corn", "maize", "makai", "મકાઈ"],
    "soybean":    ["soybean", "soya", "સોયાબીન"],
    "tal":        ["tal", "sesame", "તલ"],
    "chaval":     ["chaval", "rice", "ચોખા", "ડાંગર"],
    "sugarcane":  ["sugarcane", "sherdio", "શેરડી"],
    "tractor":    ["tractor", "ટ્રેક્ટર"],
    "pump":       ["pump", "motor pump", "water pump", "પંપ", "મોટર પંપ"],
    "thresher":   ["thresher", "thresar", "thraser", "થ્રેસર", "થ્રેશર"],
    "thresar":    ["thresher", "thresar", "thraser", "થ્રેસર", "થ્રેશર"],
    "sprayer":    ["sprayer", "duster", "સ્પ્રેયર", "ફ્વારો"],
    "weeder":     ["weeder", "power weeder", "વીડર"],
    "seeder":     ["seeder", "planter", "સીડર"],
}
# Expand: Gujarati script keys point to the same variant lists
KEYWORD_VARIANTS: Dict[str, list] = {}
for _en_key, _var_list in _KV_BASE.items():
    KEYWORD_VARIANTS[_en_key] = _var_list
    for _v in _var_list:
        if _v not in KEYWORD_VARIANTS:
            KEYWORD_VARIANTS[_v] = _var_list


class Orchestrator:
    """
    Central orchestrator. Wires route agent → flow handlers → answer.

    All answers returned in ENGLISH.
    Translation to user language is done by chat_handler AFTER this returns.
    """

    def __init__(self, schema_generator: SchemaGenerator):
        self.schema_generator  = schema_generator
        self.route_agent       = get_route_agent()
        self.query_cache       = get_query_cache()
        self.intent_router     = get_intent_router()
        self.tool_selector     = get_tool_selector()
        self.answer_generator  = get_answer_generator()
        self.knowledge_handler = get_knowledge_handler()
        self.query_executor    = get_query_executor()
        self.llm_manager       = get_llm_manager()
        self.privacy_policy    = get_privacy_policy()
        self.confirmation_layer = get_confirmation_layer()

        # Schema data
        self.condensed_schema: dict = {}
        self.all_tools: dict = {}
        self._virtual_tools: dict = {}
        self._compiled_schemas: Dict[str, str] = {}

        self._load_schemas()

    # ── Startup ────────────────────────────────────────────────────────────────

    def _load_schemas(self):
        logger.step("ORCHESTRATOR", "Loading schemas and tools")
        with Timer() as t:
            self.condensed_schema = self.schema_generator.load_condensed_schema()
            self.all_tools = self.schema_generator.load_all_tools()
            if not self.all_tools:
                tables = self.condensed_schema.get("tables", [])
                self._virtual_tools = {
                    t["name"]: {"table_name": t["name"], "columns": []}
                    for t in tables if "name" in t
                }
            self._build_compiled_schemas()
            # Derive FK dependency graph from tool JSON relationships and
            # install it on the tool selector. Single source of truth: tool JSONs.
            fk_deps = build_fk_deps_from_tools(self.all_tools or {})
            self.tool_selector.set_fk_deps(fk_deps)
        logger.step_done("ORCHESTRATOR LOAD", t.elapsed_ms,
                         tables=self.condensed_schema.get("total_tables"),
                         tools=len(self.all_tools) or len(self._virtual_tools),
                         compiled=len(self._compiled_schemas))

    def _build_compiled_schemas(self):
        """
        Pre-compile compact per-table schema strings ONCE at startup.
        Avoids rebuilding full schema text on every query.
        Only columns+types+joins — strips verbose descriptions.

        JOINS are rendered as ready-to-paste SQL JOIN clauses so the LLM
        does not have to translate arrow-notation into SQL syntax. Every
        clause uses <table>.<col> = <ref_table>.<ref_col> form, matching
        the alias-free style the LLM can then adapt with aliases.
        """
        for tname, tool in (self.all_tools or {}).items():
            cols = tool.get("columns", [])
            col_str = " | ".join(
                f"{c['name']}:{c['type'].split('(')[0]}"
                for c in cols
            )

            # Build explicit SQL JOIN clauses from each relationship.
            # "references" is of the form "<ref_table>.<ref_col>".
            rels = tool.get("relationships", [])
            join_lines: List[str] = []
            for r in rels:
                col = r.get("column", "")
                ref = r.get("references", "")
                if not col or not ref or "." not in ref:
                    continue
                ref_table, ref_col = ref.split(".", 1)
                join_kw = r.get("join_type", "JOIN")
                join_lines.append(
                    f"    {join_kw} {ref_table} ON {tname}.{col} = {ref_table}.{ref_col}"
                )
            joins_block = "\n".join(join_lines) if join_lines else "    (no foreign keys)"

            notes = [
                n for n in tool.get("notes", [])
                if "deleted_at" in n or "status" in n or "STRIP" in n
                   or "keyword" in n.lower() or "location" in n.lower()
            ]
            note_str = " | ".join(notes) if notes else ""

            self._compiled_schemas[tname] = (
                f"TABLE {tname}:\n"
                f"  COLUMNS: {col_str}\n"
                f"  JOINS:\n{joins_block}\n"
                + (f"  RULES: {note_str}\n" if note_str else "")
            )
        logger.info(f"📋 Pre-compiled {len(self._compiled_schemas)} table schemas")

    # ── Main entry point ────────────────────────────────────────────────────────

    async def process_query(
        self,
        user_query: str,
        target_language: Optional[str] = None,   # kept for API compat, NOT used for LLM
        confirmed_intent: Optional[str] = None,  # F1: set when user picked a clarification option
        keyword_hint: str = "",                  # F1: matched keyword for targeted SQL generation
        force_navigation: bool = False,          # Nav signal detected — skip route agent
    ) -> Dict[str, Any]:
        """
        Process user query through route agent → correct flow.

        IMPORTANT: Answer is always returned in ENGLISH.
        target_language param is ignored here — translation happens in chat_handler.

        confirmed_intent — when the F1 Confirmation Layer paused the pipeline and
        the user selected a clarification option, chat_handler passes the chosen
        intent_key here.  This causes _flow_sql to skip LLM tool-selection and use
        the pre-resolved table list directly.

        force_navigation — when the F1 nav signal detector found navigation patterns
        (કેવી રીતે, પગલાં, steps to, etc.), this forces NAVIGATION flow and skips
        the route agent entirely.  Prevents the route agent from mis-classifying
        navigation questions as SQL (e.g. "મારા ઓર્ડર ક્યાં છે?" → SQL).

        Returns dict with keys: answer, sources, query_results, selected_tools,
                                flow, cache_hit
        """
        pipeline_start = time.perf_counter()
        logger.pipeline_start(user_query)

        try:
            # ── SQL FLOW DISABLED FOR CURRENT RELEASE ──────────────────────
            # Database queries are disabled until user data security review
            # is complete. To re-enable, set ENABLE_SQL_FLOW=true in .env
            #
            # When disabled:
            #   - F1 confirmed_intent (which forces SQL) → redirects to app
            #   - Route agent classifying as SQL → redirects to app
            #   - Only NAVIGATION, GENERAL, and GREETING flows are active
            # ───────────────────────────────────────────────────────────────
            _sql_enabled = settings.is_sql_enabled

            if confirmed_intent and not _sql_enabled:
                logger.info(f"SQL FLOW DISABLED — confirmed_intent='{confirmed_intent}' blocked")
                return self._sql_disabled_response()

            # ── STEP 1: Route Agent ─────────────────────────────────────────
            # When F1 confirmed_intent is already set, the user already answered
            # the clarification question — the intent is definitively SQL data.
            # Skip route classification entirely and go straight to SQL so the
            # route agent cannot mis-classify as GENERAL/NAVIGATION.
            # Example: "wheat ni mahiti" — route agent might say GENERAL because
            # "mahiti" = "info", but the user confirmed they want DB price data.
            if confirmed_intent:
                logger.step(
                    "STEP 1 / ROUTE AGENT",
                    f"SKIPPED — F1 confirmed_intent='{confirmed_intent}' forces SQL flow",
                )
                logger.step_done("STEP 1 / ROUTE AGENT", 0, flow="SQL", reason="f1_bypass")
                result      = await self._flow_sql(user_query, confirmed_intent=confirmed_intent, keyword_hint=keyword_hint)
                result["flow"] = "SQL"
                total_ms = (time.perf_counter() - pipeline_start) * 1000
                logger.pipeline_end("SQL", total_ms, cached=result.get("cache_hit", False))
                return result

            if force_navigation:
                logger.step(
                    "STEP 1 / ROUTE AGENT",
                    "SKIPPED — nav signal detected, forcing NAVIGATION flow",
                )
                logger.step_done("STEP 1 / ROUTE AGENT", 0, flow="NAVIGATION", reason="nav_signal")
                result = await self._flow_navigation(user_query)
                result["flow"] = "NAVIGATION"
                total_ms = (time.perf_counter() - pipeline_start) * 1000
                logger.pipeline_end("NAVIGATION", total_ms)
                return result

            logger.step("STEP 1 / ROUTE AGENT", "Classifying question")
            with Timer() as t:
                flow: FlowType = await self.route_agent.classify(user_query)
            logger.step_done("STEP 1 / ROUTE AGENT", t.elapsed_ms, flow=flow)

            # ── Dispatch to correct flow ────────────────────────────────────
            if flow == "GREETING":
                result = await self._flow_greeting(user_query)

            elif flow == "NAVIGATION":
                result = await self._flow_navigation(user_query)

            elif flow == "GENERAL":
                result = await self._flow_general(user_query)

            elif flow == "SQL":
                # SQL flow routing — respects ENABLE_SQL_FLOW setting
                if not _sql_enabled:
                    # SQL flow disabled — redirect to app
                    logger.info(f"SQL FLOW DISABLED — route agent said SQL, returning redirect")
                    result = self._sql_disabled_response()
                    result["flow"] = "SQL_DISABLED"
                    total_ms = (time.perf_counter() - pipeline_start) * 1000
                    logger.pipeline_end("SQL_DISABLED", total_ms)
                    return result
                else:
                    # SQL flow enabled — execute database query
                    result = await self._flow_sql(user_query, confirmed_intent=confirmed_intent, keyword_hint=keyword_hint)
            
            else:
                # Unknown flow type — should not reach here
                logger.warning(f"Unknown flow type: {flow}")
                result = self._out_of_scope_response()

            result["flow"] = flow
            total_ms = (time.perf_counter() - pipeline_start) * 1000
            logger.pipeline_end(flow, total_ms, cached=result.get("cache_hit", False))
            return result

        except Exception as e:
            total_ms = (time.perf_counter() - pipeline_start) * 1000
            logger.error_with_context(e, {"action": "process_query",
                                          "query": user_query[:200],
                                          "elapsed_ms": f"{total_ms:.0f}"})
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": [], "query_results": [], "selected_tools": [],
                "flow": "ERROR", "error": str(e),
            }

    # ── Flow: GREETING ──────────────────────────────────────────────────────────

    async def _flow_greeting(self, query: str) -> Dict[str, Any]:
        logger.step("FLOW / GREETING", query[:60])
        with Timer() as t:
            messages = [
                LLMMessage(role="system", content=(
                    "You are a friendly AI assistant for Krushi Ratn, an agricultural "
                    "marketplace app for Gujarat farmers. The user greeted you. "
                    "Respond warmly in 2-3 English sentences and briefly mention what you can help with "
                    "(products, crop prices, farming videos, news, app navigation). "
                    "Do NOT respond in any Indian language — English only."
                )),
                LLMMessage(role="user", content=query),
            ]
            try:
                response = await self.llm_manager.generate(
                    messages=messages, temperature=0.7, max_tokens=100
                )
                answer = response.content.strip()
                logger.llm_call_done(1, "greeting", t.elapsed_ms,
                                     tokens_used=response.tokens_used or 0)
            except Exception as e:
                logger.error_with_context(e, {"action": "greeting_llm"})
                answer = ("Hello! Welcome to Krushi Ratn. "
                          "I can help with crop prices, K-Shop products, farming videos, "
                          "agricultural news, and app navigation. What would you like to know?")
        logger.step_done("FLOW / GREETING", t.elapsed_ms)
        logger.final_answer(answer, lang="en")
        return {
            "answer": answer, "sources": [], "query_results": [],
            "selected_tools": [], "is_greeting": True,
        }

    # ── Flow: NAVIGATION ────────────────────────────────────────────────────────

    async def _flow_navigation(self, query: str) -> Dict[str, Any]:
        logger.step("FLOW / NAVIGATION", query[:60])
        
        # NEW: Detect navigation intent to determine flow category
        nav_intent = self.confirmation_layer.classify_navigation_intent(query)
        
        # Map intent to relevant navigation flow IDs
        filtered_flow_ids = None
        
        if nav_intent == "crop_buy_nav":
            filtered_flow_ids = [
                "nav_company_crop_buy_flow_sell_alias",
                "nav_crop_price_determination",
            ]
            logger.info(f"🧭 NAVIGATION INTENT → crop_buy_nav | filtering to {len(filtered_flow_ids)} flows")
            
        elif nav_intent == "crop_sell_nav":
            filtered_flow_ids = [
                "nav_crop_sell",
                "nav_crop_order_cancel",
            ]
            logger.info(f"🧭 NAVIGATION INTENT → crop_sell_nav | filtering to {len(filtered_flow_ids)} flows")
            
        elif nav_intent == "kshop_buy_nav":
            filtered_flow_ids = [
                "nav_kshop_buy_product",
                "nav_kshop_view_order",
                "nav_kshop_cancel_order",
            ]
            logger.info(f"🧭 NAVIGATION INTENT → kshop_buy_nav | filtering to {len(filtered_flow_ids)} flows")
            
        elif nav_intent == "crop_price_nav":
            filtered_flow_ids = [
                "nav_home_check_mandi_yard_prices",
                "nav_yard_price_my_taluka",
            ]
            logger.info(f"🧭 NAVIGATION INTENT → crop_price_nav | filtering to {len(filtered_flow_ids)} flows")
            
        else:
            logger.info(f"🧭 NAVIGATION INTENT → unclear, using all flows")
            filtered_flow_ids = None
        
        # Pass filtered_flow_ids to Knowledge Handler
        with Timer() as t:
            answer = await self.knowledge_handler.answer_navigation(query, filtered_flow_ids=filtered_flow_ids)
        
        logger.step_done("FLOW / NAVIGATION", t.elapsed_ms, answer_chars=len(answer))
        logger.final_answer(answer, lang="en")
        return {
            "answer": answer, "sources": ["navigation.json"],
            "query_results": [], "selected_tools": [],
        }

    # ── Flow: GENERAL ───────────────────────────────────────────────────────────

    async def _flow_general(self, query: str) -> Dict[str, Any]:
        logger.step("FLOW / GENERAL", query[:60])
        with Timer() as t:
            answer = await self.knowledge_handler.answer_general(query)
        logger.step_done("FLOW / GENERAL", t.elapsed_ms, answer_chars=len(answer))
        logger.final_answer(answer, lang="en")
        return {
            "answer": answer, "sources": ["general_questions.json"],
            "query_results": [], "selected_tools": [],
        }

    # ── Flow: SQL ───────────────────────────────────────────────────────────────

    async def _flow_sql(self, query: str, confirmed_intent: Optional[str] = None, keyword_hint: str = "") -> Dict[str, Any]:
        """
        Full SQL pipeline with cache + intent routing optimizations.

        STEP 2: Cache check   → skip all LLM if SQL cached
        STEP 3: Intent route  → skip tool-selection LLM for obvious queries
        STEP 4: Tool selection (LLM #1, only if intent routing missed)
        STEP 5: SQL generation (LLM #2, compact schema)
        STEP 6: Execute SQL   (pure MySQL, no LLM)
        STEP 7: Answer gen    (LLM #3, English answer only)

        confirmed_intent — when set (F1 Confirmation Layer resolved ambiguity),
        Steps 3 and 4 are skipped entirely and the pre-resolved table list is
        used directly.  This is the only change F1 makes to this method.
        """

        # ── F1: Confirmed intent shortcut (Steps 3 & 4 bypassed) ────────────
        f1_injected_tools: Optional[list] = None
        intent_note: str = ""   # always defined; set when confirmed_intent present
        if confirmed_intent:
            from app.services.agent.confirmation_layer import get_confirmation_layer
            _cl = get_confirmation_layer()
            f1_injected_tools = _cl.get_confirmed_tables(confirmed_intent)
            f1_injected_tools = self.privacy_policy.filter_tool_names(f1_injected_tools)
            intent_note       = _cl.get_intent_note(confirmed_intent)
            if f1_injected_tools:
                logger.info(
                    f"🎯 F1 confirmed intent='{confirmed_intent}' → "
                    f"tables={f1_injected_tools} | note='{intent_note[:60]}'"
                )
            else:
                logger.warning(
                    f"⚠️ F1 confirmed_intent='{confirmed_intent}' returned no tables — "
                    f"falling through to normal tool selection"
                )

        # ── STEP 2: Cache check ─────────────────────────────────────────────
        logger.step("STEP 2 / CACHE CHECK", query[:60])
        cached = self.query_cache.get(query)
        if cached:
            queries, selected_tools = cached
            selected_tools = self.privacy_policy.filter_tool_names(selected_tools)
            all_valid, validation_errors = query_validator.validate_batch([q.get("sql", "") for q in queries])
            if not selected_tools or not all_valid:
                logger.warning(
                    f"Privacy policy rejected cached SQL; invalidating cache | "
                    f"errors={validation_errors}"
                )
                self.query_cache.invalidate(query)
                cached = None

        if cached:
            logger.cache_hit(query, selected_tools)
            stats = self.query_cache.stats
            logger.info(f"📊 Cache: {stats['hit_rate_percent']}% hit rate | "
                        f"${stats['cost_saved_usd']} saved | "
                        f"{stats['tokens_saved_estimate']} tokens saved")

            with Timer() as t:
                query_results = await self.query_executor.execute_parallel(queries)
            logger.step_done("STEP 2 / CACHE HIT EXECUTE", t.elapsed_ms,
                             results=len(query_results))

            specific = [r for r in query_results if r.row_count > 0 and self._is_specific(r)]
            if not specific:
                return self._no_data_response(selected_tools, query_results)

            return await self._generate_english_answer(query, specific, selected_tools,
                                                        cache_hit=True)

        logger.cache_miss(query)

        # ── STEP 3: Intent routing (zero-LLM table selection) ───────────────
        # Skip if F1 already resolved the intent.
        if f1_injected_tools:
            selected_tools = f1_injected_tools
            logger.step(
                "STEP 3 / INTENT ROUTER",
                "SKIPPED — F1 confirmed intent already resolved tables",
            )
            logger.step_done("STEP 3 / INTENT ROUTER", 0, result="f1_bypass")
        else:
            logger.step("STEP 3 / INTENT ROUTER", query[:60])
            with Timer() as t:
                routed = self.intent_router.route(query)
            if routed:
                selected_tools, rule_name = routed
                selected_tools = self.privacy_policy.filter_tool_names(selected_tools)
                logger.intent_routed(rule_name, selected_tools)
                logger.step_done("STEP 3 / INTENT ROUTER", t.elapsed_ms,
                                 rule=rule_name, tables=str(selected_tools))
            else:
                logger.step_done("STEP 3 / INTENT ROUTER", t.elapsed_ms, result="ambiguous")
                selected_tools = None

        # ── STEP 4: LLM Tool Selection (only if intent routing missed) ───────
        # Also skip if F1 already resolved the intent.
        if not selected_tools:
            logger.step("STEP 4 / TOOL SELECTION (LLM #1)", query[:60])
            available = self.schema_generator.get_available_tool_names()
            if not available and self._virtual_tools:
                available = [f"query_{t}" for t in sorted(self._virtual_tools.keys())]
            available = self.privacy_policy.filter_tool_names(available)

            with Timer() as t:
                tool_resp = await self.tool_selector.select_tools(
                    user_query=query,
                    condensed_schema=self.condensed_schema,
                    available_tools=available,
                )
            selected_tools = tool_resp.selected_tools
            selected_tools = self.privacy_policy.filter_tool_names(selected_tools)
            logger.step_done("STEP 4 / TOOL SELECTION", t.elapsed_ms,
                             tools=str(selected_tools))
            logger.tool_selection(selected_tools, query)

            if not selected_tools:
                logger.warning("Tool selection returned nothing — relaxed retry")
                with Timer() as t2:
                    selected_tools = await self._select_tools_relaxed(query, available)
                    selected_tools = self.privacy_policy.filter_tool_names(selected_tools)
                logger.step_done("STEP 4 / TOOL SELECTION RELAXED", t2.elapsed_ms,
                                 tools=str(selected_tools))
                if not selected_tools:
                    logger.no_data_found("SQL", query)
                    return self._out_of_scope_response()

        # ── STEP 5: SQL Generation (LLM #2) OR Exploratory Fallback ──────
        # Exploratory detection: strip all known filler/intent words AND domain
        # words derived from the selected table names. If nothing meaningful
        # remains (e.g. "what is kshop?", "what is buy sell?") → user is asking
        # a general "what is this?" question, not searching for a specific item.
        # In that case: skip the LLM entirely and generate a simple recent-records
        # fallback query for the primary table. No LLM call = no hallucinated filter.
        if self._is_exploratory_query(query, selected_tools):
            primary_table = selected_tools[0].replace("query_", "")
            compiled      = self._compiled_schemas.get(primary_table, "")
            fallback_q    = self._build_exploratory_sql(primary_table, compiled)
            queries       = [fallback_q]
            logger.info(
                f"🔍 EXPLORATORY QUERY DETECTED | skipped LLM SQL | "
                f"fallback table={primary_table} | sql={fallback_q['sql'][:80]}"
            )
        else:
            logger.step("STEP 5 / SQL GENERATION (LLM #2)", f"tables={selected_tools}")
            with Timer() as t:
                queries = await self._generate_sql_compact(
                    query, selected_tools,
                    keyword_hint=keyword_hint,
                    intent_note=intent_note,
                )
            logger.step_done("STEP 5 / SQL GENERATION", t.elapsed_ms,
                             queries_count=len(queries))

            if not queries:
                logger.warning("SQL generation returned no queries")
                return self._out_of_scope_response()

        for q in queries:
            logger.sql_generation(q.get("sql", ""), q.get("table_name", ""))

        # ── STEP 6: Execute SQL ──────────────────────────────────────────────
        logger.step("STEP 6 / SQL EXECUTION", f"{len(queries)} queries")
        with Timer() as t:
            query_results = await self.query_executor.execute_parallel(queries)
        for r in query_results:
            logger.sql_execution_done(r.table_name, r.row_count, r.execution_time * 1000)
        logger.step_done("STEP 6 / SQL EXECUTION", t.elapsed_ms,
                         total_rows=sum(r.row_count for r in query_results))

        # Filter: specific (has WHERE) vs broad
        specific = [r for r in query_results if r.row_count > 0 and self._is_specific(r)]
        any_rows = [r for r in query_results if r.row_count > 0]

        if not any_rows:
            logger.no_data_found("SQL", query)
            return self._no_data_response(selected_tools, query_results)

        if not specific and any_rows:
            logger.warning("Specific query returned 0 rows — item likely not in DB")
            return self._not_found_response(selected_tools, query_results)

        # Cache for future identical questions
        self.query_cache.set(query, queries, selected_tools)
        logger.info(f"💾 Query cached for future use | key={query[:50]}")

        # ── STEP 7: English Answer Generation (LLM #3) ──────────────────────
        return await self._generate_english_answer(
            query, specific, selected_tools,
            cache_hit=False,
            intent_note=intent_note,
            keyword_hint=keyword_hint,
        )

    # ── Step 7: Answer generator ────────────────────────────────────────────────

    async def _generate_english_answer(
        self,
        query: str,
        query_results: list,
        selected_tools: list,
        cache_hit: bool = False,
        intent_note: str = "",
        keyword_hint: str = "",
    ) -> Dict[str, Any]:
        """
        LLM call #3 — generate ENGLISH answer from DB rows.
        NO language instruction given to LLM. Always English output.
        Translation to user language is handled by chat_handler after this.

        Status filter is applied here (post-SQL, pre-LLM):
          - query_results          -> raw rows from DB (all statuses)
          - query_results_filtered -> rows with inactive/closed statuses removed
          LLM only sees filtered rows. Frontend receives both.
        """
        # ── Status filter (post-SQL, pre-LLM) ───────────────────────────────
        query_results = self.privacy_policy.sanitize_query_results(query_results)
        logger.step("STEP 7a / STATUS FILTER", f"{len(query_results)} result sets")
        filtered_results, any_filtered = filter_query_results(query_results)
        if any_filtered:
            total_raw      = sum(r.row_count for r in query_results)
            total_filtered = sum(r.row_count for r in filtered_results)
            logger.info(
                f"🔍 STATUS FILTER APPLIED | "
                f"raw_rows={total_raw} -> kept_rows={total_filtered}"
            )
        else:
            logger.info("🔍 STATUS FILTER | no inactive rows removed")

        # ── LLM answer generation (receives only active/open rows) ───────────
        logger.step("STEP 7b / ANSWER GENERATION (LLM #3)", f"{len(filtered_results)} result sets")
        with Timer() as t:
            answer_resp = await self.answer_generator.generate_answer(
                user_query=query,
                query_results=filtered_results,  # LLM only sees active rows
                has_greeting=False,
                target_language=None,   # NEVER pass language to LLM — Sarvam translates after
                intent_note=intent_note,
                keyword_hint=keyword_hint,
            )
        logger.step_done("STEP 7b / ANSWER GENERATION", t.elapsed_ms,
                         chars=len(answer_resp.answer))
        logger.final_answer(answer_resp.answer, lang="en")

        return {
            "answer":                 answer_resp.answer,
            "sources":                answer_resp.sources,
            "query_results":          query_results,          # raw — all rows from DB
            "query_results_filtered": filtered_results,       # post-status-filter — what LLM saw
            "selected_tools":         selected_tools,
            "cache_hit":              cache_hit,
        }

    # ── SQL helpers ─────────────────────────────────────────────────────────────

    async def _generate_sql_compact(self, query: str, selected_tools: List[str], keyword_hint: str = "", intent_note: str = "") -> List[Dict]:
        """Generate SQL using pre-compiled compact schemas. ~60% fewer tokens than full schema."""
        available = set(self.privacy_policy.filter_tool_names(self.schema_generator.get_available_tool_names()))
        selected_tools = self.privacy_policy.filter_tool_names(selected_tools)
        schema_lines = []
        for tool_name in selected_tools:
            if tool_name not in available:
                logger.warning(f"Tool not found: {tool_name}")
                continue
            table = tool_name.replace("query_", "")
            if table in self._compiled_schemas:
                schema_lines.append(self._compiled_schemas[table])
            else:
                logger.warning(f"No compiled schema for: {table}")

        if not schema_lines:
            return []

        compact_schema = "\n".join(schema_lines)

        # Build keyword hint block — injected into prompt when F1 resolved the intent
        # This tells the SQL generator EXACTLY what to search for, preventing it from
        # guessing the keyword from an ambiguous short query like "kapas" or "ઘઉં".
        keyword_hint_block = ""
        if keyword_hint:
            variants = KEYWORD_VARIANTS.get(keyword_hint.lower(),
                       KEYWORD_VARIANTS.get(keyword_hint, [keyword_hint]))
            variants_sql = " OR ".join(
                f"<column> LIKE '%{v}%'" for v in variants
            )
            keyword_hint_block = (
                f"\nKEYWORD HINT (from user clarification):\n"
                f"  The user is asking about: {keyword_hint!r}\n"
                f"  Search variants to use: {variants}\n"
                f"  Use LIKE with OR across all variants — e.g.: {variants_sql}\n"
                f"  Apply this search to the relevant name/title/product column.\n"
            )

        # Build intent note block — tells SQL generator exactly what domain to query
        intent_note_block = ""
        if intent_note:
            intent_note_block = f"\nINTENT CONFIRMED BY USER:\n  {intent_note}\n"

        system_prompt = f"""You are a MySQL query generator for Krushi Ratn agricultural app.

TABLES:
{compact_schema}{intent_note_block}{keyword_hint_block}

RULES:
0. ENUMERATION QUERIES — CHECK FIRST: If the user is asking for the LIST / SET of
   items the app tracks in a category (NOT a specific item by name), use SELECT DISTINCT
   on the descriptive name column with NO keyword LIKE filter on the category word.
   Trigger phrases: "list all X", "show all X", "show list of X", "what X are available",
   "what all X are there", "give me all X", "enumerate X", "kayi X uplabdh che",
   "X ni list batao", "બધા X બતાવો", "બતાવો બધા X".
   The category word itself (crops, yards, cities, products, news) is the CATEGORY, NOT
   a value to LIKE-search. Do NOT write LIKE '%crop%' or LIKE '%yard%'.
   Pattern:
     SELECT DISTINCT <name_col> AS <alias>
     FROM <primary_table> [JOINs as needed for descriptive name]
     WHERE <primary_table>.deleted_at IS NULL
     ORDER BY <name_col> ASC
     LIMIT 100;
   Examples:
     "list all crops" / "show crops in krushiratn" / "what crops are available" →
       SELECT DISTINCT sc.name AS crop
       FROM products p
       JOIN sub_categories sc ON p.subcategory_id = sc.id
       WHERE p.deleted_at IS NULL
       ORDER BY sc.name ASC LIMIT 100;
     "show all yards" / "list yards" →
       SELECT DISTINCT y.name AS yard
       FROM yards y
       WHERE y.deleted_at IS NULL
       ORDER BY y.name ASC LIMIT 100;
     "what cities does the app cover" / "list all cities" →
       SELECT DISTINCT c.name AS city
       FROM cities c
       WHERE c.deleted_at IS NULL
       ORDER BY c.name ASC LIMIT 100;
   This rule OVERRIDES rules 2 and 3 for enumeration queries — those rules apply only
   when the user names a SPECIFIC item (e.g. "kapas bhav", "balwan weeder price").
1. Strip intent words before extracting product keywords:
   intent words = mare, maro, karvu, karu, che, purchase, from, kshop, levu, joiye, apo, batao, please
2. Search each keyword INDEPENDENTLY with OR — never use full phrase LIKE
3. For product names: search both English and Gujarati script
   Examples: balwan+બલવાન, weeder+વીડર, kapas+કપાસ, pump+પંપ
4. JOINS — MANDATORY: for every primary table that has entries under its JOINS block,
   (a) copy those JOIN clauses verbatim into your FROM clause (JOIN vs LEFT JOIN exactly as shown),
   (b) in the SELECT list, return the joined tables' descriptive columns
       (e.g. yards.name, sub_categories.name, cities.name, kshop_companies.name)
       instead of raw foreign-key IDs (yard_id, subcategory_id, city_id, kshop_company_id).
   Never return a *_id column to the user when the referenced table is listed as a JOIN target.
   Short aliases are fine (p, y, sc, c, kp, kco, kc, kw, bp, u, n, vp), but every JOIN from
   the JOINS block MUST appear whenever the referenced table has a name/title/display_name column.
5. Always add: WHERE <table>.deleted_at IS NULL  (for tables that have deleted_at)
6. For kshop_products: always add AND status = 1
7. NEVER generate SELECT * without WHERE clause
8. NEVER generate a query with no WHERE clause
9. LOCATION NAMES — CRITICAL:
   a) Never use exact match (=) for location names. Always use LIKE with both English and Gujarati script.
   b) A location name can be a YARD name, a TALUKA name, or a CITY name — you cannot know which in advance.
      Yard names are typically named after the taluka or city they are in (e.g. yard "ગોંડલ" is in taluka "ગોંડલ").
   c) When filtering products/prices by location, ALWAYS join yards to BOTH cities AND talukas,
      then search ALL THREE name columns with OR:
        LEFT JOIN cities c ON y.city_id = c.id
        LEFT JOIN talukas t ON y.taluka_id = t.id
        WHERE (y.name LIKE '%LocationName%' OR c.name LIKE '%LocationName%' OR t.name LIKE '%LocationName%')
   d) Include both English and Gujarati script variants in each LIKE:
        (y.name LIKE '%Gondal%' OR y.name LIKE '%ગોંડલ%'
         OR c.name LIKE '%Gondal%' OR c.name LIKE '%ગોંડલ%'
         OR t.name LIKE '%Gondal%' OR t.name LIKE '%ગોંડલ%')
   e) NEVER filter location through only ONE of cities/talukas/yards — always search all three.
10. STATUS COLUMN — CRITICAL: NEVER add any status condition to WHERE clause for any table
    EXCEPT kshop_products (rule 6 above). Do NOT add AND status = 'active', AND status = 1,
    AND is_sold = 0, or any other status/availability filter on any other table.
    Status filtering is handled automatically after data retrieval — adding it in SQL
    will cause rows with valid non-'active' status values (e.g. sold_out, available)
    to be silently excluded from results.
11. TRANSITIVE JOINS — when a table references another table that itself has JOINs,
    chain them. For example, products→yards gives you yard_id. But yards→cities AND
    yards→talukas give you the location. So for product location queries, your FROM clause
    must include: products JOIN yards ON ... LEFT JOIN cities ON yards.city_id = cities.id
    LEFT JOIN talukas ON yards.taluka_id = talukas.id. Apply this chaining for ALL tables
    — always follow the full FK path shown in the JOINS block of each table.

OUTPUT FORMAT: Return ONLY a valid JSON array, no explanation:
[{{"table_name": "primary_table", "sql": "SELECT ... FROM ... WHERE ..."}}]"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=(
                f'Question: "{query}"\n'
                f'Strip intent words, extract product/data keywords, '
                f'write precise SQL with proper WHERE filters. Return JSON array only.'
            )),
        ]

        logger.llm_call_start(2, "sql_generation",
                              provider=self.llm_manager.current_provider,
                              est_tokens=len(compact_schema) // 4 + 200)
        with Timer() as t:
            response = await self.llm_manager.generate(
                messages=messages, temperature=0.0, max_tokens=1500
            )
        logger.llm_call_done(2, "sql_generation", t.elapsed_ms,
                             tokens_used=response.tokens_used or 0)

        return self._parse_sql_response(response.content)

    def _parse_sql_response(self, content: str) -> List[Dict]:
        """Parse LLM SQL response. Validates: must have WHERE, not bare SELECT *."""
        import re as _re
        from app.utils.json_parser import json_parser
        try:
            queries = json_parser.extract_queries_from_text(content)
            if not queries:
                queries = json_parser.safe_parse(content, default=[], expected_type=list) or []
            valid = []
            for q in (queries or []):
                if not isinstance(q, dict) or "sql" not in q:
                    continue
                sql = q["sql"].strip()
                if "WHERE" not in sql.upper():
                    logger.warning(f"Rejecting query with no WHERE: {sql[:100]}")
                    continue
                if _re.match(r"SELECT\s+\*\s+FROM\s+\w+\s*(?:LIMIT|$)", sql, _re.IGNORECASE):
                    logger.warning(f"Rejecting bare SELECT *: {sql[:80]}")
                    continue
                if "table_name" not in q:
                    m = _re.search(r"FROM\s+([a-zA-Z0-9_]+)", sql, _re.IGNORECASE)
                    q["table_name"] = m.group(1) if m else "unknown"
                valid.append(q)
                logger.debug(f"Valid SQL: {sql[:120]}")
            return valid
        except Exception as e:
            logger.error_with_context(e, {"action": "_parse_sql_response", "content": content[:200]})
            return []

    async def _select_tools_relaxed(self, query: str, available: list) -> list:
        """Fallback LLM tool selection — forces at least one table match."""
        schema_info = "\n".join(
            f"• {t.get('name')}: {t.get('context','')}"
            for t in self.condensed_schema.get("tables", [])
        )
        messages = [
            LLMMessage(role="system", content=(
                f"Pick the BEST MATCHING database table. MUST return at least one.\n"
                f"TABLES:\n{schema_info}\nAVAILABLE: {available}\n"
                f"Return ONLY a JSON array of tool names."
            )),
            LLMMessage(role="user", content=f'"{query}"\nBest matching tools:'),
        ]
        logger.llm_call_start(1, "tool_select_relaxed",
                              provider=self.llm_manager.current_provider)
        with Timer() as t:
            response = await self.llm_manager.generate(
                messages=messages, temperature=0.1, max_tokens=200
            )
        logger.llm_call_done(1, "tool_select_relaxed", t.elapsed_ms)
        from app.utils.json_parser import json_parser
        parsed = json_parser.parse_json_response(response.content)
        if isinstance(parsed, list):
            return [x for x in parsed if x in available]
        return []

    # ── Exploratory query helpers ──────────────────────────────────────────────

    @staticmethod
    def _is_exploratory_query(query: str, selected_tools: list) -> bool:
        # Return True when the query has no specific item/product keyword to
        # search for — i.e. the user is asking "what is X?" rather than
        # "find product Y in X".
        #
        # Algorithm:
        #   1. Lowercase + strip punctuation
        #   2. Build domain_words from selected_tools table name components
        #      WITH singular/plural variants (products→product, category→categories)
        #   3. Combine STRIP_WORDS + domain_words + BROWSE_KEYWORDS into exclusion set
        #      BROWSE_KEYWORDS covers category synonyms the domain_words miss:
        #      "crop" (synonym of products table), "mandi" (synonym of yards), etc.
        #   4. If no word survives exclusion (length > 1) -> exploratory
        import re as _re
        q_clean = _re.sub(r"[^\w\s]", " ", query.lower())
        words   = q_clean.split()

        # Extract component words from every selected table name
        # + singular/plural variants for robust matching
        domain_words: set = set()
        for tool in (selected_tools or []):
            table = tool.replace("query_", "").replace("_", " ")
            for w in table.split():
                if len(w) > 1:
                    domain_words.add(w)
                    # Plural → singular: "products" → "product", "categories" → "category"
                    if w.endswith("ies") and len(w) > 4:
                        domain_words.add(w[:-3] + "y")
                    elif w.endswith("s") and len(w) > 3:
                        domain_words.add(w[:-1])
                    # Singular → plural: "product" → "products"
                    if not w.endswith("s"):
                        domain_words.add(w + "s")

        exclusion = STRIP_WORDS | domain_words | BROWSE_KEYWORDS
        meaningful = [w for w in words if w not in exclusion and len(w) > 1]
        return len(meaningful) == 0

    @staticmethod
    def _build_exploratory_sql(primary_table: str, compiled_schema: str) -> dict:
        """
        Build a recent-records fallback SQL for exploratory queries.
        Fetches 10 most recent rows with NO keyword filter.

        Parses the pre-compiled schema to:
          1. Include JOIN clauses so FK columns resolve to human-readable names
          2. Replace raw _id columns with joined_table.name in SELECT
          3. Skip metadata columns (id, updated_at, deleted_at) for cleaner output
        """
        schema = compiled_schema or ""

        # ── WHERE clauses ──
        clauses = []
        if "deleted_at" in schema:
            clauses.append(f"{primary_table}.deleted_at IS NULL")
        if primary_table == "kshop_products":
            clauses.append(f"{primary_table}.status = 1")

        # ── Parse JOINS block → map FK column to (join_clause, ref_table) ──
        # Compiled schema format:
        #   JOINS:
        #     JOIN sub_categories ON products.subcategory_id = sub_categories.id
        #     LEFT JOIN yards ON products.yard_id = yards.id
        fk_map: Dict[str, str] = {}       # FK col → ref_table
        join_clauses: List[str] = []
        joins_match = re.search(r"JOINS:\n(.*?)(?:\n  [A-Z]|\Z)", schema, re.DOTALL)
        if joins_match:
            for line in joins_match.group(1).strip().splitlines():
                line = line.strip()
                if not line or line.startswith("(no"):
                    continue
                # Extract: "LEFT JOIN yards ON products.yard_id = yards.id"
                m = re.match(
                    r"((?:LEFT\s+)?JOIN)\s+(\w+)\s+ON\s+\w+\.(\w+)\s*=\s*\w+\.\w+",
                    line, re.IGNORECASE,
                )
                if m:
                    ref_table = m.group(2)
                    fk_col = m.group(3)       # e.g. "yard_id"
                    fk_map[fk_col] = ref_table
                    join_clauses.append(line)

        # ── Parse COLUMNS → build SELECT list ──
        # Skip metadata; replace FK _id cols with ref_table.name
        skip_cols = {"id", "updated_at", "deleted_at", "status"}
        select_parts: List[str] = []
        seen_refs: set = set()

        col_match = re.search(r"COLUMNS:\s*(.*)", schema)
        if col_match:
            for part in col_match.group(1).split("|"):
                col = part.strip().split(":", 1)[0].strip()
                if not col or col in skip_cols:
                    continue
                if col in fk_map:
                    ref_table = fk_map[col]
                    if ref_table not in seen_refs:
                        select_parts.append(f"{ref_table}.name AS {ref_table}_name")
                        seen_refs.add(ref_table)
                    # Don't also select the raw _id
                else:
                    select_parts.append(f"{primary_table}.{col}")

        select_cols = ", ".join(select_parts[:10]) if select_parts else f"{primary_table}.*"
        joins_sql = " ".join(join_clauses)
        where = " AND ".join(clauses) if clauses else "1=1"

        sql = (
            f"SELECT {select_cols} "
            f"FROM {primary_table} "
            + (f"{joins_sql} " if joins_sql else "")
            + f"WHERE {where} "
            f"ORDER BY {primary_table}.created_at DESC LIMIT 10"
        )
        return {"table_name": primary_table, "sql": sql}

    def _is_specific(self, result) -> bool:
        sql = getattr(result, "sql", "") or getattr(result, "query", "") or ""
        return "WHERE" in sql.upper()

    def _no_data_response(self, tools, results) -> Dict[str, Any]:
        # Tailor the message based on which tables were searched
        tools_str = ", ".join(str(t) for t in (tools or []))
        if "buy_sell" in tools_str:
            answer = (
                "I searched the marketplace but found no active listings for this item. "
                "This means no farmer has listed it for sale yet. "
                "You can list your own product by going to the Buy/Sell section in the app."
            )
        elif "kshop" in tools_str:
            answer = (
                "I searched K-Shop but found no products matching your request. "
                "Try searching with different keywords or browse all K-Shop categories in the app."
            )
        elif "products" in tools_str or "yards" in tools_str:
            answer = (
                "I searched the mandi price database but found no recent price data for this crop. "
                "Prices may not have been updated for this crop recently. "
                "Try a nearby yard name or check back later."
            )
        else:
            answer = (
                "I searched the database but couldn't find any information matching your question. "
                "Please try rephrasing or ask about a different topic."
            )
        return {
            "answer": answer,
            "sources": [r.table_name for r in results],
            "query_results": results, "selected_tools": tools,
        }

    def _not_found_response(self, tools, results) -> Dict[str, Any]:
        return {
            "answer": "I searched the database but couldn't find a specific match. The item you are looking for may not be available.",
            "sources": [r.table_name for r in results],
            "query_results": [], "selected_tools": tools,
        }

    def _out_of_scope_response(self) -> Dict[str, Any]:
        return {
            "answer": (
                "This information is not yet included in the Krushi Ratn AI chatbot. "
                "It will be available for you soon! "
                "Until then, I can help you with app navigation and understanding the app features. "
                "If you still need help, contact support through Profile → Help & Support."
            ),
            "sources": [], "query_results": [], "selected_tools": [],
            "is_out_of_scope": True,
        }

    def _sql_disabled_response(self) -> Dict[str, Any]:
        """
        Response when SQL flow is disabled (ENABLE_SQL_FLOW != true).
        Used during releases where only NAVIGATION + GENERAL + GREETING are active.
        To re-enable SQL: set ENABLE_SQL_FLOW=true in .env and restart.
        """
        return {
            "answer": (
                "This information is not yet included in the Krushi Ratn AI chatbot. "
                "It will be available for you soon! "
                "Until then, I can help you with app navigation and understanding the app features. "
                "If you still need help, contact support through Profile → Help & Support."
            ),
            "sources": [], "query_results": [], "selected_tools": [],
            "flow": "SQL_DISABLED",
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_instance: Optional[Orchestrator] = None


def initialize_orchestrator(schema_generator: SchemaGenerator) -> Orchestrator:
    global _instance
    logger.info("🚀 Initializing Orchestrator...")
    _instance = Orchestrator(schema_generator)
    logger.info("✅ Orchestrator ready")
    return _instance


def get_orchestrator() -> Orchestrator:
    if _instance is None:
        raise RuntimeError("Orchestrator not initialized. Call initialize_orchestrator() first.")
    return _instance
