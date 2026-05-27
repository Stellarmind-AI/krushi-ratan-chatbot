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

from app.services.agent.route_agent    import get_route_agent, FlowType
from app.services.agent.query_cache    import get_query_cache, strip_fillers, is_filler_token
from app.services.agent.intent_router  import get_intent_router
from app.services.agent.tool_selector  import get_tool_selector, build_fk_deps_from_tools
from app.services.agent.answer_generator import get_answer_generator
from app.services.agent.knowledge_handler import get_knowledge_handler
from app.services.agent.status_filter     import filter_query_results
from app.services.database.query_executor import get_query_executor
from app.services.database.query_validator import query_validator
from app.services.entity_normalizer       import get_entity_normalizer
from app.utils.schema_generator        import SchemaGenerator
from app.utils.privacy_policy          import get_privacy_policy
from app.services.llm.manager          import get_llm_manager
from app.models.chat_models            import LLMMessage
from app.core.logger                   import get_agent_logger, Timer
from app.core.config                   import settings

logger = get_agent_logger()


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
        # Eagerly load the entity catalog at server startup so the "📚
        # EntityNormalizer loaded" log appears on boot — and so the first
        # query doesn't pay the JSON-load latency.
        self.entity_normalizer = get_entity_normalizer()

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
            # We also annotate each JOIN line with any img/image/images columns
            # present in the referenced table so the LLM knows deterministically
            # which image columns are available on each JOIN target — no guessing.
            rels = tool.get("relationships", [])
            join_lines: List[str] = []
            for r in rels:
                col = r.get("column", "")
                ref = r.get("references", "")
                if not col or not ref or "." not in ref:
                    continue
                ref_table, ref_col = ref.split(".", 1)
                join_kw = r.get("join_type", "JOIN")
                # Look up image columns on the referenced table
                ref_tool = (self.all_tools or {}).get(ref_table, {})
                ref_img_cols = [
                    c["name"] for c in ref_tool.get("columns", [])
                    if c["name"] in ("img", "image", "images")
                ]
                img_note = (
                    f"  -- img cols: {', '.join(ref_img_cols)}"
                    if ref_img_cols else ""
                )
                join_lines.append(
                    f"    {join_kw} {ref_table} ON {tname}.{col} = {ref_table}.{ref_col}{img_note}"
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
        forced_flow: Optional[str] = None,       # F1 Phase 5: explicit flow ("NAVIGATION"|"GREETING"|"GENERAL") — skip route agent
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

            # ── F1 Phase 5: forced_flow from Confirmation Layer ─────────────
            # When F1 returned skip+flow, route agent classification is redundant.
            # Dispatch directly to the matching flow handler. SQL is intentionally
            # NOT a valid forced_flow value — SQL routing always goes through
            # confirmed_intent or full route_agent classification.
            if forced_flow in ("NAVIGATION", "GREETING", "GENERAL"):
                logger.step(
                    "STEP 1 / ROUTE AGENT",
                    f"SKIPPED — F1 forced_flow='{forced_flow}' bypasses route agent",
                )
                logger.step_done("STEP 1 / ROUTE AGENT", 0, flow=forced_flow, reason="f1_forced_flow")
                if forced_flow == "GREETING":
                    result = await self._flow_greeting(user_query)
                elif forced_flow == "GENERAL":
                    result = await self._flow_general(user_query)
                else:
                    result = await self._flow_navigation(user_query)
                result["flow"] = forced_flow
                total_ms = (time.perf_counter() - pipeline_start) * 1000
                logger.pipeline_end(forced_flow, total_ms)
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
        with Timer() as t:
            answer = await self.knowledge_handler.answer_navigation(query)
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

        # ── Entity normalization (intent + keyword → canonical Gujarati) ────
        # Runs BEFORE SQL generation so the prompt can inject the exact
        # Gujarati value the DB stores, eliminating LLM hallucination of
        # English/Hindi/Romanized LIKE variants.
        gujarati_keyword: str = ""
        if confirmed_intent and keyword_hint:
            resolved = self.entity_normalizer.normalize(confirmed_intent, keyword_hint)
            if resolved:
                gujarati_keyword = resolved
                logger.info(
                    f"🔤 normalize({confirmed_intent}, {keyword_hint!r}) → "
                    f"canonical={gujarati_keyword!r}"
                )
            else:
                logger.info(
                    f"🔤 normalize({confirmed_intent}, {keyword_hint!r}) → no match — "
                    f"SQL prompt will translate to Gujarati"
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
                    gujarati_keyword=gujarati_keyword,
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

    async def _generate_sql_compact(
        self,
        query: str,
        selected_tools: List[str],
        keyword_hint: str = "",
        intent_note: str = "",
        gujarati_keyword: str = "",
    ) -> List[Dict]:
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

        # Inject example queries for specific tables so the LLM has a concrete
        # reference for both category-subquery filtering and correct img column usage.
        # These examples are stored in the tool JSON but NOT in _compiled_schemas.
        # "products" is included here so the LLM sees sc.img AS crop_img in context.
        _TABLES_WITH_INJECTED_EXAMPLES = {"buy_sell_products", "kshop_products", "products"}
        example_lines: List[str] = []
        for tool_name in selected_tools:
            table = tool_name.replace("query_", "")
            if table not in _TABLES_WITH_INJECTED_EXAMPLES:
                continue
            try:
                tool_json = self.schema_generator.load_tool(table)
                examples = tool_json.get("example_queries", [])
                if examples:
                    example_lines.append(f"\nEXAMPLE QUERY for {table} (use category subquery — DO NOT deviate):")
                    for ex in examples[:1]:   # one example is enough
                        example_lines.append(f"  {ex}")
            except Exception:
                pass
        if example_lines:
            compact_schema += "\n" + "\n".join(example_lines) + "\n"

        # Build Gujarati-only keyword block.
        #
        # The DB stores names in GUJARATI script exclusively. The entity
        # normalizer (called in _flow_sql) resolves the user's keyword to
        # its canonical Gujarati value when one exists in the catalog.
        #
        # Three cases:
        #   (A) gujarati_keyword set     → catalog hit — inject the exact value;
        #                                  the LLM uses it verbatim, no variants.
        #   (B) keyword_hint set, no hit → ask the LLM to translate the user's
        #                                  keyword to Gujarati script and use
        #                                  ONLY the Gujarati form.
        #   (C) neither set              → no F1 keyword; rule 4 of the system
        #                                  prompt instructs the LLM to extract
        #                                  the keyword and translate to Gujarati.
        #
        # All three paths converge on the same outcome: every LIKE clause in
        # the generated SQL contains Gujarati script only — no English,
        # Romanized, or Hindi forms.
        keyword_hint_block = ""
        if gujarati_keyword:
            keyword_hint_block = (
                "\nCANONICAL GUJARATI KEYWORD (resolved from the catalog — use verbatim):\n"
                f"  '{gujarati_keyword}'\n"
                "  Rules:\n"
                "    • Use this EXACT Gujarati value in every LIKE clause for the keyword.\n"
                "    • Do NOT add English, Romanized, Hindi, or alternate Gujarati variants.\n"
                "    • Apply via the category subquery for buy_sell_products / kshop_products\n"
                "      (per rule 12); for all other tables apply LIKE to the relevant\n"
                "      name/title column directly.\n"
                f"    • Form: <column> LIKE '%{gujarati_keyword}%'\n"
            )
        elif keyword_hint:
            keyword_hint_block = (
                "\nTRANSLATE KEYWORD (the catalog had no entry for this keyword):\n"
                f"  User keyword: '{keyword_hint}'\n"
                "  Rules:\n"
                "    • Translate the keyword to GUJARATI SCRIPT and use ONLY that\n"
                "      single Gujarati form inside the LIKE clause.\n"
                "    • Do NOT include the English, Romanized, or Hindi form.\n"
                "    • Apply via the category subquery for buy_sell_products / kshop_products\n"
                "      (per rule 12); for all other tables apply LIKE to the relevant\n"
                "      name/title column directly.\n"
                "    • Form: <column> LIKE '%<gujarati>%'\n"
            )

        # Build intent note block — tells SQL generator exactly what domain to query
        intent_note_block = ""
        if intent_note:
            intent_note_block = f"\nINTENT CONFIRMED BY USER:\n  {intent_note}\n"

        # Conditional rule blocks — only include when relevant tables are selected.
        _has_buy_sell = "buy_sell_products" in selected_tools or "query_buy_sell_products" in selected_tools
        _has_kshop    = "kshop_products" in selected_tools or "query_kshop_products" in selected_tools

        category_rule_block = ""
        if _has_buy_sell or _has_kshop:
            parts = []
            if _has_buy_sell:
                parts.append(
                    "  • buy_sell_products: NEVER filter on bp.product_name (user-entered, unreliable). Use:\n"
                    "      bp.category_id IN (SELECT id FROM buy_sell_categories\n"
                    "                          WHERE name LIKE '%<gujarati>%' AND deleted_at IS NULL)"
                )
            if _has_kshop:
                parts.append(
                    "  • kshop_products: NEVER filter on kp.name (user-entered, unreliable). Use:\n"
                    "      kp.kshop_category_id IN (SELECT id FROM kshop_categories\n"
                    "                                WHERE name LIKE '%<gujarati>%' AND deleted_at IS NULL)"
                )
            category_rule_block = (
                "12. CATEGORY-BASED FILTERING (ABSOLUTE — no exceptions):\n"
                + "\n".join(parts) + "\n"
                "    The <gujarati> placeholder is the single Gujarati value from rule 4\n"
                "    (CANONICAL GUJARATI KEYWORD if provided, otherwise the translated\n"
                "    Gujarati form). Use ONE LIKE clause per subquery — no English/\n"
                "    Romanized/Hindi alternatives.\n"
            )

        image_rule_block = ""
        if _has_buy_sell or _has_kshop:
            img_lines = ["13. IMAGE COLUMNS (frontend-required aliases — exact match):"]
            if _has_buy_sell:
                img_lines.append(
                    "  • buy_sell_products: SELECT JSON_EXTRACT(bp.form_data, '$.Images') AS product_images,\n"
                    "    bc.image AS category_image. Do NOT select bp.images (legacy)."
                )
            if _has_kshop:
                img_lines.append(
                    "  • kshop_products: kshop_products has NO direct image column. JOIN media via mediables:\n"
                    "      LEFT JOIN mediables mb ON mb.mediable_id = kp.id AND mb.mediable_type LIKE 'App%KshopProduct'\n"
                    "      LEFT JOIN media m ON m.id = mb.media_id\n"
                    "    SELECT CONCAT(m.filename, '.', m.extension) AS product_image, kc.img AS category_img.\n"
                    "    Use LIKE 'App%KshopProduct' (NEVER `=` with backslash literal — escape layers break it)."
                )
            img_lines.append(
                "  • Other JOINed tables: project img/image columns with suffix alias\n"
                "    (e.g. sc.img AS crop_img, bc.image AS category_image)."
            )
            image_rule_block = "\n".join(img_lines) + "\n"

        # ── PREFIX-CACHE LAYOUT ────────────────────────────────────────────
        # System prompt is kept BYTE-IDENTICAL across queries (modulo 4 fixed
        # combinations of the conditional category/image blocks). Per-query
        # dynamic data (compact_schema, intent_note, keyword_hint, user query)
        # is moved to the user message so OpenAI's automatic prefix cache can
        # hit on the system prefix and discount it after the first call.
        system_prompt = f"""You are a MySQL query generator for Krushi Ratn agricultural app.

RULES:
0. SQL FORMATTING:
   a) SINGLE-LINE SQL — entire SQL value is ONE continuous string. No literal newlines,
      no backslashes, no `+` concatenation in JSON. CORRECT example:
      "sql": "SELECT p.id FROM products p WHERE p.deleted_at IS NULL"
   b) TABLE QUALIFICATION — every SELECT column qualified with table/alias when JOINs present.
      INCORRECT: SELECT id, title FROM news JOIN states ...
      CORRECT:   SELECT n.id, n.title FROM news n JOIN states s ...
      Aggregates (COUNT(*), SUM(col)) may be unqualified if column is unique.

1. AGGREGATE / LIST QUERIES — CHECK INTENT FIRST:
   The user is asking either for a NUMBER (count) or for the LIST itself. Pick the right SQL shape:

   a) COUNT / QUANTITY question — user wants a NUMBER (must use aggregate, NOT a list):
      Triggers (any language): "how many", "how much" (no price word), "count of",
      "total of", "number of", "total number", "ketla", "ketli", "ketlu", "ketle",
      "kul ketla", "ganatari", "sankhya", "kitne", "kitni", "kitna", "kul kitne",
      "કેટલા", "કેટલી", "કેટલું", "કુલ", "ગણતરી", "સંખ્યા",
      "कितने", "कितनी", "कितना", "कुल", "संख्या" — and any natural farmer
      phrasing including mixed/broken grammar.

      STEP 1 — IDENTIFY THE ENTITY:
        Before writing SQL, identify the THING the user is counting. The entity is
        usually the LAST noun of the count phrase, NOT every noun in the query.
        Examples (entity in bold):
          "how many product **categories** in buy/sell"     → entity = category
          "how many **products** in kshop"                  → entity = product
          "how many **talukas** in bhavnagar"               → entity = taluka
          "how many **companies** in kshop"                 → entity = company
          "how many **crops** are tracked"                  → entity = crop
          "how many **cities** does the app cover"          → entity = city
          "how many **videos** in app"                      → entity = video

      STEP 2 — PICK THE TABLE WHOSE ROWS *ARE* THAT ENTITY:
        Each table is the canonical home of ONE entity. Use the table whose rows
        directly represent the entity — NOT a transactional table that mentions
        the entity via a foreign key.
          category   → buy_sell_categories  / kshop_categories  / categories  / sub_categories
                       (pick the one matching the section — buy/sell vs kshop vs general)
          product    → buy_sell_products    / kshop_products
          taluka     → talukas               city → cities       state → states
          company    → kshop_companies       yard → yards
          crop       → sub_categories        seed → seeds        news → news
          video      → video_posts           video_category → video_categories
        ⚠️ COMMON MISTAKE — DO NOT pick the transactional table just because the
        query contains its keyword. "how many product categories" must use
        buy_sell_categories or kshop_categories (entity = category), NOT
        buy_sell_products / kshop_products. "how many companies" must use
        kshop_companies, NOT kshop_products.

      STEP 3 — WRITE COUNT SQL:
        Default pattern (one row = one entity):
          SELECT COUNT(*) AS count FROM <entity_table> [JOINs] WHERE <entity_table>.deleted_at IS NULL [AND <filter>]
        Use COUNT(DISTINCT fk_col) ONLY when the user's count must come from a
        TRANSACTIONAL table that has many rows per entity:
          "how many DIFFERENT crops have price data" — the entity (crop) lives
          in sub_categories, but the user is asking about presence in products:
            SELECT COUNT(DISTINCT p.subcategory_id) AS count FROM products p WHERE p.deleted_at IS NULL
          "how many companies have listings in kshop":
            SELECT COUNT(DISTINCT kp.kshop_company_id) AS count FROM kshop_products kp WHERE kp.deleted_at IS NULL AND kp.status = 1
        When entity word "DIFFERENT" / "UNIQUE" / "જુદા" appears in the question,
        prefer COUNT(DISTINCT col). Otherwise prefer COUNT(*) on the entity table.

      All other rules still apply: kshop_products still adds `AND status = 1`; location
      queries still use the three-way OR (cities/talukas/yards); soft-delete still applied;
      the single Gujarati keyword from rule 4 is still applied inside any LIKE clause.
      Examples:
        "how many products in kshop" (entity=product, table=kshop_products)
          → SELECT COUNT(*) AS count FROM kshop_products kp WHERE kp.deleted_at IS NULL AND kp.status = 1
        "how many talukas in bhavnagar" (entity=taluka, table=talukas)
          → SELECT COUNT(*) AS count FROM talukas t LEFT JOIN cities c ON t.city_id = c.id WHERE t.deleted_at IS NULL AND c.name LIKE '%ભાવનગર%'
        "ketla product category avillabel che buy sell ma" (entity=category, table=buy_sell_categories — NOT buy_sell_products)
          → SELECT COUNT(*) AS count FROM buy_sell_categories bc WHERE bc.deleted_at IS NULL
        "how many companies sell in kshop" (entity=company, table=kshop_companies — NOT kshop_products)
          → SELECT COUNT(*) AS count FROM kshop_companies kc WHERE kc.deleted_at IS NULL
        "kitne different crops" (entity=crop with "different" → DISTINCT, table=products via subcategory_id)
          → SELECT COUNT(DISTINCT p.subcategory_id) AS count FROM products p WHERE p.deleted_at IS NULL
        "kitne videos in app" (entity=video, table=video_posts)
          → SELECT COUNT(*) AS count FROM video_posts vp WHERE vp.deleted_at IS NULL

   b) LIST / ENUMERATION question — user wants the SET of items themselves:
      Triggers: "list all X", "show all X", "what X are available", "બધા X બતાવો",
      "kayi X uplabdh che", "give me all X", "enumerate X".
      Use SELECT DISTINCT on the descriptive name column with NO LIKE filter on category word:
        SELECT DISTINCT <name_col> AS <alias> FROM <table> [JOINs]
        WHERE deleted_at IS NULL ORDER BY <name_col> ASC LIMIT 100
      Example: "list all crops"
          → SELECT DISTINCT sc.name AS crop FROM products p JOIN sub_categories sc ON p.subcategory_id = sc.id WHERE p.deleted_at IS NULL ORDER BY sc.name LIMIT 100

   c) Both 1a and 1b OVERRIDE rules 3 and 4 (keyword search) — those apply ONLY when the
      user names a SPECIFIC item (e.g. "kapas bhav", "balwan weeder price"). When the user
      asks "how many" or "list all", do NOT add LIKE keyword filters on the category word
      itself (don't write LIKE '%product%' or LIKE '%crop%').

2. Strip intent words from query before extracting keywords:
   mare, maro, karvu, karu, che, purchase, from, levu, joiye, apo, batao, please.

3. Search each keyword INDEPENDENTLY with OR — never full-phrase LIKE.

4. KEYWORD SEARCH — GUJARATI SCRIPT ONLY (the DB stores Gujarati exclusively):
   The user message will contain ONE of these directives (or neither):
   a) "CANONICAL GUJARATI KEYWORD: '<value>'" — use that exact Gujarati value
      verbatim inside the LIKE clause. Do NOT add English, Romanized, Hindi,
      or alternate Gujarati variants. Form: <column> LIKE '%<value>%'.
   b) "TRANSLATE KEYWORD: '<keyword>'" — translate the keyword to Gujarati
      script and use ONLY that single Gujarati form in the LIKE clause.
      Do NOT include the English, Romanized, or Hindi form.
   c) If neither directive is present, extract the keyword from the question,
      translate it to Gujarati script, and use ONLY the Gujarati form.

   Translation rules — when you must translate (paths b and c):
     • Input may be in English, Romanized Gujarati, Hindi (Devanagari script),
       or Romanized Hindi. In every case, output a SINGLE Gujarati-script form.
     • Worked examples — translate the LEFT form to the RIGHT form before
       emitting the LIKE clause:
         "काले तिल"  (Hindi script)        → કાળા તલ
         "kale til"  (Romanized Hindi)     → કાળા તલ
         "black sesame" (English)          → કાળા તલ
         "तिल" / "til" / "sesame"          → તલ
         "कपास" / "kapas" / "cotton"       → કપાસ
         "गेहूं" / "ghau" / "wheat"        → ઘઉં
         "प्याज" / "kanda" / "onion"       → ડુંગળી   (NOT કાંદા)
         "टमाटर" / "tomato"               → ટામેટા
         "मूंगफली" / "moongfali" / "groundnut" → મગફળી
         "राजकोट" / "rajkot"              → રાજકોટ
         "भावनगर" / "bhavnagar"           → ભાવનગર
     • NEVER leave Devanagari (Hindi) script or Latin (English/Romanized)
       characters inside the LIKE clause. A LIKE clause that contains anything
       other than Gujarati-script characters between the % marks is WRONG.

   In every case: a LIKE clause for a keyword must contain ONE Gujarati form
   and nothing else. The DB has no English/Romanized/Hindi rows to match.

5. JOINS — for every primary table, copy its JOINS block verbatim (JOIN vs LEFT JOIN
   exactly as shown). In SELECT, return joined tables' descriptive columns (yards.name,
   sub_categories.name, cities.name, kshop_companies.name) NOT raw *_id columns. Short
   aliases (p, y, sc, c, kp, kco, kc, kw, bp, u, n, vp) are fine. Chain transitively —
   e.g. products→yards→{{cities,talukas}} for location, follow the full FK path.

6. Always add WHERE <table>.deleted_at IS NULL (for tables that have deleted_at).

7. STATUS COLUMN — kshop_products is the ONLY table where SQL adds a status filter
   (AND kp.status = 1). NEVER add status/is_sold/availability conditions on any other
   table — post-retrieval filter handles them. Adding them in SQL silently drops valid rows.

8. NEVER SELECT * without WHERE. NEVER generate a query without WHERE.

9. LOCATION NAMES — a location may be yard, taluka, OR city name (often shared).
   When filtering by location, ALWAYS LEFT JOIN cities AND talukas via yards, then
   search all three name columns with OR, using GUJARATI SCRIPT ONLY:
     LEFT JOIN cities c ON y.city_id = c.id
     LEFT JOIN talukas t ON y.taluka_id = t.id
     WHERE (y.name LIKE '%<gujarati>%' OR c.name LIKE '%<gujarati>%' OR t.name LIKE '%<gujarati>%')
   The location keyword must be in Gujarati script — use the canonical value from
   rule 4(a) if provided, otherwise translate the location name to Gujarati per
   rule 4(b)/4(c). Do NOT add the English/Romanized form to the LIKE clauses.
   Never use `=` for location names. Never filter on only one of yards/cities/talukas.

{category_rule_block}{image_rule_block}OUTPUT FORMAT: Return ONLY a valid JSON array, no explanation:
[{{"table_name": "primary_table", "sql": "SELECT ... FROM ... WHERE ..."}}]"""

        # User message carries ALL per-query dynamic content (schema, intent,
        # keyword hints, query) — see PREFIX-CACHE LAYOUT comment above.
        user_content = (
            f"TABLES:\n{compact_schema}"
            f"{intent_note_block}{keyword_hint_block}"
            f'\nQuestion: "{query}"\n'
            f'Strip intent words, extract product/data keywords, '
            f'write precise SQL with proper WHERE filters. Return JSON array only.'
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_content),
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
        """Parse LLM SQL response. Sanitizes, validates: must have WHERE, not bare SELECT *."""
        import re as _re
        from app.utils.json_parser import json_parser, sql_sanitizer
        try:
            queries = json_parser.extract_queries_from_text(content)
            if not queries:
                queries = json_parser.safe_parse(content, default=[], expected_type=list) or []
            valid = []
            for q in (queries or []):
                if not isinstance(q, dict) or "sql" not in q:
                    continue
                
                # ── SANITIZE: Remove LLM-generated noise (backslashes, missing prefixes) ─
                sql = sql_sanitizer.sanitize_sql(q["sql"].strip())
                q["sql"] = sql
                
                # Validation checks
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
                logger.debug(f"Valid SQL (sanitized): {sql[:120]}")
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
        """
        True when the query has no specific search keyword — user is asking
        "what is X?" / "how do I do Y?" rather than naming a real item.

        Delegates filler detection to `strip_fillers` (regex + fuzzy pipeline),
        then additionally strips domain words derived from the selected table
        names (which are context-specific and not global fillers).

        Examples (expected outcomes):
          "how i sell any product"   → []                   → exploratory
          "what is kshop"            → []                   → exploratory
          "list all yards"           → []                   → exploratory (CATEGORY root)
          "show me prodct"           → []                   → exploratory (typo handled)
          "kapas bhav"               → ["kapas"]            → specific search
          "balwan weeder price"      → ["balwan", "weeder"] → specific search
        """
        survivors = strip_fillers(query)
        if not survivors:
            return True

        # Strip domain words derived from the selected table names — these
        # are context-specific ("buy_sell_products" → "buy", "sell", "products")
        # and shouldn't be promoted to search targets.  Each component runs
        # through is_filler_token so inflections and typos are handled the
        # same way as global fillers.
        domain_words: set = set()
        for tool in (selected_tools or []):
            table = tool.replace("query_", "").replace("_", " ")
            for w in table.split():
                if len(w) > 1:
                    domain_words.add(w)

        meaningful = [
            w for w in survivors
            if w not in domain_words and not is_filler_token(w)
        ]
        return len(meaningful) == 0

    @staticmethod
    def _build_exploratory_sql(primary_table: str, compiled_schema: str) -> dict:
        """
        Build a recent-records fallback SQL for exploratory queries.
        Fetches 10 most recent rows with NO keyword filter.

        Parses the pre-compiled schema to:
          1. Include JOIN clauses so FK columns resolve to human-readable names
          2. Replace raw _id columns with joined_table.name in SELECT
          3. Project image data using table-specific patterns (Rule 13):
             - buy_sell_products → JSON_EXTRACT(form_data, '$.Images')
                 (the standalone `images` column is legacy and skipped)
             - kshop_products    → LEFT JOIN mediables (polymorphic,
                 mediable_type LIKE 'App%KshopProduct') + media,
                 then CONCAT(filename, '.', extension) AS product_image
             - Joined category/sub-category tables annotated with
                 "-- img cols: X" by _build_compiled_schemas → projected
                 with deterministic alias (e.g. bc.image AS category_image,
                 kc.img AS category_img, sc.img AS sub_categories_img)
          4. Skip metadata columns (id, updated_at, deleted_at, status) for
             cleaner output

        SELECT projection uses priority-based bucketing — image columns are
        always included regardless of column count. Capping the SELECT list
        was previously dropping img cols silently when wide tables (e.g.
        buy_sell_products) had more non-skip cols than the cap.
        """
        schema = compiled_schema or ""

        # ── WHERE clauses ──
        clauses = []
        if "deleted_at" in schema:
            clauses.append(f"{primary_table}.deleted_at IS NULL")
        if primary_table == "kshop_products":
            clauses.append(f"{primary_table}.status = 1")

        # ── Parse JOINS block → map FK column to ref_table; also capture
        # the "-- img cols: X" annotation per ref_table so we can project
        # joined-table image columns alongside the joined-table name.
        # Compiled schema format:
        #   JOINS:
        #     JOIN sub_categories ON products.subcategory_id = sub_categories.id  -- img cols: img
        #     LEFT JOIN yards ON products.yard_id = yards.id
        fk_map: Dict[str, str] = {}                  # FK col → ref_table
        img_cols_by_ref: Dict[str, List[str]] = {}   # ref_table → ["img"] / ["image"] / ...
        join_clauses: List[str] = []
        joins_match = re.search(r"JOINS:\n(.*?)(?:\n  [A-Z]|\Z)", schema, re.DOTALL)
        if joins_match:
            for line in joins_match.group(1).strip().splitlines():
                line = line.strip()
                if not line or line.startswith("(no"):
                    continue
                # Capture img-cols annotation BEFORE stripping the comment.
                # _build_compiled_schemas appends "  -- img cols: X" to JOIN
                # lines so the LLM can see which joined tables have image
                # columns. We use the same annotation here to project those
                # columns — closing the gap between the LLM and exploratory
                # SQL paths.
                img_match = re.search(r"--\s*img cols:\s*(.+?)\s*$", line)
                line_img_cols = (
                    [c.strip() for c in img_match.group(1).split(",") if c.strip()]
                    if img_match else []
                )
                # Strip any inline SQL comment before using the line in actual SQL.
                # The annotations are LLM-context-only — they must never reach the
                # SQL executor because MySQL treats "-- " as a line comment, which
                # would comment out everything after it (including WHERE and ORDER BY).
                clean_line = re.sub(r'\s*--.*$', '', line).strip()
                # Extract: "LEFT JOIN yards ON products.yard_id = yards.id"
                m = re.match(
                    r"((?:LEFT\s+)?JOIN)\s+(\w+)\s+ON\s+\w+\.(\w+)\s*=\s*\w+\.\w+",
                    clean_line, re.IGNORECASE,
                )
                if m:
                    ref_table = m.group(2)
                    fk_col = m.group(3)       # e.g. "yard_id"
                    fk_map[fk_col] = ref_table
                    if line_img_cols:
                        img_cols_by_ref[ref_table] = line_img_cols
                    join_clauses.append(clean_line)  # always use comment-stripped line

        # ── Parse COLUMNS → bucket-based SELECT list ──
        # Three buckets — priority and image are MANDATORY (never trimmed),
        # other fills remaining slots up to MAX_COLUMNS:
        #   priority — display, price, time, quantity, joined-table names
        #   image    — img/image/images on primary OR joined tables
        #   other    — everything else not in skip_cols
        skip_cols = {"id", "updated_at", "deleted_at", "status"}
        img_col_names = {"img", "image", "images"}
        priority_col_names = {
            # Display / identifier
            "name", "product_name", "title", "description",
            # Price / numeric
            "price", "min_price", "max_price", "discount_price", "price_date",
            # Quantity / metric
            "quantity_available", "weight_value", "views_count",
            # Time
            "created_at",
        }
        MAX_COLUMNS = 12

        priority_parts: List[str] = []
        image_parts:    List[str] = []
        other_parts:    List[str] = []
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
                        # Joined-table descriptive name → priority bucket
                        priority_parts.append(
                            f"{ref_table}.name AS {ref_table}_name"
                        )
                        # Joined-table image columns → image bucket (mandatory)
                        # Deterministic alias: <ref_table>_<col> (e.g. sub_categories_img,
                        # buy_sell_categories_image). Frontend scans for any column
                        # ending in img/image/images regardless of alias prefix.
                        for img_col in img_cols_by_ref.get(ref_table, []):
                            image_parts.append(
                                f"{ref_table}.{img_col} AS {ref_table}_{img_col}"
                            )
                        seen_refs.add(ref_table)
                    # Don't also select the raw _id
                elif col in img_col_names:
                    # Skip primary-table img/image/images columns — image
                    # extraction for product tables is handled by the
                    # table-specific patterns below (form_data JSON for
                    # buy_sell_products, mediables+media JOIN for kshop_products).
                    # The legacy `images` column on buy_sell_products is unused.
                    continue
                elif col in priority_col_names:
                    priority_parts.append(f"{primary_table}.{col}")
                else:
                    other_parts.append(f"{primary_table}.{col}")

        # ── Table-specific PRIMARY image patterns ──────────────────────────
        # buy_sell_products → product_images come from form_data JSON column.
        # kshop_products    → product_image via mediables (polymorphic) → media.
        # Both go into the image bucket (mandatory, never trimmed).
        if primary_table == "buy_sell_products":
            image_parts.append(
                "JSON_EXTRACT(buy_sell_products.form_data, '$.Images') AS product_images"
            )
        elif primary_table == "kshop_products":
            join_clauses.append(
                "LEFT JOIN mediables ON mediables.mediable_id = kshop_products.id "
                "AND mediables.mediable_type LIKE 'App%KshopProduct'"
            )
            join_clauses.append(
                "LEFT JOIN media ON media.id = mediables.media_id"
            )
            image_parts.append(
                "CONCAT(media.filename, '.', media.extension) AS product_image"
            )

        # Priority + image always included. Other fills remaining slots.
        select_parts: List[str] = priority_parts + image_parts
        remaining = MAX_COLUMNS - len(select_parts)
        if remaining > 0:
            select_parts.extend(other_parts[:remaining])

        select_cols = ", ".join(select_parts) if select_parts else f"{primary_table}.*"
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