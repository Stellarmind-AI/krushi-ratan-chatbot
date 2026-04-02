"""
Knowledge Handler — Answers NAVIGATION and GENERAL questions from JSON files.

Used for two flows:
  NAVIGATION → reads app/schemas/navigation.json
               answers "how to register", "where is mandi bhav screen", etc.
  GENERAL    → reads app/schemas/general_questions.json
               answers "what is krushi ratn", "is it free", "refund policy", etc.

HOW IT WORKS:
  1. Load both JSON files into memory at startup (cached in self)
  2. Auto-reload if the file has changed on disk (mtime check, no restart needed)
  3. Score every JSON entry against the user question (tag + title + word overlap)
  4. Pick top 1-3 matching entries
  5. Send matched content + question to LLM → compose clean English answer
  6. Return English answer  (chat_handler translates to Gujarati)

SCORING (updated):
  +3.0  question-title similarity bonus (prevents wrong entry winning on tie)
  +2.0  per matching tag (exact substring match in question)
  +0.5  per matching word from entry text fields (2+ char words for Gujarati support)

WHY USE LLM HERE:
  The user might ask in Gujarati, use synonyms, or ask a follow-up.
  A raw JSON entry dump would be ugly. One small LLM call (~300 tokens)
  produces a clean, conversational answer from the matched content.
  This is the ONLY LLM call in the NAVIGATION/GENERAL flow.

WHY NOT USE THE DB:
  Navigation and FAQ data is static — it doesn't live in the database.
  Putting it in JSON files means: easy to edit, no DB dependency,
  instant lookup, and the LLM doesn't need to write SQL for it.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from app.services.llm.manager  import get_llm_manager
from app.models.chat_models     import LLMMessage
from app.core.logger            import get_logger, Timer

# Lazy import to avoid circular dependency
def _get_orchestrator():
    from app.services.agent.orchestrator import get_orchestrator
    return get_orchestrator()

logger = get_logger("knowledge_handler")

# ─────────────────────────────────────────────────────────────────────────────
# JSON file paths
# ─────────────────────────────────────────────────────────────────────────────
_SCHEMAS_DIR = Path("app/schemas")
_NAV_FILE    = _SCHEMAS_DIR / "navigation.json"
_GEN_FILE    = _SCHEMAS_DIR / "general_questions.json"


def _load_json_file(path: Path) -> dict:
    """Load a JSON file. Returns empty dict on any error."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Loaded {path.name}")
        return data
    except FileNotFoundError:
        logger.error(f"❌ JSON file not found: {path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in {path}: {e}")
        return {}
    except Exception as e:
        logger.error_with_context(e, {"action": "load_json", "file": str(path)})
        return {}


def _file_mtime(path: Path) -> float:
    """Return file modification time, or 0.0 if file not found."""
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _word_overlap_score(text_a: str, text_b: str) -> float:
    """
    Count shared words between two strings.
    Uses len >= 2 (not 4) to support short Gujarati words like 'ભાવ', 'મણ'.
    Returns overlap count * 0.5.
    """
    words_a = set(w for w in text_a.split() if len(w) >= 2)
    words_b = set(w for w in text_b.split() if len(w) >= 2)
    return len(words_a & words_b) * 0.5


def _title_similarity(entry_question: str, user_question: str) -> float:
    """
    Bonus score when the entry's question title closely matches the user query.
    +3.0 if 60%+ of title words appear in user query — prevents tie-breaking
    errors when two entries have similar tags but different question titles.

    Example fix:
      User: "what languages does chatbot support"
      gq_013 title: "Can I use the app in Gujarati?"   → 0% match → 0.0
      gq_036 title: "What languages does chatbot..."   → 80% match → 3.0
      → gq_036 correctly wins
    """
    q_words     = set(w.lower() for w in user_question.split() if len(w) >= 3)
    title_words = set(w.lower().strip("?.,") for w in entry_question.split() if len(w) >= 3)
    if not title_words:
        return 0.0
    overlap_ratio = len(q_words & title_words) / len(title_words)
    return 3.0 if overlap_ratio >= 0.6 else (1.5 if overlap_ratio >= 0.4 else 0.0)


def _score_entry(entry: dict, question: str) -> float:
    """
    Score how well a JSON entry matches the user question.

    Scoring breakdown:
      +3.0  title similarity bonus (prevents wrong entry winning on ties)
      +2.0  per matching tag (exact substring match — handles EN/Romanized/Gujarati)
      +0.5  per shared word between question and entry text (len >= 2 for Gujarati)

    Returns 0.0 if no match at all.
    """
    # Use both lowercased and original question for matching
    # (Gujarati has no case so .lower() is a no-op, but English tags need it)
    q_lower    = question.lower()
    q_original = question

    score = 0.0

    # ── Title similarity bonus ────────────────────────────────────────────────
    entry_question = entry.get("question", "")
    score += _title_similarity(entry_question, question)

    # ── Tag matching ─────────────────────────────────────────────────────────
    # Each tag that appears in the question adds +2.0
    # Check both lowercased and original to handle APMC, Gujarati script, etc.
    for tag in entry.get("tags", []):
        tag_lower = tag.lower()
        if tag_lower in q_lower or tag in q_original:
            score += 2.0

    # ── Word overlap ─────────────────────────────────────────────────────────
    text_fields = []
    for key in ("question", "answer", "description", "screen",
                "how_to_reach", "how_to_use", "how_to_sell", "how_to_buy",
                "what_you_see", "tip", "statuses"):
        val = entry.get(key)
        if val and isinstance(val, str):
            text_fields.append(val)

    for key in ("sections", "sub_screens", "interactions"):
        val = entry.get(key)
        if val:
            text_fields.append(str(val))

    combined_text = " ".join(text_fields).lower()
    score += _word_overlap_score(q_lower, combined_text)

    return score


def _find_top_matches(
    entries:  List[dict],
    question: str,
    top_n:    int = 3,
) -> List[Tuple[dict, float]]:
    """
    Score all entries and return top N with score > 0, sorted descending.
    """
    scored = [(e, _score_entry(e, question)) for e in entries]
    scored = [(e, s) for e, s in scored if s > 0.0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


class KnowledgeHandler:
    """
    Answers NAVIGATION and GENERAL questions from JSON knowledge bases.
    Loaded at startup and auto-reloaded when the JSON file changes on disk.
    """

    def __init__(self):
        self.llm_manager  = get_llm_manager()
        self._nav_flows:  List[dict] = []
        self._gen_qs:     List[dict] = []
        self._nav_loaded  = False
        self._gen_loaded  = False
        # Track file modification times for auto-reload
        self._nav_mtime: float = 0.0
        self._gen_mtime: float = 0.0
        self._load_all()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_all(self):
        """Load both JSON files into memory."""
        logger.step("KNOWLEDGE HANDLER", "Loading JSON knowledge bases from disk")

        with Timer() as t:
            nav_data = _load_json_file(_NAV_FILE)
            gen_data = _load_json_file(_GEN_FILE)

        self._nav_flows  = nav_data.get("flows", [])
        self._gen_qs     = gen_data.get("questions", [])
        self._nav_loaded = len(self._nav_flows) > 0
        self._gen_loaded = len(self._gen_qs) > 0

        # Record mtimes so we can detect changes later
        self._nav_mtime  = _file_mtime(_NAV_FILE)
        self._gen_mtime  = _file_mtime(_GEN_FILE)

        logger.step_done(
            "KNOWLEDGE HANDLER LOAD",
            t.elapsed_ms,
            navigation_entries=len(self._nav_flows),
            general_entries=len(self._gen_qs)
        )

        if not self._nav_loaded:
            logger.warning(f"⚠️  navigation.json is empty or missing: {_NAV_FILE}")
        if not self._gen_loaded:
            logger.warning(f"⚠️  general_questions.json is empty or missing: {_GEN_FILE}")

    def _check_and_reload(self):
        """
        Auto-reload JSON files if they have been modified on disk since last load.
        Called at the start of every answer_navigation / answer_general call.
        This means updating general_questions.json takes effect immediately —
        no server restart required.
        """
        nav_mtime_now = _file_mtime(_NAV_FILE)
        gen_mtime_now = _file_mtime(_GEN_FILE)

        if nav_mtime_now != self._nav_mtime or gen_mtime_now != self._gen_mtime:
            logger.info(
                "🔄 JSON knowledge files changed on disk — auto-reloading..."
            )
            self._load_all()
            logger.info(
                "✅ Knowledge base auto-reloaded",
                nav=len(self._nav_flows),
                gen=len(self._gen_qs),
            )

    def reload(self):
        """
        Force-reload JSON files from disk.
        Can be called via an admin endpoint or from tests.
        Auto-reload via _check_and_reload() handles normal edits automatically.
        """
        logger.info("🔄 Force-reloading knowledge base JSONs...")
        self._load_all()
        logger.info("✅ Knowledge base reloaded",
                    nav=len(self._nav_flows), gen=len(self._gen_qs))

    # ─────────────────────────────────────────────────────────────────────────
    # Public answer methods
    # ─────────────────────────────────────────────────────────────────────────

    async def answer_navigation(self, question: str) -> str:
        """
        Answer a NAVIGATION question using navigation.json.

        DESIGN: Single LLM call with ALL navigation entries as context.
        The LLM reads every entry's screen name, description, and steps,
        then directly answers the user's question.

        WHY ONE CALL (not select → compose):
          - Keyword scoring fails for Gujarati (shared words across entries)
          - LLM selection + LLM compose = two points of failure
          - All 24 entries fit in ~1900 tokens — cheaper than two separate calls
          - ONE call = zero chance of "right entry selected, wrong answer composed"
          - Works for ANY language, ANY phrasing, with ZERO keyword maintenance

        COST: ~2000 tokens input + ~200 output = ~2200 tokens total
              vs old approach: ~700 (selector) + ~400 (composer) = ~1100 tokens
              Difference: +1100 tokens per nav query (~$0.00 at Groq free tier)
              But: 1 LLM round-trip instead of 2 → ~300ms faster
        """
        logger.step("NAVIGATION HANDLER", f"Looking up: {question[:70]}")

        # Auto-reload if file changed
        self._check_and_reload()

        if not self._nav_loaded or not self._nav_flows:
            logger.warning("Navigation data not loaded — returning fallback")
            return self._nav_fallback()

        # Build full context with ALL navigation entries
        context = self._format_nav_context(self._nav_flows)
        logger.info(f"NAV: sending all {len(self._nav_flows)} entries to single LLM call ({len(context)} chars)")

        system = (
            "You are a navigation assistant for the Krushi Ratn agricultural app.\n"
            "Below is a complete list of ALL app screens with step-by-step instructions.\n"
            "The user will ask a question in English, Gujarati, or Romanized Gujarati.\n\n"
            "FEATURES THAT EXIST IN KRUSHI RATN APP:\n"
            "- K-Shop: Buy agricultural products (equipment, seeds, supplies)\n"
            "- K-Shop Orders: View, track, and cancel K-Shop orders\n"
            "- Crop Sell: Farmers list crops for sale, receive buyer offers, confirm sale\n"
            "- Buy/Sell Marketplace: Buy or sell used items (animals, tractors, equipment)\n"
            "- Mandi Bhav / Yard Prices: Check market yard crop prices\n"
            "- Farming Videos: Watch educational farming videos\n"
            "- Agricultural News: Read farming-related news\n"
            "- AI Chatbot: Ask questions about the app\n"
            "- Profile: Update name, photo, mobile number\n"
            "- Language Settings: Change app display language\n"
            "- Switch Role: Farmer / Company / Video Creator roles\n"
            "- Video Creator: Make and upload farming videos\n"
            "- Customer Support: Contact help via Profile → Help & Support\n"
            "- Login / Register / Logout / Delete Account\n\n"
            "FEATURES THAT DO NOT EXIST IN KRUSHI RATN APP:\n"
            "- Payment system / Online payment / Payment methods / UPI / Card payment\n"
            "- Price alerts / Notifications for price changes / Alert set for crops\n"
            "- Set price for products or crops (prices come from market yard or buyer offers)\n"
            "- Weather forecast / Mausam\n"
            "- Government schemes / Yojana / Subsidy\n"
            "- Crop insurance\n"
            "- Soil testing / Soil health\n"
            "- Pest/disease identification\n"
            "- Expert chat / Agronomist consultation\n"
            "- Loan / Credit / Finance\n"
            "- Transport / Logistics / Delivery tracking\n"
            "- Geo-tagging / Farm mapping\n"
            "- Crop calendar / Sowing schedule\n\n"
            "MULTI-PART QUESTION HANDLING:\n"
            "The user sometimes asks TWO questions in one message, connected by words like\n"
            "'and', 'also', 'additionally', 'as well as', 'ane' (Gujarati), 'aur' (Hindi).\n"
            "Examples of two-part questions:\n"
            "  - 'How do I sell my old item? And how does the app help with buying and selling?'\n"
            "  - 'How do I register? Also, what features does the app have?'\n"
            "  - 'How to buy from K-Shop and how does K-Shop work?'\n"
            "When you detect two distinct questions:\n"
            "  PART 1: Answer the HOW-TO question first (exact steps from the matching screen).\n"
            "  PART 2: Then answer the general/overview question in 2-3 clear English sentences\n"
            "          based on your knowledge of Krushi Ratn from the FEATURES list above.\n"
            "  Separate the two answers with a blank line — no heading or label needed.\n\n"
            "YOUR JOB:\n"
            "1. Check if the user is asking ONE question or TWO questions.\n"
            "2. For HOW-TO questions: find the best matching screen and output its exact steps.\n"
            "3. Do NOT mix steps from different screens.\n"
            "4. Do NOT add intro sentences, tips, or explanations not in the source steps.\n"
            "5. Do NOT add a closing line like 'Let me know if you need help'.\n"
            "6. For GENERAL/OVERVIEW questions (not how-to): answer in 2-3 sentences using\n"
            "   the FEATURES list above. Do not make up features that are not listed.\n"
            "7. If the user asks about a feature that DOES NOT EXIST in the app, say exactly:\n"
            "   'This feature is not available in the Krushi Ratn app. "
            "I can help you with app navigation, K-Shop products, crop selling, "
            "buy/sell marketplace, mandi prices, farming videos, and news. "
            "If you still need help, contact support through Profile → Help & Support.'\n"
            "8. If the feature EXISTS but you cannot find matching steps, say exactly:\n"
            "   'This information is not yet included in the Krushi Ratn AI chatbot. "
            "It will be available for you soon! "
            "Until then, I can help you with app navigation and understanding the app features. "
            "If you still need help, contact support through Profile → Help & Support.'\n"
            "9. Answer in ENGLISH only — translation is handled separately."
        )

        user_message = (
            f"User Question: {question}\n\n"
            f"ALL APP SCREENS AND STEPS:\n{context}\n\n"
            f"If the question has TWO parts (how-to + general overview), answer both.\n"
            f"Otherwise find the best matching screen and output its exact steps."
        )

        messages = [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user",   content=user_message),
        ]

        logger.llm_call_start(1, "NAVIGATION_answer",
                              provider=self.llm_manager.current_provider,
                              est_tokens=len(context) // 4 + 200)
        try:
            with Timer() as t:
                response = await self.llm_manager.generate(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=900,   # raised from 600 — two-part answers need extra room
                )
            logger.llm_call_done(1, "NAVIGATION_answer", t.elapsed_ms,
                                 tokens_used=response.tokens_used or 0)

            answer = response.content.strip()

            # If LLM says feature doesn't exist or no match found, return the answer directly
            # (it's already the correct fallback message from the prompt)
            no_match_signals = ["not yet included", "don't have navigation",
                                "no screen matches", "not available in the krushi ratn",
                                "not available", "cannot find", "feature is not available"]
            if any(sig in answer.lower() for sig in no_match_signals):
                logger.info(f"NAV: LLM returned fallback message — passing through")
                return answer

            logger.final_answer(answer, lang="en")
            return answer

        except Exception as e:
            logger.error_with_context(e, {"action": "nav_answer", "query": question[:100]})
            return self._nav_fallback()

    async def answer_general(self, question: str) -> str:
        """
        Answer a GENERAL question using general_questions.json.

        Process:
          1. Auto-reload if general_questions.json was modified on disk
          2. Score all 36 FAQ entries against the question
          3. Pick top 3 matches (increased from 2 — more entries, more context needed)
          4. LLM composes a clean English answer from matched content
          5. Return English string (caller translates to Gujarati)
        """
        logger.step("GENERAL HANDLER", f"Looking up: {question[:70]}")

        # Auto-reload if file changed
        self._check_and_reload()

        if not self._gen_loaded:
            logger.warning("General QA data not loaded — returning fallback")
            return self._gen_fallback()

        with Timer() as t:
            matches = _find_top_matches(self._gen_qs, question, top_n=5)
        logger.step_done("GEN SCORING", t.elapsed_ms, matches_found=len(matches))

        # Same logic as NAV: strong keyword match → use pre-filtered, weak → send all to LLM
        top_score = matches[0][1] if matches else 0.0
        if top_score >= 4.0:
            candidates = matches
            logger.info(f"GEN: strong keyword match (score={top_score:.1f}) — using top {len(matches)}")
        else:
            candidates = [(e, 0.0) for e in self._gen_qs]
            logger.info(f"GEN: weak keyword match (score={top_score:.1f}) — sending all {len(candidates)} entries to LLM")

        if not candidates:
            logger.no_data_found("GENERAL", question)
            return self._gen_fallback()

        for entry, score in matches[:5]:
            logger.json_lookup("GENERAL", entry.get("id", "?"), score)

        # LLM picks the best entry — handles ALL languages natively
        best_entries = await self._llm_select_best(question, candidates, flow="GENERAL")

        # LLM said "no entry matches" → return universal fallback
        if not best_entries:
            logger.info("GEN: LLM found no matching entry — returning fallback")
            return self._gen_fallback()

        context = self._format_gen_context(best_entries)
        answer  = await self._compose_answer(question, context, flow="GENERAL")
        logger.final_answer(answer, lang="en")
        return answer

    # ─────────────────────────────────────────────────────────────────────────
    # Context formatters
    # ─────────────────────────────────────────────────────────────────────────

    def _format_nav_context(self, entries: List[dict]) -> str:
        parts = []
        for entry in entries:
            lines = [f"SCREEN: {entry.get('screen', '')}"]

            if entry.get("description"):
                lines.append(f"Description: {entry['description']}")
            if entry.get("how_to_reach"):
                lines.append(f"How to reach: {entry['how_to_reach']}")
            if entry.get("how_to_use"):
                lines.append(f"How to use: {entry['how_to_use']}")
            if entry.get("how_to_sell"):
                lines.append(f"How to sell: {entry['how_to_sell']}")
            if entry.get("how_to_buy"):
                lines.append(f"How to buy: {entry['how_to_buy']}")
            if entry.get("what_you_see"):
                lines.append(f"What you see: {entry['what_you_see']}")
            if entry.get("sections"):
                section_list = entry["sections"]
                if isinstance(section_list, list):
                    lines.append("Sections available:")
                    lines.extend(f"  • {s}" for s in section_list)
            if entry.get("sub_screens"):
                sub = entry["sub_screens"]
                if isinstance(sub, dict):
                    lines.append("Sub-screens:")
                    lines.extend(f"  • {k}: {v}" for k, v in sub.items())
            if entry.get("interactions"):
                lines.append(f"Actions: {entry['interactions']}")
            if entry.get("statuses"):
                lines.append(f"Order statuses: {entry['statuses']}")
            if entry.get("tip"):
                lines.append(f"Tip: {entry['tip']}")

            parts.append("\n".join(lines))

        return "\n\n---\n\n".join(parts)

    def _format_gen_context(self, entries: List[dict]) -> str:
        """Format general FAQ entries into a Q&A block for the LLM."""
        parts = []
        for entry in entries:
            q = entry.get("question", "")
            a = entry.get("answer", "")
            parts.append(f"Q: {q}\nA: {a}")
        return "\n\n".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # LLM-based entry selection — language-proof matching
    # ─────────────────────────────────────────────────────────────────────────

    async def _llm_select_best(
        self,
        question: str,
        candidates: List[Tuple[dict, float]],
        flow: str,
    ) -> List[dict]:
        """
        Use the LLM to pick the best 1-2 entries from pre-filtered candidates.

        WHY: Keyword scoring breaks for Gujarati (flexible word order, extra
        particles like 'કેવી રીતે' between key words).  The LLM understands
        ALL languages and phrasings natively — let it do the final matching.

        COST: ~100-150 tokens (just IDs + screen names + descriptions).
        This is cheaper than a wrong answer + confused user.

        Returns: list of 1-2 best-matching entry dicts.
        """
        if len(candidates) <= 1:
            return [e for e, _ in candidates]

        # Build a compact numbered list for the LLM
        options = []
        id_to_entry = {}
        for i, (entry, score) in enumerate(candidates, 1):
            eid = entry.get("id", f"entry_{i}")
            if flow == "NAVIGATION":
                label = entry.get("screen", "") or entry.get("description", "")
                desc  = entry.get("description", "")
            else:
                label = entry.get("question", "")
                desc  = entry.get("answer", "")[:100] if entry.get("answer") else ""
            options.append(f"{i}. [{eid}] {label} — {desc}")
            id_to_entry[str(i)] = entry
            id_to_entry[eid] = entry

        options_text = "\n".join(options)

        messages = [
            LLMMessage(role="system", content=(
                "You are a question matcher for the Krushi Ratn agricultural app.\n"
                "The user asked a question. Pick the BEST matching entry from the list below.\n"
                "The user may ask in English, Gujarati script, or Romanized Gujarati — handle all.\n\n"
                "RULES:\n"
                "- Return ONLY the number (e.g. '3') of the single best match.\n"
                "- If two entries are equally relevant, return both numbers comma-separated (e.g. '2,5').\n"
                "- If NO entry matches the question, return '0'.\n"
                "- Do NOT explain your choice. Just the number(s)."
            )),
            LLMMessage(role="user", content=(
                f'Question: "{question}"\n\n'
                f"Entries:\n{options_text}\n\n"
                f"Best match number(s):"
            )),
        ]

        logger.llm_call_start(0, f"{flow}_select_best",
                              provider=self.llm_manager.current_provider,
                              est_tokens=len(options_text) // 4 + 50)
        try:
            with Timer() as t:
                response = await self.llm_manager.generate(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=10,
                )
            logger.llm_call_done(0, f"{flow}_select_best", t.elapsed_ms,
                                 tokens_used=response.tokens_used or 0)

            raw = response.content.strip().rstrip(".")
            print(f"  🎯 LLM ENTRY SELECTION ({flow}): question={question!r} → picked={raw!r}", flush=True)

            # Handle "0", "none", "no match" — LLM correctly says nothing matches
            if raw.lower() in ("0", "none", "no match", "n/a") or "none" in raw.lower():
                logger.info(f"🎯 LLM says NO ENTRY matches — returning empty (will trigger fallback)")
                return []   # empty = caller handles fallback properly

            # Parse: "3" or "2,5" or "2, 5"
            selected = []
            for part in raw.replace(" ", "").split(","):
                part = part.strip()
                if part in id_to_entry:
                    selected.append(id_to_entry[part])

            if selected:
                for entry in selected:
                    eid = entry.get("id", "?")
                    label = entry.get("screen", "") or entry.get("question", "")
                    logger.info(f"🎯 LLM selected: {eid} — {label}")
                return selected

            # LLM returned something unexpected — return empty, let caller handle it
            logger.warning(f"LLM entry selection returned unexpected: {raw!r} — returning empty")
            return []

        except Exception as e:
            logger.warning(f"LLM entry selection failed: {e} — using top keyword scorers")

        # Exception fallback only: return top 1-2 by keyword score
        return [e for e, _ in candidates[:2]]

    # ─────────────────────────────────────────────────────────────────────────
    # LLM answer composition
    # ─────────────────────────────────────────────────────────────────────────

    async def _compose_answer(
        self,
        question: str,
        context:  str,
        flow:     str,
    ) -> str:
        """
        Use LLM to compose a clean, natural English answer from matched context.
        max_tokens raised to 600 — new FAQ entries have answers up to 1246 chars.
        """
        if flow == "NAVIGATION":
            system = (
                "You are a navigation assistant for the Krushi Ratn agricultural app.\n"
                "The user wants to know HOW TO USE the app or WHERE TO FIND a feature.\n\n"
                "CRITICAL RULES:\n"
                "1. Output ONLY the exact steps from the 'how_to_use' field provided below.\n"
                "2. Do NOT add, remove, or rephrase any step.\n"
                "3. Do NOT add intro sentences, extra tips, or explanations not in the source.\n"
                "4. Do NOT add a closing line like 'Let me know if you need help'.\n"
                "5. Keep every step as a numbered list exactly as written.\n"
                "6. Do NOT mention: JSON, database, API, or technical internals.\n"
                "7. Answer in ENGLISH only."
            )
        else:  # GENERAL
            system = (
                "You are a helpful assistant for the Krushi Ratn agricultural marketplace app.\n"
                "Answer the user's general question using ONLY the information provided below.\n"
                "Be friendly, concise, and accurate.\n"
                "Do NOT make up information that is not in the provided content.\n\n"
                "FEATURES THAT EXIST IN KRUSHI RATN APP:\n"
                "K-Shop, Crop Sell, Buy/Sell Marketplace, Mandi Bhav / Yard Prices, "
                "Farming Videos, Agricultural News, AI Chatbot, Profile Management, "
                "Language Settings, Role Switching (Farmer/Company/Video Creator), "
                "Customer Support, Login/Register/Logout/Delete Account.\n\n"
                "FEATURES THAT DO NOT EXIST: Payment system/Online payment/UPI/Card payment, "
                "Price alerts/Notifications for price changes, "
                "Setting price for products or crops (prices come from market yard or buyer offers), "
                "Weather forecast, Government schemes/Yojana, "
                "Crop insurance, Soil testing, Pest identification, Expert chat, "
                "Loan/Credit, Transport/Delivery tracking, Geo-tagging, Crop calendar.\n\n"
                "If the user asks about a feature that DOES NOT EXIST, say:\n"
                "'This feature is not available in the Krushi Ratn app. "
                "I can help you with app navigation, K-Shop products, crop selling, "
                "buy/sell marketplace, mandi prices, farming videos, and news. "
                "If you still need help, contact support through Profile → Help & Support.'\n\n"
                "Answer in ENGLISH only. Do not write any Gujarati or Hindi words."
            )

        user_message = (
            f"User Question: {question}\n\n"
            f"Relevant Information:\n{context}\n\n"
            f"Provide a clear, helpful English answer based only on the above information."
        )

        messages = [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user",   content=user_message),
        ]

        call_num = 1
        logger.llm_call_start(
            call_num,
            f"{flow}_compose",
            provider=self.llm_manager.current_provider,
            est_tokens=len(context) // 4 + 200
        )

        try:
            with Timer() as t:
                response = await self.llm_manager.generate(
                    messages    = messages,
                    temperature = 0.3,
                    max_tokens  = 600,   # raised from 350 — new answers up to 1246 chars
                )
            logger.llm_call_done(
                call_num,
                f"{flow}_compose",
                t.elapsed_ms,
                tokens_used=response.tokens_used or 0
            )
            answer = response.content.strip()
            logger.debug(f"LLM answer ({flow}): {answer[:100]}")
            return answer

        except Exception as e:
            logger.error_with_context(e, {
                "action": "_compose_answer",
                "flow":   flow,
                "query":  question[:100]
            })
            logger.warning("LLM compose failed — returning raw context excerpt")
            return context[:600].strip()

    # ─────────────────────────────────────────────────────────────────────────
    # Fallback responses
    # ─────────────────────────────────────────────────────────────────────────

    async def _sql_fallback(self, question: str):
        """
        When NAV or GENERAL has no matching entry, try SQL pipeline directly.
        Bypasses route agent to avoid infinite recursion.
        Returns answer string if SQL found data, else None.

        DISABLED when ENABLE_SQL_FLOW != true (no DB queries in current release).
        """
        # Check if SQL flow is enabled
        from app.core.config import settings
        _sql_enabled = settings.is_sql_enabled
        if not _sql_enabled:
            logger.info("SQL fallback SKIPPED — ENABLE_SQL_FLOW is false")
            return None

        try:
            orchestrator = _get_orchestrator()
            result  = await orchestrator._flow_sql(question)
            answer  = result.get("answer", "")
            no_data = [
                "couldn't find", "no information", "not available",
                "no data", "not found", "cannot find",
                "no active listings", "no products matching", "no listings",
                "no products found", "no results", "no matching",
                "no recent price", "not in our database",
                "may not be available", "nothing was found",
            ]
            if answer and not any(p in answer.lower() for p in no_data):
                logger.info("SQL fallback succeeded for NAV/GEN question")
                return answer
            return None
        except Exception as e:
            logger.error_with_context(e, {"action": "_sql_fallback", "query": question[:100]})
            return None

    @staticmethod
    def _nav_fallback() -> str:
        return (
            "This information is not yet included in the Krushi Ratn AI chatbot. "
            "It will be available for you soon! "
            "Until then, I can help you with app navigation and understanding the app features. "
            "If you still need help, contact support through Profile → Help & Support."
        )

    @staticmethod
    def _feature_not_in_app() -> str:
        return (
            "This feature is not available in the Krushi Ratn app. "
            "I can help you with app navigation, K-Shop products, crop selling, "
            "buy/sell marketplace, mandi prices, farming videos, and news. "
            "If you still need help, contact support through Profile → Help & Support."
        )

    @staticmethod
    def _gen_fallback() -> str:
        return (
            "This information is not yet included in the Krushi Ratn AI chatbot. "
            "It will be available for you soon! "
            "Until then, I can help you with app navigation and understanding the app features. "
            "If you still need help, contact support through Profile → Help & Support."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[KnowledgeHandler] = None


def get_knowledge_handler() -> KnowledgeHandler:
    global _instance
    if _instance is None:
        _instance = KnowledgeHandler()
    return _instance