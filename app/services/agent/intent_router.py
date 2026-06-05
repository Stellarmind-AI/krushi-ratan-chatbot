"""
Intent Router — NEUTRALIZED (no-op).

HISTORY:
  Originally this file contained keyword-based pre-routing (a long list of
  "if query contains 'kapas' → crop_price, if 'weeder' → kshop_product", etc.).
  It was meant to skip the LLM tool-selector call for ~30-40% of "obvious"
  queries and save ~200ms + ~325 tokens per query.

WHY IT WAS REMOVED:
  1. INCONSISTENT ROUTING — same user intent phrased slightly differently
     routed to different tables. "show crops in krushiratna" failed because
     "crops" wasn't a keyword. "what crops are available" worked because it
     fell through to the LLM. Users got two different answers to the same
     question. This is a fundamental limit of keyword matching — you can
     never enumerate every valid phrasing.

  2. MAINTENANCE BURDEN — every new product, new crop, new phrase required
     a code change + deployment. The knowledge_cache.py module exists exactly
     to avoid this kind of hardcoding.

  3. COST/BENEFIT LOSS — the "savings" are ~$0.00003 per query on Groq.
     Even at 10,000 queries/day that's 30 cents. Engineering time spent
     debugging keyword gaps dwarfs any savings.

WHAT HAPPENS NOW:
  route() always returns None. The orchestrator's Step 3 treats this as
  "ambiguous" and falls through to Step 4 (LLM tool selection), which runs
  the tool_selector LLM call for every query. The tool selector uses the
  knowledge cache to understand real product/crop content and makes a
  consistent, context-aware decision.

TO RESTORE KEYWORD ROUTING (not recommended):
  git revert this file to its previous version.
"""

from typing import List, Optional, Tuple
from app.core.logger import get_logger

logger = get_logger("intent_router")


class IntentRouter:
    """
    No-op router. Always returns None so every query is routed by the
    LLM tool selector in Step 4. This guarantees consistent, intent-aware
    routing with no hardcoded keyword lists to maintain.
    """

    def route(self, question: str) -> Optional[Tuple[List[str], str]]:
        """
        Always returns None → orchestrator falls through to LLM tool selector.

        The LLM tool selector has access to:
          - The full list of available tables with descriptions
          - Live database content via the knowledge cache (actual crop names,
            K-Shop categories, product samples)
          - Explicit disambiguation rules in its system prompt
        It makes a far more accurate decision than any keyword list could.
        """
        return None


# Singleton — kept for backward compat with orchestrator's import
intent_router = IntentRouter()


def get_intent_router() -> IntentRouter:
    return intent_router