"""
Answer Generator - Third LLM Call.
Generates natural language answers from query results with greeting support.
"""

from typing import List, Optional
from app.services.llm.manager import get_llm_manager
from app.models.chat_models import LLMMessage, QueryResult, AnswerGenerationResponse
from app.core.logger import get_agent_logger

logger = get_agent_logger()


class AnswerGenerator:
    """Generates natural language answers from query results."""
    
    def __init__(self):
        """Initialize answer generator."""
        self.llm_manager = get_llm_manager()
    
    async def generate_answer(
        self,
        user_query: str,
        query_results: List[QueryResult],
        has_greeting: bool = False,
        target_language: Optional[str] = None,
        intent_note: str = "",
        keyword_hint: str = "",
    ) -> AnswerGenerationResponse:
        """
        Generate natural language answer from query results.

        intent_note  — confirmed domain from F1 (e.g. "User confirmed: CROP MARKET PRICES")
        keyword_hint — matched keyword from F1 (e.g. "kapas") for grounding the answer

        Returns:
            AnswerGenerationResponse with natural language answer
        """
        try:
            logger.info("💬 ANSWER GENERATION STARTED", query=user_query[:100])

            # Build system prompt
            system_prompt = self._build_system_prompt(has_greeting, intent_note, keyword_hint)
            
            # Build user message with query results
            user_message = self._build_user_message(user_query, query_results, has_greeting)
            
            # Prepare messages
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_message)
            ]
            
            # Call LLM
            response = await self.llm_manager.generate(
                messages=messages,
                temperature=0.1,  # Low — strict factual formatting, no creativity
                max_tokens=2500
            )
            
            answer = response.content.strip()
            
            # Extract source tables
            sources = [result.table_name for result in query_results]
            
            logger.final_answer(answer=answer, tokens=response.tokens_used)
            
            return AnswerGenerationResponse(
                answer=answer,
                sources=sources
            )
            
        except Exception as e:
            logger.error_with_context(e, {
                "action": "generate_answer",
                "query": user_query[:200]
            })
            raise
    
    def _build_system_prompt(self, has_greeting: bool, intent_note: str = "", keyword_hint: str = "") -> str:
        """Rich answer prompt — present DB data clearly with specs and context."""

        greeting_note = ""
        if has_greeting:
            greeting_note = "Start with a brief greeting (Hello / Hi). Then give the data immediately.\n"

        # Intent grounding block — prevents hallucination by telling the LLM
        # exactly what domain the user confirmed and what keyword was searched
        intent_block = ""
        if intent_note or keyword_hint:
            intent_block = "\nCONTEXT FROM USER INTENT CONFIRMATION:\n"
            if intent_note:
                intent_block += f"  - {intent_note}\n"
            if keyword_hint:
                intent_block += f"  - The user asked specifically about: '{keyword_hint}'\n"
            intent_block += (
                "  - STRICT RULE: Answer ONLY based on the database rows below. "
                "Do NOT add any information not present in the rows. "
                "Do NOT guess or extrapolate. If the data is empty or irrelevant, say so clearly.\n"
            )

        return f"""You are a helpful assistant for Krushi Ratn, an agricultural marketplace app for Gujarat farmers.
{greeting_note}{intent_block}
Your job: present database results concisely and naturally in ENGLISH.
Translation to the user's language is handled separately after your response.

CRITICAL ANTI-HALLUCINATION RULES:
- ONLY report what is in the database rows provided below.
- NEVER invent product names, prices, quantities, or availability.
- NEVER say "available" or "in stock" unless a row explicitly says so.
- If rows from a table are EMPTY or 0, report that honestly — do NOT make up alternatives.

IMPORTANT: Always refer to Krushi Ratn as an "app" or "application" — NEVER say "website".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORBIDDEN PHRASES — NEVER USE THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✗ "There are X rows found"
✗ "Found X results in [table]"
✗ "From [table_name] (Found X rows)"
✗ "Here are the details:"
✗ "I found X products"
✗ Any mention of table names, SQL, database, rows, columns
✗ Any count like "21 water pumps found" at the start of the answer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO WRITE THE ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — PRODUCT LISTS (K-Shop products, multiple results):
Start with a short intro sentence, then list ONLY name + price. No specs, no descriptions.

Intro examples (pick based on query):
  "The following water pumps are available in Krushi Ratn:"
  "Here are the tractors available on Krushi Ratn:"
  "These sprayers are currently listed in Krushi Ratn:"

Then list items in this exact format (one per line):
  1. [Product Name] — ₹[price] (Discount: ₹[discount_price])
  2. [Product Name] — ₹[price] (Discount: ₹[discount_price])
  ...

DO NOT add bullet points, engine specs, displacement, description, or any
extra details under each product. Just name + price. Nothing else.

Show at most 10 items. If there are more, end with:
  "For more options, please visit the Krushi Ratn app."

RULE 2 — SINGLE PRODUCT:
Give a clean one-paragraph answer with name, price, and 1-2 most relevant
specs if the user asked about specs. Otherwise just name + price.

RULE 3 — CROP / MANDI PRICES:
"[Crop] prices at [yard/city]: ₹[min]-₹[max] per [unit]."
If multiple locations, list each on a new line.

RULE 4 — EMPTY RESULTS:
Do NOT say "no rows found" or "no data in the database".
Say it naturally based on what was asked:
  - Seeds: "No seeds are currently available on Krushi Ratn."
  - Products: "No [product type] are currently listed on Krushi Ratn."
  - Prices under a limit: "No products are currently priced under ₹[X]. The lowest-priced options start at ₹[Y]." (only if data exists)

RULE 5 — COUNT QUERIES (e.g. "how many products?"):
One short sentence: "Krushi Ratn currently lists [N] [things]."
Never "There are N rows found."

RULE 6 — NO UNREQUESTED EXTRAS:
- Do NOT add tips, suggestions, or "Would you like more details?"
- Do NOT add category summaries unless asked
- Do NOT explain what Krushi Ratn is unless asked
- Do NOT mention "in our database"

Keep the answer short, natural, and directly responsive to what was asked.
"""

    
    def _build_user_message(
        self,
        user_query: str,
        query_results: List[QueryResult],
        has_greeting: bool
    ) -> str:
        """Build user message with query results."""
        
        # Format query results
        results_text = self._format_query_results(query_results)
        
        greeting_reminder = ""
        if has_greeting:
            greeting_reminder = "\n\nREMEMBER: Start your response with an appropriate greeting since the user greeted you!"
        
        user_message = f"""User Question: "{user_query}"

Database Query Results:
{results_text}
{greeting_reminder}

Please provide a natural, helpful answer to the user's question based on these results."""
        
        return user_message
    
    def _format_query_results(self, query_results: List[QueryResult]) -> str:
        """
        Format query results for the prompt.
        Does NOT include table names or row counts — those leak into
        the LLM's answer as "Found X rows in Y table" which is user-facing noise.
        """

        if not query_results:
            return "No data available."

        formatted_parts = []

        for result in query_results:
            if result.row_count == 0:
                formatted_parts.append("(empty result set)")
                continue

            # Format rows (limit to first 20 sent to LLM — the prompt caps display at 10)
            rows = result.rows[:20]

            for i, row in enumerate(rows, 1):
                row_items = []
                for key, value in row.items():
                    if value is None:
                        value = "N/A"
                    if isinstance(value, str) and len(value) > 400:
                        value = value[:400] + "..."

                    row_items.append(f"{key}: {value}")

                formatted_parts.append(f"Row {i}: {', '.join(row_items)}")

            if result.row_count > 20:
                formatted_parts.append(f"(plus {result.row_count - 20} more rows not shown)")

        return '\n'.join(formatted_parts)


# Global answer generator instance
answer_generator = AnswerGenerator()


def get_answer_generator() -> AnswerGenerator:
    """Get answer generator instance."""
    return answer_generator