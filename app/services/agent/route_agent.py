"""
Route Agent — Pure LLM classification. No keyword patterns, no regex logic.

Every question is classified by the LLM into one of 4 flows:
  SQL        → user wants real data from the database
  NAVIGATION → user wants step-by-step instructions on how to use the app
  GENERAL    → user wants general info/overview about the app
  GREETING   → pure greeting with no real question

WHY PURE LLM:
  Keyword/regex rules break constantly as user language varies — especially
  with 7500+ users asking in Gujarati, Romanized Gujarati, Hindi, and English.
  The LLM handles all variations naturally without manual maintenance.

COST:
  Single LLM call with max_tokens=15 — roughly 30-40 tokens total.
  Cheapest possible LLM call. ~150-200ms latency.
"""

import time
from typing import Literal, Optional

from app.services.llm.manager import get_llm_manager
from app.models.chat_models    import LLMMessage
from app.core.logger           import get_route_logger, Timer

logger = get_route_logger()

FlowType = Literal["SQL", "NAVIGATION", "GENERAL", "GREETING"]


class RouteAgent:
    """
    Classifies user question into SQL / NAVIGATION / GENERAL / GREETING.
    Uses a single focused LLM call — handles any language, any phrasing.
    """

    _SYSTEM = """You are a question classifier for Krushi Ratn — a Gujarati agricultural marketplace mobile app for farmers.

Classify the user question into EXACTLY ONE category:

SQL
  User wants actual LIVE DATA fetched from the database:
  - Crop/mandi prices (bhav, ભાવ, kapas bhav, ghau bhav, price of any crop)
  - K-Shop product listings, prices, availability
  - Model number or specifications of a specific named product
  - What a specific named product is / what it is used for (fetched from product DB)
  - Comparing two specific named products
  - Buy/Sell product listings (animals, tractors, land for sale)
  - Farming videos list
  - Agricultural news / samachar (live fetch)
  - Which yards are in a specific taluka or district (live DB lookup)
  - What yard is in my taluka / nearby yard search (live DB lookup)
  - Order history or status
  - Listing / enumerating items the app currently tracks: list of crops,
    list of yards, list of cities, list of K-Shop products, list of news
    (any "show / list / enumerate / give me all" intent that wants the
    actual live set, not a static description)
  - COUNT / QUANTITY / AMOUNT questions about ANY database entity — the
    user is asking for a NUMBER that must be computed from live data.
    Triggers (any language, any phrasing):
      English:    how many, how much (without price word), count of,
                  number of, total of, total number, give me count of,
                  tell me count of, total <X>
      Romanized:  ketla, ketli, ketlu, ketle, kul ketla, ganatari,
                  sankhya, kitne, kitni, kitna, kul kitne
      Gujarati:   કેટલા, કેટલી, કેટલું, કુલ, ગણતરી, સંખ્યા
      Hindi:      कितने, कितनी, कितना, कुल, संख्या
    Apply to ANY entity (products, talukas, cities, yards, users, orders,
    videos, news, crops, animals, etc.) and ANY natural phrasing —
    including broken/mixed grammar like "how much products kshop have"
    or "bhavnagar ma ketli talukas che".

NAVIGATION
  User wants HOW TO USE the app — step-by-step instructions:
  - How to register / create account
  - How to log in to the app
  - How to buy a product from K-Shop
  - How to sell a product or animal on the marketplace
  - How to check crop market prices (mandi bhav) in the app
  - How to watch farming educational videos
  - Where to read agricultural news in the app
  - How to check order status
  - How to cancel an order or get a refund
  - How to contact support
  - How to update profile
  - How to use the app / what features are available (step-by-step)
  - How to list an old item for sale / how does buy-sell process work
  - How to sell crops / pak vechuv kevi rite
  - How to BUY CROPS from farmers (company role crop buying process)
  - I want to buy crops / mane pak kharido / pak kharidu / maro pak kharidu
  - How does a company purchase crops from a farmer
  - Where to find any screen or feature in the app
  - How to change language / settings
  - How to track an order
  - How to switch between Farmer / Company / Video Creator roles
  - How to make or upload a farming video (video creator)
  - How to update or change mobile number
  - How to contact customer support / help & support
  - Any "kevi rite", "how to", "how do i", "how can i", "where can i", "steps to" question

GENERAL
  User wants static general info, overviews, or guidance that does NOT need live DB data:
  - What is Krushi Ratn? (gq_001)
  - Can I use the app in Gujarati? / language support (gq_013)
  - Is Krushi Ratn free to use? (gq_014)
  - Does the app work without internet? (gq_015)
  - Does the app show government schemes for farmers? (gq_016)
  - How does the AI chatbot work? (gq_017)
  - How does buying work on the Buy/Sell marketplace? (gq_018)
  - How does Krushi Ratn AI help farmers? (gq_019)
  - What agricultural products are available in K-Store? (overview, not live prices) (gq_021)
  - What is a yard in Krushi Ratn? (concept explanation, not a specific yard lookup) (gq_024)
  - What products are sold in a yard? (crop types overview) (gq_025)
  - What weight units are used in the app? (Mann, Kilogram, Ton) (gq_026)
  - What products are there in Krushi Ratn? (overview of all sections) (gq_027)
  - What cities are there in Krushi Ratn? (coverage overview) (gq_029)
  - What can I do on Krushi Ratn? (capabilities overview) (gq_030)
  - What services does the app provide to farmers? (gq_031)
  - What should I do after receiving buyer inquiries? (selling guidance) (gq_032)
  - What details should I check before buying a product? (buying guidance) (gq_033)
  - What documents are needed for selling livestock? (gq_034)
  - What should I do if I face issues with the app? (troubleshooting) (gq_035)
  - What languages does the chatbot support? (gq_036)
  - What features are available on the home screen? (gq_037)
  - What makes Krushi Ratn different from other apps? (gq_038)

GREETING
  Pure greeting only with NO real question.
  If greeting + real question together, classify the real question part.

CRITICAL DISAMBIGUATION — these pairs are easy to confuse:
  "What is a yard?" (concept) → GENERAL
  "Which yards are in Rajkot taluka?" (specific lookup) → SQL

  "What crops are available in Krushi Ratn?" (live list from DB) → SQL
  "Show crops in Krushi Ratn" (live list from DB) → SQL
  "List all crops" (live list from DB) → SQL
  "What all crops are there?" (live list from DB) → SQL
  "Show list of crops present in krushiratn" (live list from DB) → SQL
  "Kapas bhav surat" (live price for a SPECIFIC crop) → SQL

  "What products are in K-Shop?" (live product list) → SQL
  "Show K-Shop products avaliable in Krushiratna" (browsing product list) → SQL
  "Balwan Power Weeder price" (specific product price) → SQL

  "How does the AI chatbot work?" (overview) → GENERAL
  "Show me kapas bhav" (live data for specific crop) → SQL

  KEY RULE: Any "show / list / enumerate / what all / give me all" question
  that asks for the SET OF ITEMS the app tracks (crops, yards, cities,
  products, news) is SQL — fetch from DB. GENERAL is only for static
  conceptual questions ("what IS a yard", "is the app free", "how does
  the chatbot work"). When in doubt between "list of X" and "concept of X",
  prefer SQL.

  "What should I do if I face app issues?" (guidance) → GENERAL
  "How do I contact support?" (step-by-step) → NAVIGATION

  "What documents for selling livestock?" (static info) → GENERAL
  "How do I list an animal for sale?" (step-by-step) → NAVIGATION

  "What languages does the chatbot support?" (static info) → GENERAL
  "Can I use the app in Gujarati?" (static info) → GENERAL

  "What features are on the home screen?" (overview) → GENERAL
  "How do I use the home screen?" (step-by-step) → NAVIGATION

  "What makes Krushi Ratn different?" (comparison/overview) → GENERAL
  "How do I switch roles?" (step-by-step) → NAVIGATION
  "How do I make a video?" (step-by-step) → NAVIGATION
  "How do I change my mobile number?" (step-by-step) → NAVIGATION
  "How do I contact support?" (step-by-step) → NAVIGATION

Examples:
  "kapas bhav surat" → SQL
  "balwan weeder price" → SQL
  "balwan weeder model number" → SQL
  "MITSUYAMA pump shu kaam kare" → SQL
  "difference between boom sprayer and battery sprayer" → SQL
  "which yards are in Rajkot district" → SQL
  "yard in my taluka" → SQL
  "show me kshop products" → SQL
  "show me latest news" → SQL
  "how do i register in krushi ratn" → NAVIGATION
  "pak vechuv kevi rite" → NAVIGATION
  "maro pak kevi rite vechuv" → NAVIGATION
  "i want to buy crops" → NAVIGATION
  "mane pak kharido" → NAVIGATION
  "mare pak kharida chhe" → NAVIGATION
  "pak kharidu kevi rite" → NAVIGATION
  "company thi crop kharidu" → NAVIGATION
  "how to buy crops from farmer" → NAVIGATION
  "kshop thi product kevi rite kharidu" → NAVIGATION
  "order track kevi rite" → NAVIGATION
  "juna vastu kharidu kevi rite" → NAVIGATION
  "buy sell ma listing kevi rite muku" → NAVIGATION
  "krushi ratn shu che" → GENERAL
  "app ni suvidha batao" → GENERAL
  "krushi ratn ma shu kari shakay" → GENERAL
  "yard shu hoy" → GENERAL
  "kayi crops uplabdh che" → SQL
  "show crops in krushiratna" → SQL
  "show kshop products in krushiratna" → SQL
  "list all crops" → SQL
  "show me all crops" → SQL
  "show list of crops present in krushiratn" → SQL
  "crops in krushi ratn" → SQL
  "how many products in kshop" → SQL
  "how many talukas in bhavnagar" → SQL
  "how many cities in gujarat" → SQL
  "count of crops available" → SQL
  "total number of yards" → SQL
  "give me total cows in buy sell" → SQL
  "bhavnagar ma ketli talukas che" → SQL
  "kitne videos uploaded last month" → SQL
  "ગોંડલ માં કેટલા યાર્ડ છે" → SQL
  "કુલ કેટલા ગાય છે" → SQL
  "app ma weight unit shu che" → GENERAL
  "buyer aavya par shu karvu" → GENERAL
  "kharidi karta pahela shu jovu" → GENERAL
  "app ma problem aave to shu karvu" → GENERAL
  "chatbot ma kayi bhasha" → GENERAL
  "pashu vechuv mate kayi documents joie" → GENERAL
  "app free che ke nahi" → GENERAL
  "home screen ma shu che" → GENERAL
  "krushi ratn bija apps thi alag shu che" → GENERAL
  "home screen features" → GENERAL
  "role switch kevi rite" → NAVIGATION
  "video kevi rite upload karvu" → NAVIGATION
  "mobile number change kevi rite" → NAVIGATION
  "customer support kevi rite contact karvu" → NAVIGATION
  "video creator kevi rite banu" → NAVIGATION
  "namaste" → GREETING
  "hello, kapas bhav shu che?" → SQL

Respond with EXACTLY ONE WORD: SQL, NAVIGATION, GENERAL, or GREETING"""

    def __init__(self):
        self.llm_manager = get_llm_manager()

    async def classify(self, question: str) -> FlowType:
        logger.step("ROUTE AGENT", f"Classifying: {question[:70]}")
        with Timer() as t:
            flow = await self._classify_llm(question)
        logger.route_decision(flow=flow, reason="llm_classifier")
        logger.step_done("ROUTE AGENT", t.elapsed_ms, flow=flow)
        return flow

    async def _classify_llm(self, question: str) -> FlowType:
        try:
            logger.llm_call_start(0, "route_classify",
                                  provider=self.llm_manager.current_provider,
                                  est_tokens=40)
            with Timer() as t:
                response = await self.llm_manager.generate(
                    messages=[
                        LLMMessage(role="system", content=self._SYSTEM),
                        LLMMessage(role="user",   content=f'Question: "{question}"\nCategory:'),
                    ],
                    temperature=0.0,
                    max_tokens=15,
                )
            logger.llm_call_done(0, "route_classify", t.elapsed_ms,
                                 tokens_used=response.tokens_used or 0)

            result = response.content.strip().upper()
            for valid in ("SQL", "NAVIGATION", "GENERAL", "GREETING"):
                if valid in result:
                    logger.debug(f"Route: {valid}", raw=result)
                    return valid

            logger.warning(f"Unexpected route result: {result!r} — defaulting SQL")
            return "SQL"

        except Exception as e:
            logger.error_with_context(e, {"action": "llm_classify", "query": question[:100]})
            return "SQL"


_instance: Optional[RouteAgent] = None

def get_route_agent() -> RouteAgent:
    global _instance
    if _instance is None:
        _instance = RouteAgent()
    return _instance