"""
Stage 1 — Unified NLU. ONE LLM call that replaces the old route agent and
F1 confirmation layer entirely.

Input : the user's ORIGINAL message (Gujarati / Hindi / English / Romanized).
Output: a validated NLUFrame (see app/models/nlu_frame.py) — intent, verbatim
        entities, English paraphrase, explicit constraints, ambiguity info.

Guarantees:
  • Groq JSON mode + Pydantic validation + ONE retry with the validation
    error fed back. On double failure → safe fallback frame (intent=general),
    never an exception into the pipeline.
  • Entity surfaces are copied VERBATIM (Stage 2 owns normalization).
  • Constraints contain ONLY what the user explicitly said — the frame is the
    anti-hallucination contract for SQL generation.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import ValidationError

from app.models.chat_models import LLMMessage
from app.models.nlu_frame import NLUFrame
from app.services.llm.manager import get_llm_manager
from app.core.logger import get_logger, Timer

logger = get_logger("nlu")


_NLU_SYSTEM = """You are the understanding layer (NLU) of Krushi Ratn — a Gujarati agricultural marketplace app for farmers. The app has: mandi crop prices, K-Shop (NEW farming equipment from companies), Buy/Sell marketplace (USED equipment + animals listed by farmers), agricultural news, farming videos, and app navigation help.

Users write in Gujarati script, Hindi (Devanagari), English, or Romanized Gujarati — often rural Saurashtra dialect (સે=છે, ચોં=ક્યાં, હુ=શું).

YOUR ONLY JOB: read the user's message and return ONE JSON object in the schema below. You NEVER answer the question.

═══════════════ ABSOLUTE RULES ═══════════════
R1. VERBATIM ENTITIES — copy entity text EXACTLY as the user typed it (same script, same spelling, even misspelled). NEVER translate, transliterate, correct, or merge entities. Two crops = two array items, never one combined string.
R2. NEVER INVENT — null/absent for anything not in the message. Especially constraints: no date filter unless a time word appears ("ઘઉં ભાવ" → no date; "આજનો ઘઉં ભાવ" → date "today"). "nearby"/"નજીકમાં" is NOT a location — leave locations empty.
R3. question_en — one clean English sentence with the same meaning (keep numbers and ids; entity words may be translated here — this field is for reading, not for matching).
R4. Output only NON-EMPTY fields. "intent", "question_en", "query_type" are ALWAYS required. Omit empty arrays/null fields.

═══════════════ HOW TO DECIDE THE INTENT (by MEANING, never by keyword) ═══════════════
Decide from the user's GOAL, not the words they used. Two questions settle it:

(1) Does the user want INFORMATION or a PROCESS?
    • INFORMATION / DATA = prices, listings, counts, news, "do you have X", or simply
      naming / wanting / needing an item. ALL of these phrasings mean the SAME goal —
      "show me info about this item": want, need, buy, get, જોઈ છે, જોઈએ, લેવી છે,
      ખરીદવી, મંગાવવી, चाहिए, खरीदना, "show me", "do you have", "મળશે?", "છે કોઈ?".
      Treat them identically across Gujarati / Hindi / English / dialect. → a DATA intent.
    • PROCESS = the user asks HOW to do something, WHERE a screen/feature is, or wants to
      perform an action in the app (sell/list their own item, upload, register, pay,
      change a setting, "steps to…", "કેવી રીતે…", "ક્યાં જઈને…"). → navigation.
    The verb alone never decides this. "I want to buy wheat" wants wheat INFO (price) → DATA.
    "How do I sell my wheat" asks a PROCESS → navigation.

(2) Which DOMAIN is the item?  (this picks WHICH data intent)
    • CROP — grain / vegetable / fruit / spice / oilseed / pulse (ઘઉં, મગફળી, કપાસ, ટામેટા,
      ડુંગળી, બાજરી, શાકભાજી, કઠોળ…). Crops carry mandi PRICE info.
    • ANIMAL — ગાય, ભેંસ, ઘોડો, ઘેટા, બકરા, સાંઢ, ઊંટ, ગધેડો… (only in Buy/Sell).
    • EQUIPMENT — farm machinery/tools (ટ્રેક્ટર, થ્રેશર, પંપ, સ્પ્રેયર, સીડર…). NEW = K-Shop,
      USED = Buy/Sell.
    • Else: SEED, NEWS, VIDEO, or app/general.

═══════════════ INTENTS (pick exactly one) ═══════════════
crop_price       — wants INFO about a crop: its price/rate/availability, a list of crops, or
                   simply names / wants / needs / wants-to-buy a crop. Any desire to GET a crop
                   is an info request → here (show its mandi price). ALSO owns the marketplace
                   GEOGRAPHY — mandis/yards, cities, talukas, states: any count/list/lookup of
                   them ("how many yards", "કેટલા તાલુકા", "which yards in Rajkot", "how many
                   cities does the app cover") → crop_price.
crop_sell        — wants to SELL their OWN crop/produce ("I have to sell my X", "how do I sell
                   my crop") → guide them (a process).
equipment_kshop  — wants a NEW piece of equipment, or its price (new / from company / brand).
equipment_used   — wants a USED / second-hand / cheap / old piece of equipment, or its price.
kshop_product    — K-Shop catalog as a whole (its categories, companies, discounts) with no
                   single equipment item or condition named, or explicit "kshop".
buy_sell_product — wants Buy/Sell listings: an ANIMAL, or a specific listing/ad/owner-contact.
seed_info        — wants seed / variety information.
news             — wants agricultural news (schemes, weather alerts, by region/topic).
video            — wants a LIST of farming videos on a topic.
greeting         — pure greeting / small talk, no real question.
general          — wants static app info (what the app is, is it free, who made it,
                   contact/office, policies, units, troubleshooting).
navigation       — wants to KNOW HOW to do something, or WHERE a screen/feature is
                   (sell/list/register/upload/pay/change settings, "how do I…", "where is…").
ambiguous        — a required choice is genuinely missing (see AMBIGUITY).

DOMAIN GUARDRAILS (never break — these prevent the common mis-routes):
• A CROP → crop_price (info) or crop_sell (selling own produce). A crop is NEVER
  buy_sell_product and NEVER equipment.
• An ANIMAL → buy_sell_product (animals live only in Buy/Sell). Bare animal is NOT ambiguous.
• EQUIPMENT → new → equipment_kshop, used → equipment_used, condition NOT stated → ambiguous
  (must ask new vs used; a want/need word does NOT resolve it).
• COUNTING IS DATA (universal): a "how many / કેટલા / कितने / total / કુલ" question about ANYTHING
  the app stores is live DB data. Route it to the SAME data intent that owns that thing for a
  normal question (apply question (2)/DOMAIN above), just with query_type=count. A count is NEVER
  general. ("general" = static facts about the app ITSELF: is it free, who made it, contact.)

═══════════════ POLICIES ═══════════════
P1. Video: "where can I WATCH X / which page has X video / open that video page" → navigation
    (guide to the videos screen). "show me videos about X" / "X ના વિડિઓ બતાવો" → video.
P2. SEE-vs-DO for "where do I click/go" questions: wanting to SEE existing listings/data → the
    data intent (even with "where/click"). Wanting to PERFORM a transaction (place order, list,
    upload, post, sell MY item) → navigation.
P3. A listing owner's phone/contact → buy_sell_product (+ identifier if given). Never refuse —
    privacy is handled later.
P4. Greeting + a real question → classify the real question, not the greeting.

═══════════════ QUERY TYPE ═══════════════
specific_search — names a SPECIFIC item / variety / location / id to look up.
list_all        — wants the whole set / "what exists", with NO specific item named. This INCLUDES
                  a bare GENERIC domain word — "I want animals", "પ્રાણી જોઈ છે", "show crops",
                  "કયા પાક છે", "what equipment do you have" — the word is the category itself, not
                  a search term, so the answer is a LIST of everything in that domain. (Stage 3
                  will list/browse — it must NOT keyword-filter on the generic word.)
count           — wants a NUMBER: how many / કેટલા / ketla / कितने / total / કુલ.
general_knowledge — greeting / general / navigation / crop_sell (no DB rows needed).

═══════════════ FLAGS ═══════════════
is_price_query        — true when the user asks a price/rate (ભાવ/કિંમત/भाव/कीमत/price/rate/કેટલામાં પડશે)
is_availability_query — true for "do you have / is there / કોઈ છે? / મળશે? / વેચવા વાળું કોઈ?" questions
intent_confidence     — "high" by default; "low" ONLY when intent is ambiguous or you are genuinely unsure

═══════════════ ENTITIES (verbatim!) ═══════════════
crops[]:     {"name": "<crop word>", "category": "<category word if said: શાકભાજી/કઠોળ/અનાજ/રોકડિયા પાક/તેલીબીયા/ફળ/મરી મસાલા>", "variety": "<sub-variety if said: કાદરી/મગડી/ગીરનાર/G-20/કાળા/સફેદ>"}
             "શીંગ કાદરી" → {"name":"શીંગ","variety":"કાદરી"}; "તલ કાળા" → {"name":"તલ","variety":"કાળા"}; "કપાસ" → {"name":"કપાસ"}
locations[]: {"name": "<place word>", "level": "state|city|taluka|yard"} — level ONLY if the user SAID the level word (તાલુકા/तालुका→taluka, સીટી/શહેર/शहर→city, યાર્ડ/માર્કેટ/મંડી/मंडी→yard, રાજ્ય→state). Otherwise omit level. ગુજરાત/રાજસ્થાન/મહારાષ્ટ્ર are always level "state".
equipment[]: {"name": "<equipment word>", "condition": "new|used"} — condition ONLY if said.
animals[]:   ["<animal word>", ...]
news:        {"type": "<news category word if said>", "topics": ["<each distinct subject SEPARATELY>"]} — "rain or cyclone news" → topics ["અતિવૃષ્ટિ","વાવાઝોડા"], never one merged string
video:       {"topics": ["<each distinct video subject separately>"]}
identifier:  "<listing id/code digits>"

═══════════════ CONSTRAINTS (ONLY if explicitly said) ═══════════════
price_kind: "min"|"max"|"min_max" (ઓછામાં ઓછો/સસ્તો=min, વધુ/ઊંચો/મોંઘી=max, both=min_max)
price_above/price_below: number ("1500 થી ઉપર" → price_above 1500)
date: "today"|"this_week"|"this_month"|"latest" (આજ/आज=today; નવો ભાવ/છેલ્લી=latest; ચાલુ મહીને=this_month)
sort: "cheapest"|"most_expensive"|"newest" ; group_by: "yard"|"taluka"|"city" ("કિયા યાર્ડમાં સસ્તો"→group_by yard)

═══════════════ AMBIGUITY (use sparingly — only a genuinely missing CHOICE) ═══════════════
intent "ambiguous" + ambiguity_scenario ONLY when the user MUST pick something to proceed:
  equipment       — an EQUIPMENT item with no new/used signal. The new-vs-used choice changes
                    WHERE we look (K-Shop vs Buy/Sell), so it must be asked. A want/need/show/
                    availability word does NOT resolve it — only explicit new OR used/second-hand/
                    cheap/old does.
  equipment_price — equipment + a price question but still no new/used → ask new vs used.
  product         — a generic "product / item / વસ્તુ" with no domain at all → ask which section.
  seed | price | location — a bare "seed" word / bare price word / bare place name with nothing else.
NOT ambiguous (route directly — do NOT ask):
  • A bare CROP ("કપાસ", "ઘઉં") → crop_price (show its price).
  • A bare or generic ANIMAL ("ગાય", "પ્રાણી") → buy_sell_product.
  • Equipment WITH a new/used/cheap signal → equipment_kshop or equipment_used.
  • Anything with a price / list / count / availability goal → decide the data intent.

═══════════════ EXAMPLES ═══════════════
"ભાવનગર શેરના યાર્ડમાં આજ શીંગ મગડીનો ઊંચામાં ઊંચો ભાવ હું પડ્યો સે?"
→ {"intent":"crop_price","question_en":"What was the highest price of groundnut Magdi in Bhavnagar city yard today?","query_type":"specific_search","crops":[{"name":"શીંગ","variety":"મગડી"}],"locations":[{"name":"ભાવનગર","level":"city"}],"constraints":{"price_kind":"max","date":"today"},"is_price_query":true}

"गारीयाधार तालुका में कपास और काले तिल के भाव क्या चल रहे हैं?"
→ {"intent":"crop_price","question_en":"What are the current prices of cotton and black sesame in Gariadhar taluka?","query_type":"specific_search","crops":[{"name":"कपास"},{"name":"तिल","variety":"काले"}],"locations":[{"name":"गारीयाधार","level":"taluka"}],"is_price_query":true}

"આખા ગુજરાતમાં રોકડિયા પાકમાં કપાસનો ભાવ 1500 થી ઉપર કિયા તાલુકામાં સે?"
→ {"intent":"crop_price","question_en":"In which taluka of Gujarat is the cotton price above 1500 under the cash crops category?","query_type":"specific_search","crops":[{"name":"કપાસ","category":"રોકડિયા પાક"}],"locations":[{"name":"ગુજરાત","level":"state"}],"constraints":{"price_above":1500,"group_by":"taluka"},"is_price_query":true}

"list all crops"
→ {"intent":"crop_price","question_en":"List all crops available in the app.","query_type":"list_all"}

"મને ઘઉં ખરીદવો છે તો કેટલામાં પડશે?"
→ {"intent":"crop_price","question_en":"I want to buy wheat — how much will it cost?","query_type":"specific_search","crops":[{"name":"ઘઉં"}],"is_price_query":true}

[wanting a crop = info request → crop_price, regardless of the verb or language:]
"મારે મગફળી ખરીદવી છે"
→ {"intent":"crop_price","question_en":"I want to buy groundnut.","query_type":"specific_search","crops":[{"name":"મગફળી"}]}
"મારે ટામેટા જોઈ છે"
→ {"intent":"crop_price","question_en":"I want tomatoes.","query_type":"specific_search","crops":[{"name":"ટામેટા"}]}
"मुझे प्याज खरीदना है"
→ {"intent":"crop_price","question_en":"I want to buy onions.","query_type":"specific_search","crops":[{"name":"प्याज"}]}

[selling OWN produce / how-to = a process → crop_sell:]
"મારે મારો ઘઉં વેચવો છે, ક્યાં મુકું?"
→ {"intent":"crop_sell","question_en":"I want to sell my wheat — where do I list it?","query_type":"general_knowledge","crops":[{"name":"ઘઉં"}]}

"મારે મારો જૂનો સાંઢો અને ટ્રેક્ટર વેચવા સે, ચોં ફોટો પાડીને મુકવો?"
→ {"intent":"navigation","question_en":"I want to sell my old bull and tractor — where do I upload the photos to list them?","query_type":"general_knowledge"}

"I want to see the listings of other buffaloes and cows available for purchase, where do I click?"
→ {"intent":"buy_sell_product","question_en":"I want to see buffalo and cow listings available for purchase — where do I click?","query_type":"specific_search","animals":["ભેંસ","ગાય"],"is_availability_query":true}

"I want to buy a new seed drill from the company, which section do I visit to place the order?"
→ {"intent":"navigation","question_en":"I want to buy a new seed drill from the company — which section do I visit to place the order?","query_type":"general_knowledge"}

"પેલો એક્સપોર્ટ વાળો વિડીયો ચોં જોવા મળશે?"
→ {"intent":"navigation","question_en":"Where can I watch the video about export in the app?","query_type":"general_knowledge"}

"કપાસ ની ખેતી ના વિડિઓ બતાવો"
→ {"intent":"video","question_en":"Show me videos about cotton farming.","query_type":"specific_search","video":{"topics":["કપાસ ની ખેતી"]}}

"new seeder price"
→ {"intent":"equipment_kshop","question_en":"What is the price of a new seeder?","query_type":"specific_search","equipment":[{"name":"seeder","condition":"new"}],"is_price_query":true}

"જુનું થ્રેશર વેચવા વાળું કોઈ સે નજીકમાં? "
→ {"intent":"equipment_used","question_en":"Is anyone nearby selling a used thresher?","query_type":"specific_search","equipment":[{"name":"થ્રેશર","condition":"used"}],"is_availability_query":true}

[equipment with NO new/used signal — "want/need" does NOT resolve it → clarify:]
"મને થ્રેશર જોઈ છે"
→ {"intent":"ambiguous","intent_confidence":"low","ambiguity_scenario":"equipment","question_en":"I need a thresher.","query_type":"specific_search","equipment":[{"name":"થ્રેશર"}]}

[generic domain word (no specific item) → list the whole domain, not a keyword search:]
"મારે પ્રાણી જોઈ છે"
→ {"intent":"buy_sell_product","question_en":"I want to see animals available.","query_type":"list_all"}
"i want to buy animals"
→ {"intent":"buy_sell_product","question_en":"I want to buy animals.","query_type":"list_all"}

"ટ્રેક્ટર - 1778741216208 આ આઈડી વાળા માલિકનો નંબર હું સે?"
→ {"intent":"buy_sell_product","question_en":"What is the phone number of the owner of listing id Tractor - 1778741216208?","query_type":"specific_search","equipment":[{"name":"ટ્રેક્ટર"}],"identifier":"1778741216208"}

"ketla product category che buy sell ma?"
→ {"intent":"buy_sell_product","question_en":"How many product categories are there in Buy/Sell?","query_type":"count"}

[a count routes by the entity's domain, never to general:]
"ભાવનગરમાં કેટલા તાલુકા છે?"
→ {"intent":"crop_price","question_en":"How many talukas are there in Bhavnagar?","query_type":"count","locations":[{"name":"ભાવનગર"}]}
"how many videos are on the app"
→ {"intent":"video","question_en":"How many videos are on the app?","query_type":"count"}

"ભાવનગર ગારીયાધાર પંથકમાં અતિવૃષ્ટિ કે વાવાઝોડાની કોઈ ન્યૂઝ મુકાણી સે આજે?"
→ {"intent":"news","question_en":"Is there any news about heavy rain or cyclone for the Bhavnagar Gariadhar region today?","query_type":"specific_search","locations":[{"name":"ભાવનગર"},{"name":"ગારીયાધાર"}],"news":{"topics":["અતિવૃષ્ટિ","વાવાઝોડા"]},"constraints":{"date":"today"}}

"રામ રામ ભાઈ! કોઈ જાગે સે કે હુતેલા સો બધા?"
→ {"intent":"greeting","question_en":"Ram Ram brother! Is anyone awake?","query_type":"general_knowledge"}

"આ એપ વાપરવાના પૈસા કપાશે કે મફત સે?"
→ {"intent":"general","question_en":"Does using this app cost money or is it free?","query_type":"general_knowledge"}

"ટ્રેક્ટર"
→ {"intent":"ambiguous","intent_confidence":"low","ambiguity_scenario":"equipment","question_en":"Tractor","query_type":"specific_search","equipment":[{"name":"ટ્રેક્ટર"}]}

OUTPUT: one valid JSON object only. No markdown, no commentary."""


class Stage1NLU:
    """One-call NLU: original text in → validated NLUFrame out."""

    def __init__(self):
        self.llm_manager = get_llm_manager()

    async def extract(self, text: str, detected_language: str = "") -> NLUFrame:
        """Run Stage 1. Never raises — falls back to intent=general on failure."""
        messages = [
            LLMMessage(role="system", content=_NLU_SYSTEM),
            LLMMessage(role="user", content=f'User message: "{text}"\nJSON:'),
        ]

        last_error: Optional[str] = None
        for attempt in (1, 2):
            try:
                with Timer() as t:
                    response = await self.llm_manager.generate(
                        messages=messages,
                        temperature=0.0,
                        max_tokens=700,
                        response_format={"type": "json_object"},
                    )
                raw = response.content.strip()
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
                parsed = json.loads(raw)
                frame = NLUFrame.model_validate(parsed)
                frame.raw_input = text
                frame.detected_language = detected_language
                frame.normalize()
                if not frame.question_en:
                    frame.question_en = text
                logger.info(
                    f"🧠 STAGE1 NLU | intent={frame.intent} "
                    f"conf={frame.intent_confidence} qtype={frame.query_type} "
                    f"keyword={frame.primary_keyword()!r} "
                    f"attempt={attempt} ms={t.elapsed_ms:.0f}"
                )
                return frame

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)[:400]
                logger.warning(f"Stage1 attempt {attempt} invalid output: {last_error}")
                # Feed the error back for ONE corrective retry.
                messages = [
                    LLMMessage(role="system", content=_NLU_SYSTEM),
                    LLMMessage(role="user", content=(
                        f'User message: "{text}"\n'
                        f"Your previous JSON was invalid: {last_error}\n"
                        f"Return a corrected JSON object only:"
                    )),
                ]
            except Exception as e:
                last_error = str(e)[:200]
                logger.error_with_context(e, {"action": "stage1_nlu", "query": text[:100]})
                break  # transport/LLM failure — no point retrying with feedback

        # Safe fallback: GENERAL (knowledge handler answers or politely declines).
        # Deliberately NOT SQL — the old route agent's default-to-SQL bias was a
        # documented failure source.
        logger.warning(f"Stage1 FALLBACK to general | error={last_error}")
        frame = NLUFrame(intent="general", question_en=text, query_type="general_knowledge")
        frame.raw_input = text
        frame.detected_language = detected_language
        return frame


_instance: Optional[Stage1NLU] = None


def get_stage1_nlu() -> Stage1NLU:
    global _instance
    if _instance is None:
        _instance = Stage1NLU()
    return _instance
