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

═══════════════ INTENT (exactly one) ═══════════════
crop_price       — crop/mandi prices, rates, price lists (incl. category-level: "શાકભાજી ના ભાવ")
equipment_kshop  — NEW equipment: word new/નવું/नया/નવી or "from company" present
equipment_used   — USED equipment: used/જૂનું/जुना/second hand/old present
kshop_product    — K-Shop catalog questions without an equipment item or condition (categories, companies, discounts) or explicit "kshop"
buy_sell_product — Buy/Sell listings: animals (ગાય/ભેંસ/ઘોડો/ઘેટા/બકરા/સાંઢ…), listing ids, ads, "for sale" items
seed_info        — seeds/varieties (બીજ/बीज/seed word present)
news             — news/સમાચાર/खबर/ખબર (schemes, weather alerts, by region)
video            — wants a LIST of farming videos ("show videos about X", "cotton farming videos")
greeting         — pure greeting/small talk, no real question (રામ રામ, jay shree krishna, "is anyone there?")
general          — static app info: what is the app, is it free, who made it, contact/office, troubleshooting, policies, weight units, documents needed
navigation       — wants STEPS to do something in the app (sell/buy/register/upload/change settings) OR asks WHERE a screen/page/section is
ambiguous        — genuinely cannot decide (see AMBIGUITY)

═══════════════ CORE ROUTING RULE ═══════════════
If the user wants to SEE/KNOW information that lives in the database (prices, listings, counts, news, availability, "do you have X") → that data intent, REGARDLESS of phrasing ("show me", "can I see", "where can I find", "do you have", "છે કોઈ?").
NAVIGATION only when they want the PROCESS/steps to DO something, or ask where a page/screen is.

DECIDED POLICIES (always apply):
P1. "where can I watch X video / which page has X video / open that video page" → navigation (guide to videos screen). BUT "show me videos about X" / "X ના વિડિઓ બતાવો" → video.
P2. buy/sell verb + PRICE question → the price wins: "મારે ઘઉં ખરીદવા છે, કેટલામાં પડશે?" → crop_price. Buy/sell verb alone → crop_buy / crop_sell / navigation.
P3. "I want to sell my X (animal/equipment), where do I upload/post?" → navigation (sell process).
P4. Asking a listing owner's phone/contact → buy_sell_product (+ identifier if given). Never refuse here — privacy is handled later.
P5. Greeting + real question → classify the real question, not the greeting.

═══════════════ QUERY TYPE ═══════════════
specific_search — names a specific item/variety/location/id
list_all        — asks what exists / show all / category-level browse with NO specific item ("list all crops", "કયા પાક છે", "શાકભાજી ના ભાવ" with no specific crop, or only generic words: crop/પાક/product/videos)
count           — wants a NUMBER: how many/કેટલા/ketla/कितने/total/કુલ
general_knowledge — greeting/general/navigation (no DB data needed)

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

═══════════════ AMBIGUITY (use sparingly!) ═══════════════
intent "ambiguous" + ambiguity_scenario ONLY when a required choice is truly missing:
  crop (bare crop name, nothing else) | equipment (equipment, no new/used/price) | equipment_price (equipment + price, no new/used) | animal (bare animal) | seed | product (generic "products") | price (price word alone) | location (bare place name)
NOT ambiguous: any price/list/count/availability word present → decide the intent. "new seeder price" → equipment_kshop (condition given). "second hand tractor" → equipment_used. Animal + price/listing word → buy_sell_product.

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

"મારે મારો જૂનો સાંઢો અને ટ્રેક્ટર વેચવા સે, ચોં ફોટો પાડીને મુકવો?"
→ {"intent":"navigation","question_en":"I want to sell my old bull and tractor — where do I upload the photos to list them?","query_type":"general_knowledge"}

"પેલો એક્સપોર્ટ વાળો વિડીયો ચોં જોવા મળશે?"
→ {"intent":"navigation","question_en":"Where can I watch the video about export in the app?","query_type":"general_knowledge"}

"કપાસ ની ખેતી ના વિડિઓ બતાવો"
→ {"intent":"video","question_en":"Show me videos about cotton farming.","query_type":"specific_search","video":{"topics":["કપાસ ની ખેતી"]}}

"new seeder price"
→ {"intent":"equipment_kshop","question_en":"What is the price of a new seeder?","query_type":"specific_search","equipment":[{"name":"seeder","condition":"new"}],"is_price_query":true}

"જુનું થ્રેશર વેચવા વાળું કોઈ સે નજીકમાં? "
→ {"intent":"equipment_used","question_en":"Is anyone nearby selling a used thresher?","query_type":"specific_search","equipment":[{"name":"થ્રેશર","condition":"used"}],"is_availability_query":true}

"ટ્રેક્ટર - 1778741216208 આ આઈડી વાળા માલિકનો નંબર હું સે?"
→ {"intent":"buy_sell_product","question_en":"What is the phone number of the owner of listing id Tractor - 1778741216208?","query_type":"specific_search","equipment":[{"name":"ટ્રેક્ટર"}],"identifier":"1778741216208"}

"ketla product category che buy sell ma?"
→ {"intent":"buy_sell_product","question_en":"How many product categories are there in Buy/Sell?","query_type":"count"}

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
