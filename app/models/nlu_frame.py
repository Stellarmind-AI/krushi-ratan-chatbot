"""
NLU Frame — the structured representation every user message is converted to.

This is the language-agnostic canonical form of a query (the "semantic frame").
Stage 1 (one LLM call) fills the LLM-owned fields; code injects raw_input and
detected_language and derives everything else. After Stage 1, the rest of the
pipeline consumes ONLY this object — the original language never matters again.

Design rules (agreed):
  • Arrays-always: entity collections are lists even for a single item.
  • Verbatim surfaces: entity text is copied EXACTLY as the user typed it —
    Stage 2 (catalog resolution) owns normalization, never the LLM.
  • LLM outputs only what code cannot compute. needs_clarification is derived
    (intent == "ambiguous"); raw_input/detected_language are injected by code.
  • extra="ignore" so stray LLM keys never break validation.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ── Vocabulary ───────────────────────────────────────────────────────────────

IntentType = Literal[
    "crop_price",        # mandi/market crop prices (incl. category-scoped lists)
    "crop_buy",          # wants to BUY crops → navigation guidance
    "crop_sell",         # wants to SELL crops → navigation guidance
    "equipment_kshop",   # NEW equipment from K-Shop
    "equipment_used",    # USED equipment from Buy/Sell
    "kshop_product",     # K-Shop catalog (categories/companies/discounts)
    "buy_sell_product",  # Buy/Sell listings: animals, ads, listing ids
    "seed_info",         # seed / variety info
    "news",              # agricultural news
    "video",             # wants a list of farming videos
    "greeting",          # pure greeting / small talk
    "general",           # static app info (free?, contact, what is the app)
    "navigation",        # how-to steps / where is a screen
    "ambiguous",         # cannot decide — clarification needed
]

AmbiguityScenario = Literal[
    "crop", "equipment", "equipment_price", "animal",
    "seed", "product", "price", "location",
]

QueryType = Literal["specific_search", "list_all", "count", "general_knowledge"]

# Intents answered from the database (dispatched to the SQL flow).
SQL_INTENTS = {
    "crop_price", "equipment_kshop", "equipment_used", "kshop_product",
    "buy_sell_product", "seed_info", "news", "video",
}

# New-taxonomy intent → legacy intent keys still used by the SQL flow
# (INTENT_TO_TABLES / entity_normalizer). Removed in Phase 3 when the SQL
# flow consumes the frame natively.
INTENT_TO_LEGACY_KEY = {
    "news":  "local_news",
    "video": "video_search",
}


# ── Entity blocks ────────────────────────────────────────────────────────────

class CropEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: Optional[str] = None      # crop word as typed: "ઘઉં" / "गेहूं" / "wheat"
    category: Optional[str] = None  # category word if said: "શાકભાજી", "કઠોળ", "રોકડિયા પાક"
    variety: Optional[str] = None   # sub-variety word: "કાદરી", "મગડી", "G-20"

    def surface(self) -> str:
        """Search keyword: name + variety composed (code, not LLM)."""
        parts = [p for p in (self.name, self.variety) if p]
        return " ".join(parts) if parts else (self.category or "")


class LocationEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    # Set ONLY when the user explicitly said the level word (તાલુકા, સીટી, યાર્ડ…).
    level: Optional[Literal["state", "city", "taluka", "yard"]] = None


class EquipmentEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    condition: Optional[Literal["new", "used"]] = None  # null = user didn't say


class NewsEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Optional[str] = None                       # news category word: "સરકારી યોજના", "weather"
    topics: List[str] = Field(default_factory=list)  # free-text topics: ["અતિવૃષ્ટિ", "વાવાઝોડું"]
    # Deprecated single-topic fallback — merged into topics by normalize().
    topic: Optional[str] = None


class VideoEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    topics: List[str] = Field(default_factory=list)
    # Deprecated single-topic fallback — merged into topics by normalize().
    topic: Optional[str] = None


class Constraints(BaseModel):
    """ONLY constraints the user EXPLICITLY asked for. Anything not present
    here must never appear in generated SQL (hallucination guard)."""
    model_config = ConfigDict(extra="ignore")
    price_kind: Optional[Literal["min", "max", "min_max"]] = None
    price_above: Optional[float] = None
    price_below: Optional[float] = None
    date: Optional[Literal["today", "this_week", "this_month", "latest"]] = None
    sort: Optional[Literal["cheapest", "most_expensive", "newest"]] = None
    group_by: Optional[Literal["yard", "taluka", "city"]] = None


# ── The frame ────────────────────────────────────────────────────────────────

class NLUFrame(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # — LLM-owned —
    intent: IntentType
    intent_confidence: Literal["high", "low"] = "high"
    ambiguity_scenario: Optional[AmbiguityScenario] = None
    question_en: str = ""
    crops: List[CropEntity] = Field(default_factory=list)
    locations: List[LocationEntity] = Field(default_factory=list)
    equipment: List[EquipmentEntity] = Field(default_factory=list)
    animals: List[str] = Field(default_factory=list)
    news: NewsEntity = Field(default_factory=NewsEntity)
    video: VideoEntity = Field(default_factory=VideoEntity)
    identifier: Optional[str] = None
    constraints: Constraints = Field(default_factory=Constraints)
    query_type: QueryType = "specific_search"
    is_price_query: bool = False
    is_availability_query: bool = False

    # — Code-owned (injected after validation; never asked from the LLM) —
    raw_input: str = ""
    detected_language: str = ""

    # — Derived —
    @property
    def needs_clarification(self) -> bool:
        return self.intent == "ambiguous"

    @property
    def is_sql_intent(self) -> bool:
        return self.intent in SQL_INTENTS

    @property
    def legacy_intent_key(self) -> str:
        """Old INTENT_TO_TABLES key for the Phase-1 SQL-flow bridge."""
        return INTENT_TO_LEGACY_KEY.get(self.intent, self.intent)

    def primary_keyword(self) -> str:
        """Single search keyword for the Phase-1 bridge (entity_normalizer
        handles one keyword until Phase 2 makes resolution multi-entity).
        Priority: subject entities first, locations last."""
        for crop in self.crops:
            s = crop.surface()
            if s:
                return s
        for eq in self.equipment:
            if eq.name:
                return eq.name
        for animal in self.animals:
            if animal:
                return animal
        if self.video.topics:
            return self.video.topics[0]
        if self.news.topics:
            return self.news.topics[0]
        if self.news.type:
            return self.news.type
        for loc in self.locations:
            if loc.name:
                return loc.name
        return ""

    def normalize(self) -> "NLUFrame":
        """Deterministic post-validation fixes (code-side sanity rules)."""
        # Merge deprecated single-topic fields into the topics arrays.
        if self.news.topic and self.news.topic not in self.news.topics:
            self.news.topics.append(self.news.topic)
            self.news.topic = None
        if self.video.topic and self.video.topic not in self.video.topics:
            self.video.topics.append(self.video.topic)
            self.video.topic = None
        # Scenario only meaningful when ambiguous; default it when missing.
        if self.intent == "ambiguous" and not self.ambiguity_scenario:
            self.ambiguity_scenario = "product"
        if self.intent != "ambiguous":
            self.ambiguity_scenario = None
        # Count/list questions are never "ambiguous entity" cases.
        if self.query_type in ("count", "list_all") and self.intent == "ambiguous":
            self.intent_confidence = "low"
        return self
