"""
fake_detection_service.py
─────────────────────────
Groq API analysis + aiomysql storage for fake_identification table only.

- Uses aiomysql directly (same as your database.py — no SQLAlchemy)
- Reads settings from app.core.config (your existing .env, no duplication)
- Accepts optional video_post_id from frontend (FK → video_posts.id)
- Only touches fake_identification table — all other tables untouched
- Completely independent from Krushi chatbot flow

Place at: app/fake_detection_service.py
"""
import json
import logging
from typing import Any, Dict, List, Optional

import aiomysql
from groq import Groq

from app.core.config import settings
from app.fake_detection_models import CREATE_TABLE_SQL

logger = logging.getLogger(__name__)

MAX_CHARS   = 15_000
API_TIMEOUT = 30

# ── Groq Prompt ───────────────────────────────────────────────────────────────
DETECTION_PROMPT = """
You are an expert agricultural misinformation detector for Indian farmers.

Analyse the following video transcript and determine:
1. Whether the content is related to farming/agriculture.
2. Whether the content contains misinformation, misleading claims, unsafe advice,
   or scientifically incorrect agricultural practices.
3. Whether the content should be approved, flagged, or rejected.

Carefully analyze the following translated transcript text:

----------------------
[INSERT_TRANSLATED_TEXT_HERE]
----------------------

Perform the analysis using the following reasoning steps internally:

STEP 1: Determine whether the content is related to agriculture, including:
- Crop farming
- Horticulture
- Livestock
- Fisheries
- Agricultural techniques
- Rural farming practices
- Soil, irrigation, fertilizer, pesticides
- Agricultural economics
- Sustainability

If it is NOT related to agriculture, mark it as not farming related.

STEP 2: If it IS farming-related, evaluate:
- Are the claims scientifically accurate?
- Are there exaggerated yield claims?
- Are unsafe pesticide or chemical practices suggested?
- Are unverified home remedies presented as guaranteed solutions?
- Are there scam-like product promotions?
- Could the advice cause economic loss, crop damage, environmental harm, or health risks?

STEP 3: Assign the following scores:
- farming_relevance_score (0-100)
- misinformation_risk_score (0-100)

STEP 4: Classify severity:
- "none"     → Accurate and safe
- "low"      → Minor exaggeration but not harmful
- "moderate" → Questionable or partially misleading
- "high"     → Dangerous, harmful, or clearly false

STEP 5: Make a final decision:
- "approve" → Farming-related and safe
- "flag"    → Farming-related but suspicious or partially misleading
- "reject"  → Not farming-related OR high misinformation risk

CRITICAL INSTRUCTIONS:
- Be strict and conservative in evaluation.
- Do NOT assume missing context.
- Do NOT add explanations outside JSON.
- Output ONLY valid JSON.
- Do NOT include markdown.
- Do NOT include commentary.

Return output strictly in this format:
{
  "is_farming_related": true or false,
  "farming_relevance_score": integer,
  "is_misinformation": true or false,
  "misinformation_risk_score": integer,
  "severity": "none" | "low" | "moderate" | "high",
  "decision": "approve" | "flag" | "reject",
  "reason": "Detailed professional explanation of classification, scoring logic, and risks."
}
"""


# ── Groq Detector ─────────────────────────────────────────────────────────────
class GroqFakeDetector:
    """Calls Groq API with the 5-step analysis prompt and validates response."""

    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY missing from .env")
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model  = settings.GROQ_MODEL

    def analyse(self, transcript: str) -> Dict[str, Any]:
        if not transcript or not transcript.strip():
            raise ValueError("transcript_text cannot be empty.")
        if len(transcript) > MAX_CHARS:
            raise ValueError(f"Transcript too long — max {MAX_CHARS} characters.")

        prompt = DETECTION_PROMPT.replace("[INSERT_TRANSLATED_TEXT_HERE]", transcript)

        logger.info("Calling Groq | model=%s | chars=%d", self.model, len(transcript))

        completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.1,
            max_tokens=1000,
            timeout=API_TIMEOUT,
        )

        raw    = completion.choices[0].message.content
        result = self._parse_json(raw)
        self._validate(result)

        logger.info(
            "Groq done | farming=%s(%s) | misinfo=%s(%s) | decision=%s",
            result["is_farming_related"], result["farming_relevance_score"],
            result["is_misinformation"],  result["misinformation_risk_score"],
            result["decision"],
        )
        return result

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        content = raw.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
        return json.loads(content.strip())

    @staticmethod
    def _validate(r: Dict[str, Any]) -> None:
        required = [
            "is_farming_related", "farming_relevance_score",
            "is_misinformation",  "misinformation_risk_score",
            "severity", "decision", "reason",
        ]
        for field in required:
            if field not in r:
                raise ValueError(f"Groq response missing field: '{field}'")
        if r["severity"] not in ("none", "low", "moderate", "high"):
            raise ValueError(f"Invalid severity: {r['severity']}")
        if r["decision"] not in ("approve", "flag", "reject"):
            raise ValueError(f"Invalid decision: {r['decision']}")


# ── DB — own pool, only touches fake_identification ───────────────────────────
class FakeDetectionDB:
    """
    Own aiomysql pool — completely separate from chatbot's db_manager pool.
    ONLY creates / reads / writes / deletes fake_identification rows.
    No other table is ever touched.
    """

    def __init__(self):
        self.pool   = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self.pool is not None

    def _ensure_ready(self):
        if not self.is_ready:
            raise RuntimeError("Fake detection database is not available.")

    async def init(self):
        """Create pool, ensure table exists, run any pending migrations."""
        if self._ready:
            return

        password = settings.DB_PASSWORD.strip("'\"")

        self.pool = await aiomysql.create_pool(
            host=settings.DB_HOST,
            port=int(settings.DB_PORT),
            user=settings.DB_USER,
            password=password,
            db=settings.DB_NAME,
            minsize=2,
            maxsize=5,
            autocommit=True,
            charset="utf8mb4",
            pool_recycle=1800,
            connect_timeout=10,
        )

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                # Step 1 — create table if it doesn't exist at all
                await cur.execute(CREATE_TABLE_SQL)

                # Step 2 — check if video_post_id column exists using information_schema
                # (AWS RDS MySQL does NOT support IF NOT EXISTS on ALTER TABLE)
                await cur.execute("""
                    SELECT COUNT(*) FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA = %s
                      AND TABLE_NAME   = 'fake_identification'
                      AND COLUMN_NAME  = 'video_post_id'
                """, (settings.DB_NAME,))
                row = await cur.fetchone()
                column_exists = row[0] > 0

                if not column_exists:
                    logger.info("Migrating fake_identification: adding video_post_id column + index + FK...")

                    # Add column
                    await cur.execute("""
                        ALTER TABLE fake_identification
                        ADD COLUMN video_post_id BIGINT UNSIGNED NULL DEFAULT NULL
                    """)

                    # Add index
                    await cur.execute("""
                        ALTER TABLE fake_identification
                        ADD INDEX idx_fake_video_post_id (video_post_id)
                    """)

                    # Add FK — only add if not already present
                    await cur.execute("""
                        SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS
                        WHERE TABLE_SCHEMA    = %s
                          AND TABLE_NAME      = 'fake_identification'
                          AND CONSTRAINT_NAME = 'fk_fake_identification_video'
                    """, (settings.DB_NAME,))
                    fk_row = await cur.fetchone()
                    if fk_row[0] == 0:
                        await cur.execute("""
                            ALTER TABLE fake_identification
                            ADD CONSTRAINT fk_fake_identification_video
                                FOREIGN KEY (video_post_id)
                                REFERENCES video_posts(id)
                                ON DELETE CASCADE
                                ON UPDATE NO ACTION
                        """)

                    logger.info("Migration complete — video_post_id column + FK added.")
                else:
                    logger.info("fake_identification already has video_post_id column — skipping migration.")

        self._ready = True
        logger.info(
            "FakeDetectionDB ready — fake_identification table with "
            "video_post_id FK to video_posts."
        )

    async def close(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
        self.pool = None
        self._ready = False

    # ── INSERT ────────────────────────────────────────────────────────────────
    async def insert(
        self,
        transcript:     str,
        analysis:       Dict[str, Any],
        video_post_id:  Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        INSERT one analysis row.
        video_post_id comes from the frontend — stored as FK to video_posts.id.
        If the video_post_id doesn't exist in video_posts, MySQL raises FK error
        which we surface as a 422 back to the caller.
        """
        self._ensure_ready()
        sql = """
            INSERT INTO fake_identification
                (transcript_text, is_farming_related, farming_relevance_score,
                 is_misinformation, misinformation_risk_score,
                 severity, decision, reason, video_post_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            transcript,
            int(analysis["is_farming_related"]),
            analysis["farming_relevance_score"],
            int(analysis["is_misinformation"]),
            analysis["misinformation_risk_score"],
            analysis["severity"],
            analysis["decision"],
            analysis["reason"],
            video_post_id,   # None → NULL in MySQL (allowed, no FK violation)
        )
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(sql, params)
                new_id = cur.lastrowid

        return await self.get_by_id(new_id)

    # ── SELECT ONE ────────────────────────────────────────────────────────────
    async def get_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        self._ensure_ready()
        sql = "SELECT * FROM fake_identification WHERE id = %s LIMIT 1"
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(sql, (record_id,))
                row = await cur.fetchone()
        return self._serialize(row) if row else None

    # ── SELECT ALL (paginated) ────────────────────────────────────────────────
    async def get_all(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        self._ensure_ready()
        sql = """
            SELECT * FROM fake_identification
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(sql, (limit, offset))
                rows = await cur.fetchall()
        return [self._serialize(r) for r in rows]

    # ── DELETE ────────────────────────────────────────────────────────────────
    async def delete_by_id(self, record_id: int) -> bool:
        self._ensure_ready()
        sql = "DELETE FROM fake_identification WHERE id = %s"
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (record_id,))
                return cur.rowcount > 0

    # ── serializer ────────────────────────────────────────────────────────────
    @staticmethod
    def _serialize(row: dict) -> Dict[str, Any]:
        """Convert raw DB row → clean API dict (bool, isoformat, etc.)."""
        return {
            "id":                        row["id"],
            "transcript_text":           row["transcript_text"],
            "is_farming_related":        bool(row["is_farming_related"]),
            "farming_relevance_score":   row["farming_relevance_score"],
            "is_misinformation":         bool(row["is_misinformation"]),
            "misinformation_risk_score": row["misinformation_risk_score"],
            "severity":                  row["severity"],
            "decision":                  row["decision"],
            "reason":                    row["reason"],
            "video_post_id":             row["video_post_id"],  # int or None
            "created_at":                row["created_at"].isoformat() if row["created_at"] else None,
        }


# ── Singletons ────────────────────────────────────────────────────────────────
_fake_db:       FakeDetectionDB   = FakeDetectionDB()
_fake_detector: GroqFakeDetector  = None   # lazy-init


def _get_detector() -> GroqFakeDetector:
    global _fake_detector
    if _fake_detector is None:
        _fake_detector = GroqFakeDetector()
    return _fake_detector


# ── Lifecycle (called from main.py lifespan) ──────────────────────────────────
async def init_fake_detection():
    """Call once at app startup inside lifespan."""
    await _fake_db.init()
    _get_detector()
    logger.info("✅ Fake detection service ready.")


async def close_fake_detection():
    """Call at app shutdown inside lifespan."""
    await _fake_db.close()


# ── Public functions used by fake_detection_api.py ───────────────────────────
async def analyse_and_store(
    transcript:    str,
    video_post_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analyse transcript with Groq → store in fake_identification → return full dict.
    video_post_id is optional — sent by frontend, stored as FK to video_posts.id.
    """
    analysis = _get_detector().analyse(transcript)
    record   = await _fake_db.insert(transcript, analysis, video_post_id)
    logger.info(
        "Stored fake_identification id=%d video_post_id=%s decision=%s",
        record["id"], record["video_post_id"], record["decision"],
    )
    return record


async def get_detection_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    return await _fake_db.get_by_id(record_id)


async def get_all_detections(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    return await _fake_db.get_all(limit=limit, offset=offset)


async def delete_detection(record_id: int) -> bool:
    return await _fake_db.delete_by_id(record_id)
