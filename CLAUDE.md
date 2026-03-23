# CLAUDE.md — Krushi AI Backend

> This file is the single source of truth for any AI assistant (or developer) working on this codebase.
> Read this before touching any file.

---

## Project Overview

**Krushi** is a FastAPI-based AI backend for an Indian agricultural marketplace platform called **Krushi Ratn**. It serves farmers with two independent AI-powered subsystems running in a single app:

1. **AI Chatbot** — WebSocket-based conversational agent that answers natural-language (and voice) queries about the marketplace database using a 3-step LLM pipeline.
2. **Fake Detection Service** — REST API that analyses video transcripts for agricultural misinformation using Groq LLM and stores results in MySQL.

**Target users:** Indian farmers (Gujarati and Hindi language support is critical)
**Database:** `krushi_node` — MySQL, 43 tables
**Primary language:** Python 3.10+

---

## Repository Structure

```
app/
├── main.py                        # FastAPI app entry point, lifespan, routing
├── core/
│   ├── config.py                  # All settings via pydantic-settings (.env)
│   ├── database.py                # Async MySQL pool manager (aiomysql), auto-reconnect
│   └── logger.py                  # Structured logger setup
├── models/
│   └── chat_models.py             # All Pydantic request/response models
├── schemas/
│   ├── condensed_schema.json      # Lightweight schema — 43 tables, used in Step 1 (tool selection)
│   ├── full_schema.json           # Detailed schema — used internally
│   ├── general_questions.json     # FAQ/general Q&A data
│   ├── navigation.json            # UI navigation config
│   └── tools/                     # One JSON tool file per DB table (20+ files)
│       ├── categories_tool.json
│       ├── kshop_products_tool.json
│       ├── buy_sell_products_tool.json
│       └── ...
├── cache/
│   └── query_cache.json           # Persistent query cache (hash → SQL + tools)
├── fake_detection_api.py          # FastAPI router — CRUD for fake_identification table
├── fake_detection_service.py      # Groq analyser + FakeDetectionDB (own pool)
└── fake_detection_models.py       # Dataclass + CREATE TABLE SQL for fake_identification
```

> **Note:** `services/`, `websocket/`, `utils/` directories exist but were not included in this snapshot. They contain the orchestrator, WebSocket handler, LLM manager, STT/TTS services, and schema generator.

---

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Framework | FastAPI + uvicorn | Async throughout |
| Database | MySQL via `aiomysql` | Connection pool, auto-reconnect |
| Primary LLM | Groq — `llama-3.3-70b-versatile` | Tool selection + answer generation |
| SQL LLM | OpenAI `gpt-4o` (or Groq if no key) | SQL generation only — better accuracy |
| STT / TTS | Sarvam AI (`saarika:v2` / `bulbul:v1`) | Primary — Indian language support |
| STT fallback | Deepgram | When Sarvam unavailable |
| TTS fallback | gTTS | When Sarvam unavailable |
| Translation | Sarvam AI `mayura:v1` | Hindi/Gujarati ↔ English |
| Config | `pydantic-settings` | Loaded from `.env` |
| Transport | WebSocket (chat) + REST (fake detection) | |

---

## Environment Variables (.env)

```env
# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=krushi_node
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# LLM Keys
GROQ_API_KEY=          # Required — primary LLM
OPENAI_API_KEY=        # Optional — improves SQL generation quality

# Sarvam AI (Indian language STT/TTS/Translation)
SARVAM_API_KEY=        # Primary voice service
SARVAM_DEFAULT_LANGUAGE=hi-IN
SARVAM_DEFAULT_SPEAKER=meera

# Deepgram (STT fallback)
DEEPGRAM_API_KEY=

# App
APP_HOST=0.0.0.0
APP_PORT=8000
ENVIRONMENT=development   # development | production
LOG_LEVEL=INFO

# LLM Config
GROQ_MODEL=llama-3.3-70b-versatile
OPENAI_MODEL=gpt-4o
SQL_PROVIDER=auto         # auto | openai | groq

# Rate Limits
GROQ_RPM_LIMIT=30
GROQ_TOKENS_PER_MINUTE=14400
OPENAI_RPM_LIMIT=60
OPENAI_TOKENS_PER_MINUTE=90000

# Paths
SCHEMA_DIR=app/schemas
TOOLS_DIR=app/schemas/tools
```

---

## API Endpoints

### Chatbot
| Method | Path | Description |
|--------|------|-------------|
| `WS` | `/ws/chat` | WebSocket — text and voice chat |
| `GET` | `/` | Root health ping |
| `GET` | `/health` | Full health check (DB + LLMs) |
| `GET` | `/api/stats` | DB table count + feature flags |

### Fake Detection
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/fake-detection` | Analyse transcript + store result |
| `GET` | `/api/v1/fake-detection` | List all results (paginated) |
| `GET` | `/api/v1/fake-detection/{id}` | Get one result by ID |
| `DELETE` | `/api/v1/fake-detection/{id}` | Delete one result |

---

## Chatbot Pipeline (3-Step Agent)

Every user message (text or voice) goes through this pipeline:

```
User Input (text or base64 audio)
        │
        ▼
[STT — Sarvam AI / Deepgram]       ← only if voice input
        │
        ▼
Step 1 — Tool Selection (Groq)
  Input:  user query + condensed_schema.json (43 table summaries)
  Output: list of relevant tool names (e.g. ["query_kshop_products", "query_categories"])
        │
        ▼
Step 2 — SQL Generation (OpenAI GPT-4o preferred / Groq fallback)
  Input:  user query + full tool schemas for selected tools
  Output: list of { table_name, sql } objects
  Rules:  SELECT only, always WHERE deleted_at IS NULL, always LIMIT
        │
        ▼
Step 3 — Answer Generation (Groq)
  Input:  user query + all query results (rows as JSON)
  Output: natural language answer, streamed word-by-word
        │
        ▼
[TTS — Sarvam AI / gTTS]           ← if voice output requested
        │
        ▼
WebSocket Response (streaming text + optional audio chunks)
```

### Query Cache
Queries are cached in `app/cache/query_cache.json` keyed by a hash of the normalized query. Cache hits skip Steps 1 & 2 entirely and jump straight to SQL execution.

---

## Fake Detection Pipeline

```
POST /api/v1/fake-detection
  { transcript_text, video_post_id? }
        │
        ▼
GroqFakeDetector.analyse(transcript)
  → 5-step Groq prompt (agriculture relevance + misinformation analysis)
  → Returns: is_farming_related, farming_relevance_score, is_misinformation,
             misinformation_risk_score, severity, decision, reason
        │
        ▼
FakeDetectionDB.insert(transcript, analysis, video_post_id)
  → Stores in fake_identification table
  → video_post_id is FK to video_posts.id (nullable, CASCADE DELETE)
        │
        ▼
Returns full DetectResponse record
```

**Decision values:** `approve` | `flag` | `reject`
**Severity values:** `none` | `low` | `moderate` | `high`

---

## Database — Key Tables

### Marketplace
| Table | Purpose |
|-------|---------|
| `users` | Farmer/user accounts — referenced as `user_id`, `seller_id`, `buyer_id`, `farmer_id` |
| `categories` | Main product categories |
| `sub_categories` | Sub-categories under categories — used in `products`, `seeds`, `user_products` |
| `products` | Crop/commodity market prices — `min_price`, `max_price`, `price_date`, `yard_id` |
| `yards` | Market yard (mandi) locations — `city_id`, `state_id`, `taluka_id` |
| `buy_sell_products` | Farmer marketplace listings — `seller_id`, `price`, `quantity`, `status` |
| `buy_sell_orders` | Transactions — `buyer_id`, `seller_id` |
| `kshop_products` | K-Shop products (Gujarati names) — `kshop_company_id`, `price`, `status=1` |
| `kshop_orders` | K-Shop order records |
| `farmer_orders` | Farmer-perspective orders — `user_id`, `company_id` |
| `seeds` | Seed products with variety info |

### Media
| Table | Purpose |
|-------|---------|
| `media` | Media file references (images, documents) |
| `mediables` | **Polymorphic join table** — links media to any model (`mediable_type` + `mediable_id`) |

> ⚠️ **Important:** To get images for any record, always JOIN through `mediables` using `mediable_type` (model name) and `mediable_id` (record PK). Do not assume images are stored directly on the record.

### Video Platform
| Table | Purpose |
|-------|---------|
| `video_posts` | Educational agricultural videos — `user_id` (creator), `video_category_id`, `views_count` |
| `video_categories` | Video category labels |
| `video_comments` | Comments with `parent_comment_id` for threading |
| `video_likes`, `video_saves`, `video_shares`, `video_views` | Engagement tracking |
| `fake_identification` | Misinformation analysis results — `video_post_id` FK |

### Geography
| Table | Purpose |
|-------|---------|
| `states` | Indian states |
| `cities` | Cities within states |
| `talukas` | Sub-districts within cities |

---

## Database Rules (ALWAYS Follow)

1. **READ-ONLY** — only `SELECT` queries are ever generated. No `INSERT`, `UPDATE`, `DELETE` from the chatbot.
2. **Soft deletes** — every query on soft-deletable tables must include `WHERE <table>.deleted_at IS NULL`.
3. **Always use LIMIT** — never `SELECT *` without a `WHERE` clause and `LIMIT`.
4. **JOIN type matters** — use `JOIN` for required FKs, `LEFT JOIN` for nullable FKs.
5. **Language** — product names in K-Shop are in Gujarati. Use `LIKE '%gujarati_word%'` for search.

---

## Database Connection Architecture

There are **two separate aiomysql connection pools**:

| Pool | Owner | Size | Purpose |
|------|-------|------|---------|
| `db_manager` (DatabaseManager) | `core/database.py` | 10–30 connections | Chatbot SQL queries |
| `_fake_db` (FakeDetectionDB) | `fake_detection_service.py` | 2–5 connections | fake_identification table only |

This is intentional isolation. Do not merge them. Each has its own `init()` / `close()` called from lifespan.

**Auto-reconnect:** `db_manager` retries up to 3 times on lost-connection MySQL errors (codes 2006, 2013, 2055) before raising.

---

## WebSocket Message Protocol

All WebSocket messages are JSON with a `type` field:

### Client → Server
| type | Fields | Description |
|------|--------|-------------|
| `text_input` | `text`, `session_id?` | Text message |
| `audio_input` | `audio_data` (base64), `audio_format`, `session_id?` | Voice message |
| `control` | `action`: `clear_history` \| `stop_generation` | Control commands |

### Server → Client
| type | Fields | Description |
|------|--------|-------------|
| `text_output` | `text`, `is_complete`, `full_text?` | Streamed text response |
| `audio_output` | `audio_data` (base64), `audio_format`, `is_complete` | TTS audio chunks |
| `status` | `status`, `details?` | Pipeline stage updates |
| `error` | `error`, `error_type?` | Error messages |

**Status values:** `thinking` → `generating_sql` → `executing_query` → `generating_answer` → `complete`

---

## Known Issues & Gotchas

### 1. HealthCheckResponse type mismatch ⚠️
`chat_models.py` defines `status: Literal["healthy", "unhealthy"]` but `main.py` health check can return `"degraded"`. This will cause a Pydantic validation error at runtime when the DB is up but the primary LLM is down.

**Fix:** Change the Literal in `HealthCheckResponse` to include `"degraded"`:
```python
status: Literal["healthy", "degraded", "unhealthy"]
```

### 2. No authentication on any endpoint
`allow_origins=["*"]` and no auth middleware. The fake detection API is fully public. Add API key auth or JWT before production.

### 3. In-memory chat history
Sessions are stored in memory only. All conversation history is lost on server restart. Consider Redis or DB-backed sessions for production.

### 4. MySQL version dependency in migrations
`FakeDetectionDB.init()` uses `information_schema` checks (compatible with MySQL 5.7+) to safely handle the `video_post_id` migration. The `MIGRATION_SQLS` list in `fake_detection_models.py` uses `IF NOT EXISTS` on `ALTER TABLE` which requires **MySQL 8.0+**. These two are inconsistent — use the `information_schema` approach everywhere.

### 5. No SQL injection protection at schema level
SQL is LLM-generated. The pipeline relies entirely on LLM compliance with the "SELECT only" rule. There is no query parser or allowlist validation after generation. Consider adding a simple SQL AST check before execution.

### 6. Query cache is file-based
`query_cache.json` is written to disk. In multi-worker deployments (e.g. `uvicorn --workers 4`) this will cause race conditions. Use Redis for shared cache in production.

---

## Current Work in Progress

### Confirmation Layer Logic & Data Sharing
**Status:** Planned (~15h estimated)

**Goal:** Classify entries in the confirmation layer as either **crops** or **products**, extract the full dataset including images, and share with Pratik.

**Key tasks:**
- Identify which tables constitute the "confirmation layer"
- Define and implement rule-based classifier (crop vs product) — likely using `categories`, `sub_categories`, `products`, `buy_sell_categories` + category name patterns
- Extract all rows + resolve images via `mediables` polymorphic join
- Export as structured CSV/JSON + images folder and hand off

**Image extraction note:** Images are not stored directly on records. They must be resolved through:
```sql
SELECT m.*
FROM media m
JOIN mediables mb ON mb.media_id = m.id
WHERE mb.mediable_type = '<ModelName>'
  AND mb.mediable_id = <record_id>
```
`mediable_type` is the Laravel model class name (e.g. `App\Models\Product`). Check existing records in `mediables` to confirm the exact string used for each model.

---

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Fill in DB credentials and API keys

# Run development server
python -m app.main
# or
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Startup sequence:**
1. MySQL connection pool initialized
2. Fake detection DB pool + table migration
3. Schema JSON files loaded from `app/schemas/`
4. LLM orchestrator initialized with schema tools
5. Server ready

---

## Adding a New DB Table to the Chatbot

1. Create `app/schemas/tools/<table_name>_tool.json` following the same structure as existing tool files (tool_name, description, table_name, columns, relationships, example_queries, notes).
2. Add an entry to `app/schemas/condensed_schema.json` with `name` and `context`.
3. Restart the server — `initialize_schemas()` loads all JSON files on startup automatically.

---

## Code Conventions

- All DB operations are `async` — never use sync drivers or `time.sleep()`
- Use `app.core.logger` (structured logging) not bare `print()`
- Pydantic models live in `app/models/chat_models.py` — add new request/response models there
- Fake detection code is fully self-contained in its 3 files — do not import chatbot internals into it
- Tool JSON files are data, not code — edit them directly, no code change needed
- Never commit API keys — all secrets via `.env` only