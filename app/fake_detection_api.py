"""
fake_detection_api.py
─────────────────────
FastAPI router — full CRUD for fake_identification table ONLY.
Zero access to any other Krushi table.
Zero interference with chatbot routing.

Place at: app/fake_detection_api.py

Wire in app/main.py:
    from app.fake_detection_api import router as fake_router
    app.include_router(fake_router, prefix="/api/v1")
"""
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, validator

from app.fake_detection_service import (
    get_all_detections,
    get_detection_by_id,
    _get_detector,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/fake-detection",
    tags=["Fake Detection"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    transcript_text: str

    @validator("transcript_text")
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("transcript_text cannot be empty.")
        return v.strip()


class DetectResponse(BaseModel):
    id:                        int
    transcript_text:           str
    is_farming_related:        bool
    farming_relevance_score:   int
    is_misinformation:         bool
    misinformation_risk_score: int
    severity:                  str
    decision:                  str
    reason:                    str
    created_at:                str


class AnalyseOnlyResponse(BaseModel):
    """Response for analyse-only endpoint — no DB storage."""
    is_farming_related:        bool
    farming_relevance_score:   int
    is_misinformation:         bool
    misinformation_risk_score: int
    severity:                  str
    decision:                  str
    reason:                    str


# ── ANALYSE ONLY — no DB storage ─────────────────────────────────────────────
@router.post(
    "/analyse",
    response_model=AnalyseOnlyResponse,
    status_code=200,
    summary="Analyse a transcript — result returned to frontend only, NOT stored in DB",
)
async def analyse_only(payload: DetectRequest) -> Dict[str, Any]:
    """
    **POST /api/v1/fake-detection/analyse**

    Sends transcript to Groq LLM (5-step analysis) and returns the result
    directly to the frontend. Nothing is written to the database.

    - `transcript_text` — required, the video transcript text
    """
    try:
        return _get_detector().analyse(payload.transcript_text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Analyse-only failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")

# ── READ ALL — paginated list ─────────────────────────────────────────────────
@router.get(
    "",
    response_model=List[DetectResponse],
    summary="List all detection results (newest first)",
)
async def list_detections(
    limit:  int = Query(default=50, ge=1, le=200, description="Max rows to return"),
    offset: int = Query(default=0,  ge=0,         description="Rows to skip"),
) -> List[Dict[str, Any]]:
    """
    **GET /api/v1/fake-detection?limit=50&offset=0**

    Paginated list from `fake_identification`, newest first.
    """
    try:
        return await get_all_detections(limit=limit, offset=offset)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("List detections failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {str(exc)}")


# ── READ ONE ──────────────────────────────────────────────────────────────────
@router.get(
    "/{record_id}",
    response_model=DetectResponse,
    summary="Get a single detection result by ID",
)
async def get_detection(record_id: int) -> Dict[str, Any]:
    """
    **GET /api/v1/fake-detection/{id}**

    Fetch one stored result by primary key.
    """
    try:
        result = await get_detection_by_id(record_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No record found with id={record_id}",
            )
        return result
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Get detection failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {str(exc)}")
