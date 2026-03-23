"""
fake_detection_models.py
─────────────────────────
Data model for fake_identification table.
No SQLAlchemy ORM — matches your project's raw aiomysql pattern.

Place at: app/fake_detection_models.py
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class FakeIdentification:
    """Mirrors every column in the fake_identification MySQL table."""
    id:                        int
    transcript_text:           str
    is_farming_related:        bool
    farming_relevance_score:   int
    is_misinformation:         bool
    misinformation_risk_score: int
    severity:                  str            # none | low | moderate | high
    decision:                  str            # approve | flag | reject
    reason:                    str
    video_post_id:             Optional[int]  # FK → video_posts.id (sent from frontend)
    created_at:                datetime

    def to_dict(self) -> dict:
        return {
            "id":                        self.id,
            "transcript_text":           self.transcript_text,
            "is_farming_related":        self.is_farming_related,
            "farming_relevance_score":   self.farming_relevance_score,
            "is_misinformation":         self.is_misinformation,
            "misinformation_risk_score": self.misinformation_risk_score,
            "severity":                  self.severity,
            "decision":                  self.decision,
            "reason":                    self.reason,
            "video_post_id":             self.video_post_id,
            "created_at":                self.created_at.isoformat() if self.created_at else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CREATE TABLE
# Safe to run every startup — IF NOT EXISTS means it skips if already there.
# video_post_id is NULLABLE so the API works even if frontend sends no id.
# FK CASCADE DELETE means: if video_posts row is deleted, this row auto-deletes.
# ─────────────────────────────────────────────────────────────────────────────
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS fake_identification (
    id                        INT              NOT NULL AUTO_INCREMENT PRIMARY KEY,
    transcript_text           LONGTEXT         NOT NULL,
    is_farming_related        TINYINT(1)       NOT NULL DEFAULT 0,
    farming_relevance_score   INT              NOT NULL DEFAULT 0,
    is_misinformation         TINYINT(1)       NOT NULL DEFAULT 0,
    misinformation_risk_score INT              NOT NULL DEFAULT 0,
    severity                  VARCHAR(20)      NOT NULL DEFAULT 'none',
    decision                  VARCHAR(20)      NOT NULL DEFAULT 'approve',
    reason                    LONGTEXT         NOT NULL,
    video_post_id             BIGINT UNSIGNED  NULL DEFAULT NULL,
    created_at                DATETIME         NOT NULL DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_fake_video_post_id (video_post_id),

    CONSTRAINT fk_fake_identification_video
        FOREIGN KEY (video_post_id)
        REFERENCES video_posts(id)
        ON DELETE CASCADE
        ON UPDATE NO ACTION

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

# ─────────────────────────────────────────────────────────────────────────────
# MIGRATION SQLS
# Only needed if fake_identification table ALREADY EXISTS from a previous deploy
# without the video_post_id column. Run these once manually on your DB, or
# the service will run them automatically at startup via init_fake_detection().
# MySQL 8.0+ supports IF NOT EXISTS on ALTER — safe to run multiple times.
# ─────────────────────────────────────────────────────────────────────────────
MIGRATION_SQLS = [
    # Step 1 — add column
    """
    ALTER TABLE fake_identification
    ADD COLUMN IF NOT EXISTS
        video_post_id BIGINT UNSIGNED NULL DEFAULT NULL;
    """,

    # Step 2 — add index
    """
    ALTER TABLE fake_identification
    ADD INDEX IF NOT EXISTS
        idx_fake_video_post_id (video_post_id);
    """,

    # Step 3 — add FK constraint
    """
    ALTER TABLE fake_identification
    ADD CONSTRAINT IF NOT EXISTS fk_fake_identification_video
        FOREIGN KEY (video_post_id)
        REFERENCES video_posts(id)
        ON DELETE CASCADE
        ON UPDATE NO ACTION;
    """,
]