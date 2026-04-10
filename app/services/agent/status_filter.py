"""
Status Filter Layer — post-SQL, pre-LLM.

Filters rows from QueryResult objects where a status column indicates
the item is inactive, sold out, rejected, cancelled, etc.

Only rows with a recognised KEEP status (or no status column at all) are
passed to the LLM.  The raw unfiltered rows are preserved separately so
the frontend can receive both.

Extending later:
  - New column name  → add to STATUS_COLUMNS
  - New "keep" value → add to KEEP_VALUES
"""

from typing import List, Tuple
from app.models.chat_models import QueryResult
from app.core.logger import get_agent_logger

logger = get_agent_logger()

# ── Configuration ──────────────────────────────────────────────────────────────

# Column names that carry an active/inactive signal.
# Add new names here if you discover more in the DB.
STATUS_COLUMNS: List[str] = [
    "status",
]

# Values that mean the row IS active/open — keep these rows.
# Everything else (including unknown values) is treated as inactive and removed.
# Stored as a set of lowercase strings for O(1) lookup.
# Numeric 1 is also included as int for safety.
KEEP_VALUES = {
    1,           # numeric active
    "1",         # string "1"
    "active",
    "available",
    "running",
    "confirmed",
}


# ── Core filter ────────────────────────────────────────────────────────────────

def _get_status_column(row: dict) -> str | None:
    """Return the first matching status column name found in the row, or None."""
    row_keys_lower = {k.lower(): k for k in row.keys()}
    for col in STATUS_COLUMNS:
        if col.lower() in row_keys_lower:
            return row_keys_lower[col.lower()]
    return None


def _is_active_row(row: dict) -> bool:
    """
    Return True if the row should be kept.

    Rules:
      - No status column present  → keep (pass through)
      - Status value in KEEP_VALUES → keep
      - Status value NOT in KEEP_VALUES (unknown string, unknown numeric, etc.) → remove
    """
    status_col = _get_status_column(row)
    if status_col is None:
        return True  # table has no status column — keep row as-is

    raw_value = row[status_col]

    # Normalise: strip whitespace, compare case-insensitively for strings
    if isinstance(raw_value, str):
        normalised = raw_value.strip().lower()
    elif isinstance(raw_value, int):
        normalised = raw_value  # keep as int for numeric check
    else:
        # Unexpected type (Decimal, float, etc.) — treat as inactive
        normalised = str(raw_value).strip().lower()

    # Check both the original int and string forms
    return normalised in KEEP_VALUES or raw_value in KEEP_VALUES


def filter_query_results(
    query_results: List[QueryResult],
) -> Tuple[List[QueryResult], bool]:
    """
    Apply status filtering to a list of QueryResult objects.

    Returns:
        filtered_results  — new list of QueryResult with inactive rows removed
        any_filtered      — True if at least one row was removed across all tables

    For each QueryResult:
      - If no row has any STATUS_COLUMNS key → result is returned unchanged
      - Otherwise rows are filtered; row_count is updated to match
    """
    if not query_results:
        return [], False

    filtered_results: List[QueryResult] = []
    any_filtered = False

    for qr in query_results:
        if not qr.rows:
            filtered_results.append(qr)
            continue

        # Check whether this table even has a status column (check first row as probe)
        probe_col = _get_status_column(qr.rows[0])
        if probe_col is None:
            # No status column in this table — pass through unchanged
            filtered_results.append(qr)
            continue

        # Filter rows
        kept_rows = [row for row in qr.rows if _is_active_row(row)]
        removed   = len(qr.rows) - len(kept_rows)

        if removed > 0:
            any_filtered = True
            logger.info(
                f"🔍 STATUS FILTER | table={qr.table_name} | "
                f"original={len(qr.rows)} rows | kept={len(kept_rows)} | removed={removed}"
            )

        # Build a new QueryResult with filtered rows; preserve everything else
        filtered_qr = QueryResult(
            table_name     = qr.table_name,
            sql            = qr.sql,
            rows           = kept_rows,
            row_count      = len(kept_rows),
            execution_time = qr.execution_time,
        )
        filtered_results.append(filtered_qr)

    return filtered_results, any_filtered