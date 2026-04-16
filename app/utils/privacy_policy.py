"""Deterministic privacy policy for chatbot database access.

The raw database schema can contain private and operational tables. This module
is the single source of truth for which tables/columns may reach schema tools,
SQL execution, logs, and answer generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


DEFAULT_POLICY_PATH = Path("app/schemas/privacy_policy.json")


def _as_set(values: Iterable[str] | None) -> Set[str]:
    return {str(v).strip().lower() for v in (values or []) if str(v).strip()}


def _compile_patterns(values: Iterable[str] | None) -> List[re.Pattern]:
    return [re.compile(str(v), re.IGNORECASE) for v in (values or []) if str(v).strip()]


@dataclass(frozen=True)
class PrivacyPolicy:
    """Loaded privacy policy with helper methods for generation and runtime."""

    version: str
    mode: str
    deny_unknown_tables: bool
    allowed_tables: Set[str]
    excluded_tables: Set[str]
    join_only_tables: Mapping[str, Set[str]]
    auto_exclude_table_patterns: Sequence[re.Pattern]
    blocked_column_patterns: Sequence[re.Pattern]
    user_identifier_column_patterns: Sequence[re.Pattern]
    source_path: str
    policy_hash: str

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "PrivacyPolicy":
        policy_path = Path(path)
        with policy_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        join_only_raw = raw.get("join_only_tables", {})
        join_only = {
            table.lower(): _as_set(config.get("allowed_columns", []))
            for table, config in join_only_raw.items()
        }
        canonical = json.dumps(raw, sort_keys=True, ensure_ascii=False)
        return cls(
            version=str(raw.get("version", "unknown")),
            mode=str(raw.get("mode", "public_plus_names")),
            deny_unknown_tables=bool(raw.get("deny_unknown_tables", True)),
            allowed_tables=_as_set(raw.get("allowed_tables", [])),
            excluded_tables=_as_set(raw.get("excluded_tables", [])),
            join_only_tables=join_only,
            auto_exclude_table_patterns=_compile_patterns(raw.get("auto_exclude_table_patterns", [])),
            blocked_column_patterns=_compile_patterns(raw.get("blocked_column_patterns", [])),
            user_identifier_column_patterns=_compile_patterns(raw.get("user_identifier_column_patterns", [])),
            source_path=str(policy_path),
            policy_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
        )

    def is_auto_excluded_table(self, table_name: str) -> bool:
        name = (table_name or "").lower()
        return any(pattern.search(name) for pattern in self.auto_exclude_table_patterns)

    def is_excluded_table(self, table_name: str) -> bool:
        name = (table_name or "").lower()
        return name in self.excluded_tables or self.is_auto_excluded_table(name)

    def is_queryable_table(self, table_name: str) -> bool:
        name = (table_name or "").lower()
        if self.is_excluded_table(name) or name in self.join_only_tables:
            return False
        if name in self.allowed_tables:
            return True
        return not self.deny_unknown_tables

    def is_join_only_table(self, table_name: str) -> bool:
        return (table_name or "").lower() in self.join_only_tables

    def is_sql_visible_table(self, table_name: str) -> bool:
        return self.is_queryable_table(table_name) or self.is_join_only_table(table_name)

    def allowed_join_columns(self, table_name: str) -> Set[str]:
        return set(self.join_only_tables.get((table_name or "").lower(), set()))

    def is_blocked_column_name(self, column_name: str) -> bool:
        name = (column_name or "").lower()
        return any(pattern.search(name) for pattern in self.blocked_column_patterns)

    def is_user_identifier_column(self, column_name: str) -> bool:
        name = (column_name or "").lower()
        return any(pattern.search(name) for pattern in self.user_identifier_column_patterns)

    def is_safe_tool_column(self, table_name: str, column_name: str) -> bool:
        table = (table_name or "").lower()
        column = (column_name or "").lower()
        if self.is_join_only_table(table):
            return column in self.allowed_join_columns(table)
        if not self.is_queryable_table(table):
            return False
        if self.is_blocked_column_name(column):
            return False
        if self.is_user_identifier_column(column):
            return False
        return True

    def is_safe_selected_column(self, table_name: Optional[str], column_name: str) -> bool:
        table = (table_name or "").lower()
        column = (column_name or "").lower()
        if self.is_blocked_column_name(column):
            return False
        if self.is_join_only_table(table):
            return column in self.allowed_join_columns(table)
        if table and not self.is_queryable_table(table):
            return False
        if self.is_user_identifier_column(column):
            return False
        return True

    def is_safe_result_key(self, key: str) -> bool:
        name = (key or "").lower()
        if self.is_blocked_column_name(name):
            return False
        if self.is_user_identifier_column(name):
            return False
        return True

    def filter_tool_names(self, tool_names: Iterable[str] | None) -> List[str]:
        filtered: List[str] = []
        for tool in tool_names or []:
            name = str(tool).strip()
            table = name.replace("query_", "", 1)
            if name.startswith("query_") and self.is_queryable_table(table) and name not in filtered:
                filtered.append(name)
        return filtered

    def sanitize_rows(self, rows: Sequence[Mapping[str, Any]] | None) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for row in rows or []:
            sanitized.append({
                key: value
                for key, value in dict(row).items()
                if self.is_safe_result_key(str(key))
            })
        return sanitized

    def sanitize_query_result(self, result: Any) -> Any:
        from app.models.chat_models import QueryResult

        rows = self.sanitize_rows(getattr(result, "rows", []))
        return QueryResult(
            table_name=getattr(result, "table_name", "unknown"),
            sql=getattr(result, "sql", ""),
            rows=rows,
            row_count=len(rows),
            execution_time=getattr(result, "execution_time", 0.0),
        )

    def sanitize_query_results(self, results: Sequence[Any] | None) -> List[Any]:
        return [self.sanitize_query_result(result) for result in (results or [])]


def _default_policy_path() -> str:
    return os.environ.get("SCHEMA_PRIVACY_POLICY_PATH", str(DEFAULT_POLICY_PATH))


@lru_cache(maxsize=8)
def load_privacy_policy(path: str | None = None) -> PrivacyPolicy:
    return PrivacyPolicy.from_file(path or _default_policy_path())


def get_privacy_policy(path: str | None = None) -> PrivacyPolicy:
    return load_privacy_policy(path)

