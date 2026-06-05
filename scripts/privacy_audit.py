"""Audit full_schema.json against the chatbot privacy policy.

This script does not change production policy or generated tools. It scans the
raw schema and writes a review report so new private tables/columns are visible
before SQL flow is enabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.privacy_policy import get_privacy_policy


def audit_schema(full_schema_path: Path, policy_path: Path) -> Dict[str, Any]:
    policy = get_privacy_policy(str(policy_path))
    with full_schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    report: Dict[str, Any] = {
        "policy_version": policy.version,
        "policy_hash": policy.policy_hash,
        "full_schema": str(full_schema_path),
        "unknown_denied_tables": [],
        "excluded_tables_present": [],
        "join_only_tables_present": [],
        "allowed_tables_with_sensitive_columns": [],
    }

    for table in schema.get("tables", []):
        table_name = table.get("table_name", "")
        table_entry = {"table": table_name}

        if policy.is_excluded_table(table_name):
            report["excluded_tables_present"].append(table_entry)
            continue

        if policy.is_join_only_table(table_name):
            report["join_only_tables_present"].append({
                **table_entry,
                "allowed_columns": sorted(policy.allowed_join_columns(table_name)),
            })
            continue

        if not policy.is_queryable_table(table_name):
            report["unknown_denied_tables"].append(table_entry)
            continue

        sensitive_columns: List[str] = []
        user_id_columns: List[str] = []
        for column in table.get("columns", []):
            col_name = column.get("name", "")
            if policy.is_blocked_column_name(col_name):
                sensitive_columns.append(col_name)
            elif policy.is_user_identifier_column(col_name):
                user_id_columns.append(col_name)

        if sensitive_columns or user_id_columns:
            report["allowed_tables_with_sensitive_columns"].append({
                **table_entry,
                "blocked_columns": sensitive_columns,
                "user_identifier_columns": user_id_columns,
            })

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit raw DB schema privacy risks.")
    parser.add_argument("--schema", default="app/schemas/full_schema.json")
    parser.add_argument("--policy", default="app/schemas/privacy_policy.json")
    parser.add_argument("--output", default="reports/privacy_audit.json")
    args = parser.parse_args()

    report = audit_schema(Path(args.schema), Path(args.policy))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    risk_count = (
        len(report["unknown_denied_tables"])
        + len(report["excluded_tables_present"])
        + len(report["allowed_tables_with_sensitive_columns"])
    )
    print(f"Privacy audit written to {output}")
    print(f"Review items: {risk_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
