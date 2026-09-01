#!/usr/bin/env python3
"""Compose and audit the official Lenormand submission CSV.

Task-1 supplies risk_level/evidence; the factor system supplies factors. All
rows are aligned against the organizer test file rather than by file order.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

import pandas as pd


RISK_LABELS = {"Indicator", "Ideation", "Behavior", "Attempt"}


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    raise ValueError(f"Unsupported table: {path}")


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def parse_factor_cell(value: object) -> list[str]:
    if isinstance(value, list):
        result = value
    else:
        text = str(value).strip()
        if not text:
            return []
        result = ast.literal_eval(text)
    if not isinstance(result, (list, tuple)) or not all(isinstance(x, str) for x in result):
        raise ValueError(f"Invalid factors cell: {value!r}")
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate factor in cell: {value!r}")
    return list(result)


def post_column(frame: pd.DataFrame) -> str:
    for name in ("post", "text", "content"):
        if name in frame.columns:
            return name
    raise ValueError("Test file has no post/text/content column for evidence audit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--task1", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    test = read_table(args.test).fillna("")
    task1 = read_table(args.task1).fillna("")
    factors = read_table(args.factors).fillna("")
    require_columns(test, {"row_id"}, args.test)
    require_columns(task1, {"row_id", "risk_level", "evidence"}, args.task1)
    require_columns(factors, {"row_id", "factors"}, args.factors)

    for name, frame in (("test", test), ("task1", task1), ("factors", factors)):
        frame["row_id"] = frame["row_id"].astype(str)
        if frame["row_id"].duplicated().any():
            raise ValueError(f"Duplicate row_id in {name}")

    expected = test["row_id"].tolist()
    if set(task1["row_id"]) != set(expected) or set(factors["row_id"]) != set(expected):
        raise ValueError("Task-1/factor row IDs do not match the organizer test set")

    task1 = task1.set_index("row_id").loc[expected]
    factors = factors.set_index("row_id").loc[expected]
    result = pd.DataFrame(
        {
            "row_id": expected,
            "risk_level": task1["risk_level"].astype(str).tolist(),
            "evidence": task1["evidence"].astype(str).tolist(),
            "factors": factors["factors"].tolist(),
        }
    )

    invalid_risk = sorted(set(result["risk_level"]) - RISK_LABELS)
    if invalid_risk:
        raise ValueError(f"Invalid risk labels: {invalid_risk}")

    parsed = [parse_factor_cell(value) for value in result["factors"]]
    result["factors"] = [repr(items) for items in parsed]

    posts = dict(zip(test["row_id"].astype(str), test[post_column(test)].astype(str)))
    invalid_evidence: list[tuple[str, str]] = []
    for row in result.itertuples(index=False):
        phrases = [p.strip() for p in str(row.evidence).split(";") if p.strip()]
        for phrase in phrases:
            if phrase not in posts[row.row_id]:
                invalid_evidence.append((row.row_id, phrase))
    if invalid_evidence:
        preview = invalid_evidence[:5]
        raise ValueError(f"Evidence is not an exact post substring; examples: {preview}")

    result.loc[result["risk_level"].eq("Indicator"), "evidence"] = ""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        {
            "output": str(args.output),
            "rows": len(result),
            "columns": result.columns.tolist(),
            "mean_factors": sum(map(len, parsed)) / max(1, len(parsed)),
            "sha256": digest,
        }
    )


if __name__ == "__main__":
    main()

