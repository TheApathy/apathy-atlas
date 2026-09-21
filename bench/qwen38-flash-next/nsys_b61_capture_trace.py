# SPDX-License-Identifier: AGPL-3.0-only
"""Fail-closed Nsight kernel-family by CUDA-graph attribution checks."""

from __future__ import annotations

import csv
import re
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

FAMILIES = {
    "attention": re.compile(r"paged_decode_attn|(?:^|[^a-z])attention(?:[^a-z]|$)"),
    "gdn": re.compile(r"gated_delta_rule|compute_gdn|(?:^|[^a-z])gdn(?:[^a-z]|$)"),
    "moe": re.compile(r"(?:^|_)moe(?:_|$)|(?:^|_)expert(?:_|$)"),
    "ple": re.compile(r"qwen4_ple_"),
    "terminal": re.compile(r"(?:^|_)argmax(?:_|$)"),
}


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _table_columns(db: sqlite3.Connection, table: str) -> list[tuple[Any, ...]]:
    return list(db.execute(f"PRAGMA table_info({_quote(table)})"))


def _kernel_table(db: sqlite3.Connection) -> tuple[str, dict[str, tuple[str, str]]]:
    tables = [
        row[0]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    ]
    candidates = []
    for table in tables:
        if "KERNEL" not in table.upper():
            continue
        columns = {
            row[1].lower(): (row[1], row[2].upper())
            for row in _table_columns(db, table)
        }
        if "graphnodeid" in columns:
            candidates.append((table, columns))
    if len(candidates) != 1:
        raise RuntimeError(
            "trace does not expose one unambiguous graphNodeId kernel table"
        )
    return candidates[0]


def _string_table(db: sqlite3.Connection) -> tuple[str, str, str]:
    tables = [
        row[0]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    ]
    candidates = []
    for table in tables:
        columns = {row[1].lower(): row[1] for row in _table_columns(db, table)}
        if table.lower() == "stringids" and {"id", "value"} <= columns.keys():
            candidates.append((table, columns["id"], columns["value"]))
    if len(candidates) != 1:
        raise RuntimeError("trace does not expose one unambiguous StringIds table")
    return candidates[0]


def _kernel_rows(db: sqlite3.Connection) -> list[tuple[Any, Any, Any, Any]]:
    table, columns = _kernel_table(db)
    required = {"graphnodeid", "start", "end"}
    if not required <= columns.keys():
        raise RuntimeError("kernel table lacks graph/start/end attribution")
    name_key = next(
        (
            key
            for key in ("demangledname", "shortname", "name", "mangledname")
            if key in columns
        ),
        None,
    )
    if name_key is None:
        raise RuntimeError("kernel table lacks a name attribution column")
    graph, start, end = (columns[key][0] for key in ("graphnodeid", "start", "end"))
    name, declared = columns[name_key]
    selected = ",".join(f"k.{_quote(value)}" for value in (graph, start, end))
    if any(marker in declared for marker in ("TEXT", "CHAR", "CLOB")):
        sql = f"SELECT {selected}, k.{_quote(name)} FROM {_quote(table)} AS k"
    else:
        strings, string_id, string_value = _string_table(db)
        sql = (
            f"SELECT {selected}, s.{_quote(string_value)} FROM {_quote(table)} AS k "
            f"JOIN {_quote(strings)} AS s ON k.{_quote(name)}=s.{_quote(string_id)}"
        )
    rows = list(db.execute(sql))
    total = db.execute(f"SELECT count(*) FROM {_quote(table)}").fetchone()[0]
    if len(rows) != total:
        raise RuntimeError("kernel name resolution did not cover the full trace")
    return rows


def _classify(name: str) -> str | None:
    matches = [
        family for family, pattern in FAMILIES.items() if pattern.search(name.lower())
    ]
    if len(matches) > 1:
        raise RuntimeError(f"compound kernel-family attribution: {matches}")
    return matches[0] if matches else None


def _csv_census(
    trace_csv: Path,
) -> tuple[int, Decimal, dict[str, int], dict[str, Decimal]]:
    with trace_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        duration_key = "Duration (ns)"
        if "Name" not in fields or duration_key not in fields:
            raise RuntimeError("missing CUDA trace Name/Duration (ns) columns")
        rows = 0
        families = dict.fromkeys(FAMILIES, 0)
        total_duration = Decimal(0)
        family_durations = dict.fromkeys(FAMILIES, Decimal(0))
        for row in reader:
            rows += 1
            try:
                duration = Decimal(row[duration_key].replace(",", ""))
            except (InvalidOperation, AttributeError) as error:
                raise RuntimeError("invalid CUDA trace duration") from error
            if not duration.is_finite() or duration < 0:
                raise RuntimeError("invalid CUDA trace duration")
            total_duration += duration
            family = _classify(row["Name"])
            if family is not None:
                families[family] += 1
                family_durations[family] += duration
        return rows, total_duration, families, family_durations


def validate_trace(trace_csv: Path, sqlite_path: Path) -> dict[str, Any]:
    csv_rows, csv_duration, csv_families, csv_family_durations = _csv_census(trace_csv)
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as db:
        rows = _kernel_rows(db)
    cross_tab = {
        family: {
            location: {"rows": 0, "duration_ns": 0} for location in ("graph", "eager")
        }
        for family in FAMILIES
    }
    location_rows = {"graph": 0, "eager": 0}
    sqlite_duration = 0
    for graph_node, start, end, name in rows:
        if not isinstance(name, str):
            raise RuntimeError("unresolved non-text kernel name")
        if type(start) is not int or type(end) is not int:
            raise RuntimeError("CUDA activity timestamps must be integer nanoseconds")
        duration = end - start
        if duration < 0:
            raise RuntimeError("invalid CUDA activity duration")
        sqlite_duration += duration
        location = "graph" if graph_node not in (None, 0) else "eager"
        location_rows[location] += 1
        family = _classify(name)
        if family is not None:
            cross_tab[family][location]["rows"] += 1
            cross_tab[family][location]["duration_ns"] += duration
    totals = {
        family: sum(cell["rows"] for cell in locations.values())
        for family, locations in cross_tab.items()
    }
    if any(count < 400 for count in totals.values()):
        raise RuntimeError(f"insufficient kernel-family attribution: {totals}")
    if csv_families != totals:
        raise RuntimeError("CSV/SQLite kernel-family census mismatch")
    sqlite_family_durations = {
        family: sum(cell["duration_ns"] for cell in locations.values())
        for family, locations in cross_tab.items()
    }
    if csv_rows != len(rows) or csv_duration != sqlite_duration:
        raise RuntimeError("CSV/SQLite total row/duration mismatch")
    if any(
        csv_family_durations[family] != duration
        for family, duration in sqlite_family_durations.items()
    ):
        raise RuntimeError("CSV/SQLite kernel-family duration mismatch")
    if any(value < 400 for value in location_rows.values()):
        raise RuntimeError("insufficient graph/eager attribution")
    return {
        "cuda_rows": {"csv": csv_rows, "sqlite_kernels": len(rows)},
        "cuda_duration_ns": {"csv": int(csv_duration), "sqlite": sqlite_duration},
        "duration_unit": "ns",
        "location_rows": location_rows,
        "family_rows": totals,
        "family_by_location": cross_tab,
    }
