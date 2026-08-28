"""
event_context.py

A structured way to tell the system about real-world context that pure
statistics can't see: holidays, festivals, supply-chain disruptions,
weather events, promotions, etc. When an anomaly is investigated, any
event whose date falls near that anomaly's week gets surfaced to the
local LLM as extra context for its hypothesis discussion.

This is deliberately separate from evidence_retriever.py's events.csv
handling (which is specific to the built-in demo dataset). This module
works for BOTH: the demo dataset's events.csv AND arbitrary user-supplied
event annotations for uploaded datasets (Universal Mode), through one
common interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd


@dataclass
class RealWorldEvent:
    date: pd.Timestamp
    category: str          # e.g. "Holiday", "Supply Chain", "Weather", "Promotion"
    description: str
    region: str | None = None   # None = applies to all regions/categories


def load_events_from_dataframe(df: pd.DataFrame) -> list[RealWorldEvent]:
    """
    Builds a list of RealWorldEvent from any dataframe that has at least
    a date-like column and a description-like column. Column names are
    matched loosely so this works both with the demo's events.csv
    (date, event_type, description) and a simple user-uploaded CSV
    (date, description) or (date, category, description, region).
    """
    if df is None or df.empty:
        return []

    cols_lower = {c.lower(): c for c in df.columns}

    date_col = next((cols_lower[c] for c in ("date", "event_date", "time") if c in cols_lower), None)
    desc_col = next((cols_lower[c] for c in ("description", "event", "details", "note") if c in cols_lower), None)
    category_col = next((cols_lower[c] for c in ("category", "event_type", "type") if c in cols_lower), None)
    region_col = next((cols_lower[c] for c in ("region", "dimension", "location") if c in cols_lower), None)

    if date_col is None or desc_col is None:
        return []

    events = []
    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    for idx, row in df.iterrows():
        date = parsed_dates.loc[idx]
        if pd.isna(date):
            continue
        events.append(RealWorldEvent(
            date=date,
            category=str(row[category_col]) if category_col else "Event",
            description=str(row[desc_col]),
            region=str(row[region_col]) if region_col and pd.notna(row.get(region_col)) else None,
        ))
    return events


def load_events_from_csv(file_or_path) -> list[RealWorldEvent]:
    """Convenience wrapper: reads a CSV (path or Streamlit UploadedFile) into events."""
    df = pd.read_csv(file_or_path)
    return load_events_from_dataframe(df)


def get_events_for_week(
    events: list[RealWorldEvent],
    week_number: int,
    min_date: pd.Timestamp,
    region: str | None = None,
    buffer_days: int = 3,
) -> list[RealWorldEvent]:
    """
    Returns events whose date falls within the given week (+/- buffer_days,
    since real-world effects often lead or lag the week boundary a little -
    e.g. a holiday on a Sunday can affect the following week's numbers).

    If `region` is given, events tied to a different specific region are
    excluded; events with region=None (general) always match.
    """
    if not events:
        return []

    week_start = min_date + timedelta(weeks=week_number - 1)
    window_start = week_start - timedelta(days=buffer_days)
    window_end = week_start + timedelta(days=7 + buffer_days)

    matches = []
    for e in events:
        if not (window_start <= e.date <= window_end):
            continue
        if region and e.region and e.region.lower() != region.lower():
            continue
        matches.append(e)
    return matches


def format_events_for_prompt(events: list[RealWorldEvent]) -> str:
    """Renders matched events as a short text block for injection into an LLM prompt."""
    if not events:
        return "No known real-world events (holidays, disruptions, promotions) are on record for this week."
    lines = ["Real-world context for this week:"]
    for e in events:
        region_note = f" [{e.region}]" if e.region else ""
        lines.append(f"  - {e.date.date()} ({e.category}){region_note}: {e.description}")
    return "\n".join(lines)