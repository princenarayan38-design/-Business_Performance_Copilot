"""
generic_evidence.py

The Universal Mode counterpart to evidence_retriever.py. That module is
hardcoded to the demo's delivery.csv / customer_feedback.csv / events.csv.
This module does the same JOB - pull raw rows that fall within an
anomaly's time window - but against ANY user-uploaded supporting
dataset(s), with columns the user maps themselves (same pattern as
universal_pipeline.py's main-metric mapping).

Design choice: evidence here is matched by TIME WINDOW ONLY (same
honesty caveat as the demo pipeline) - this is temporal correlation, not
proven causation. That framing is preserved end-to-end into
generic_hypothesis.py and the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from .evidence_retriever import EvidenceItem


@dataclass
class GenericEvidenceSource:
    """
    One user-uploaded supporting dataset, with the columns the user
    picked to identify date, (optional) category/dimension, and which
    columns to show in the evidence summary.
    """
    label: str                      # e.g. "Delivery / Operations", "Customer Feedback"
    id_prefix: str                  # short code for evidence IDs, e.g. "OPS", "FB"
    df: pd.DataFrame
    date_col: str
    dimension_col: str | None
    summary_cols: list[str] = field(default_factory=list)  # columns to show in the summary text


@dataclass
class GenericEvidencePackage:
    anomaly: object  # ai.anomaly_detector.Anomaly - typed loosely to avoid a circular import
    records_by_source: dict[str, list[EvidenceItem]]  # keyed by source label

    @property
    def total_evidence_count(self) -> int:
        return sum(len(v) for v in self.records_by_source.values())

    @property
    def all_records(self) -> list[EvidenceItem]:
        flat = []
        for records in self.records_by_source.values():
            flat.extend(records)
        return flat


def retrieve_generic_evidence(
    anomaly,
    sources: list[GenericEvidenceSource],
    min_date: pd.Timestamp,
    buffer_days: int = 3,
) -> GenericEvidencePackage:
    """
    For each user-supplied evidence source, pulls the rows whose date
    falls within the anomaly's week (+/- buffer_days), optionally
    filtered to the anomaly's dimension_value if that source has a
    dimension column mapped.
    """
    week_start = min_date + timedelta(weeks=anomaly.week - 1)
    window_start = week_start - timedelta(days=buffer_days)
    window_end = week_start + timedelta(days=7 + buffer_days)

    records_by_source: dict[str, list[EvidenceItem]] = {}

    for source in sources:
        df = source.df.copy()
        df["_parsed_date"] = pd.to_datetime(df[source.date_col], errors="coerce")
        mask = df["_parsed_date"].between(window_start, window_end)

        if source.dimension_col and source.dimension_col in df.columns:
            mask &= df[source.dimension_col].astype(str).str.lower() == str(anomaly.dimension_value).lower()

        filtered = df[mask]

        items = []
        for idx, row in filtered.iterrows():
            summary_parts = []
            cols_to_show = source.summary_cols if source.summary_cols else [
                c for c in df.columns if c not in (source.date_col, "_parsed_date")
            ]
            for col in cols_to_show[:5]:  # keep summaries readable
                summary_parts.append(f"{col}={row[col]}")
            summary = ", ".join(summary_parts) if summary_parts else "(no additional columns)"

            items.append(EvidenceItem(
                evidence_id=f"{source.id_prefix}-{idx}",
                source_file=source.label,
                evidence_type="structured",
                summary=f"[{row['_parsed_date'].date()}] {summary}",
                raw_data=row.drop(labels=["_parsed_date"], errors="ignore").to_dict(),
            ))

        records_by_source[source.label] = items

    return GenericEvidencePackage(anomaly=anomaly, records_by_source=records_by_source)