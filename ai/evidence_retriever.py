"""
evidence_retriever.py (Upgraded)

Given an Anomaly object, queries the raw underlying CSV files (delivery,
customer_feedback, events) to pull specific, raw rows matching the anomaly's
dimension and time window, featuring dynamic buffering and sentiment prioritization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import pandas as pd

from .anomaly_detector import Anomaly

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


@dataclass
class EvidenceItem:
    evidence_id: str
    source_file: str
    evidence_type: str  # "structured" or "unstructured"
    summary: str        # Short human-readable label for debugging/display
    raw_data: dict      # The literal underlying row data as a dictionary


@dataclass
class AnomalyEvidencePackage:
    anomaly: Anomaly
    delivery_records: list[EvidenceItem]
    feedback_records: list[EvidenceItem]
    event_records: list[EvidenceItem]

    @property
    def total_evidence_count(self) -> int:
        return len(self.delivery_records) + len(self.feedback_records) + len(self.event_records)

    @property
    def all_records(self) -> list[EvidenceItem]:
        """
        Generic flat view of every evidence item regardless of source -
        used by hypothesis_tester.py so it can work identically against
        this demo-specific package AND ai.generic_evidence's
        GenericEvidencePackage (which has no delivery/feedback/event
        distinction), without hypothesis_tester needing to know which
        kind of package it was given.
        """
        return self.delivery_records + self.feedback_records + self.event_records


def _load_raw_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path, parse_dates=["date"])
    
    # Re-derive week column consistently with data_processor.py
    min_date = df["date"].min()
    df["week"] = ((df["date"] - min_date).dt.days // 7) + 1
    return df


def retrieve_evidence_for_anomaly(anomaly: Anomaly) -> AnomalyEvidencePackage:
    """
    Pulls raw rows from delivery.csv, customer_feedback.csv, and events.csv
    matching the anomaly dimension and dynamic time window buffer.
    """
    target_week = anomaly.week
    dim_col = anomaly.dimension       # e.g., "region"
    dim_val = anomaly.dimension_value # e.g., "North"

    # ENHANCEMENT: Dynamic window buffering based on metric type
    # Delivery/supply chain anomalies often have lag effects starting 1 week prior
    window_buffer = 1 if "Delivery" in anomaly.metric or "Days" in anomaly.metric else 0

    # 1. Load raw datasets
    delivery_df = _load_raw_csv("delivery.csv")
    feedback_df = _load_raw_csv("customer_feedback.csv")
    events_df = _load_raw_csv("events.csv")

    min_w = target_week - window_buffer
    max_w = target_week + window_buffer
    
    # 2. Filter delivery records
    delivery_filtered = delivery_df[
        (delivery_df[dim_col] == dim_val) & 
        (delivery_df["week"].between(min_w, max_w))
    ]
    
    delivery_items = []
    for idx, row in delivery_filtered.iterrows():
        delivery_items.append(
            EvidenceItem(
                evidence_id=f"DEL-{idx}",
                source_file="delivery.csv",
                evidence_type="structured",
                summary=f"Week {row['week']} delivery stats: {row['average_delivery_days']} avg days, {row['late_delivery_rate']*100:.1f}% late rate",
                raw_data=row.to_dict()
            )
        )

    # 3. Filter feedback records
    feedback_filtered = feedback_df[
        (feedback_df[dim_col] == dim_val) & 
        (feedback_df["week"].between(min_w, max_w))
    ]
    
    # ENHANCEMENT: If investigating a negative performance spike, prioritize negative feedback
    if anomaly.actual > anomaly.expected and "negative" in feedback_filtered["sentiment"].values:
        feedback_filtered = feedback_filtered[feedback_filtered["sentiment"] == "negative"]

    feedback_items = []
    for idx, row in feedback_filtered.iterrows():
        feedback_items.append(
            EvidenceItem(
                evidence_id=f"FB-{idx}",
                source_file="customer_feedback.csv",
                evidence_type="unstructured",
                summary=f"[{row['sentiment'].upper()}] Category: {row['category']} - '{row['feedback']}'",
                raw_data=row.to_dict()
            )
        )

    # 4. Filter relevant events matching the window
    min_date = delivery_df["date"].min()
    events_df["week"] = ((events_df["date"] - min_date).dt.days // 7) + 1
    
    events_filtered = events_df[events_df["week"].between(min_w, max_w)]
    
    event_items = []
    for idx, row in events_filtered.iterrows():
        event_items.append(
            EvidenceItem(
                evidence_id=f"EVT-{idx}",
                source_file="events.csv",
                evidence_type="structured",
                summary=f"Event ({row['event_type']}): {row['description']}",
                raw_data=row.to_dict()
            )
        )

    return AnomalyEvidencePackage(
        anomaly=anomaly,
        delivery_records=delivery_items,
        feedback_records=feedback_items,
        event_records=event_items
    )


if __name__ == "__main__":
    from anomaly_detector import detect_all_anomalies

    anomalies = detect_all_anomalies()
    if anomalies:
        top_anomaly = anomalies[0]
        print(f"Top Anomaly Found:\n{top_anomaly}\n")
        
        package = retrieve_evidence_for_anomaly(top_anomaly)
        print(f"Retrieved {package.total_evidence_count} evidence rows with dynamic buffering:")
        print(f"  - Delivery records: {len(package.delivery_records)}")
        print(f"  - Feedback records: {len(package.feedback_records)}")
        print(f"  - Event records: {len(package.event_records)}")
    else:
        print("No anomalies detected.")