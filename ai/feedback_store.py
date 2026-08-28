"""
feedback_store.py

Lets an analyst mark each generated hypothesis as CONFIRMED (verified
against what actually happened) or REJECTED (investigated and ruled out)
after they've done real legwork - called the regional manager, checked
the carrier's incident log, whatever it takes.

This is the single most important missing piece for using this system in
real life: without it, "Validated (score 0.85)" is just a number nobody
has ever checked against reality. With it, you build an actual track
record you can use to answer "is our scoring formula any good?"

Storage is a flat JSON file (data/feedback.json) rather than a database -
deliberately simple, since the point here is correctness and durability
across app restarts, not scale. A team deploying this for real should
swap this for a proper table, but the interface (record_feedback /
get_feedback / get_calibration_stats) would stay the same.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEEDBACK_PATH = os.path.join(FEEDBACK_DIR, "feedback.json")

CONFIRMED = "Confirmed"
REJECTED = "Rejected"


@dataclass
class FeedbackEntry:
    key: str                # unique id for this (anomaly, hypothesis) pair
    metric: str
    dimension_value: str
    week: int
    hypothesis_id: str
    hypothesis_title: str
    system_verdict: str     # what test_hypotheses() said (Validated/Inconclusive/Refuted)
    system_score: float     # the final_score at the time feedback was given
    analyst_verdict: str    # CONFIRMED or REJECTED - what the human found to be true
    note: str
    timestamp: str


def make_feedback_key(anomaly, hypothesis_id: str) -> str:
    """Deterministic key so re-analyzing the same anomaly finds prior feedback."""
    return f"{anomaly.metric}|{anomaly.dimension_value}|{anomaly.week}|{hypothesis_id}"


def _load_all() -> dict:
    if not os.path.exists(FEEDBACK_PATH):
        return {}
    try:
        with open(FEEDBACK_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_all(data: dict) -> None:
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(data, f, indent=2)


def record_feedback(
    anomaly, hypothesis_id: str, hypothesis_title: str,
    system_verdict: str, system_score: float,
    analyst_verdict: str, note: str = "",
) -> None:
    """Saves (or overwrites) an analyst's confirm/reject decision."""
    key = make_feedback_key(anomaly, hypothesis_id)
    entry = FeedbackEntry(
        key=key,
        metric=anomaly.metric,
        dimension_value=anomaly.dimension_value,
        week=anomaly.week,
        hypothesis_id=hypothesis_id,
        hypothesis_title=hypothesis_title,
        system_verdict=system_verdict,
        system_score=system_score,
        analyst_verdict=analyst_verdict,
        note=note,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    data = _load_all()
    data[key] = asdict(entry)
    _save_all(data)


def get_feedback(anomaly, hypothesis_id: str) -> Optional[dict]:
    """Returns prior feedback for this exact (anomaly, hypothesis) pair, if any."""
    key = make_feedback_key(anomaly, hypothesis_id)
    return _load_all().get(key)


def get_all_feedback() -> list[dict]:
    return list(_load_all().values())


def get_calibration_stats() -> dict:
    """
    The whole point of collecting feedback: does the system's own verdict
    (Validated/Inconclusive/Refuted) actually line up with what analysts
    confirmed in reality? This is the closest thing to ground-truth
    validation this project has - everything upstream (thresholds,
    confidence formula) should eventually be tuned against this.

    Returns counts and a naive "agreement rate": how often a system
    verdict of Validated was analyst-CONFIRMED, and how often Refuted
    was analyst-REJECTED (both count as the system agreeing with reality).
    """
    entries = get_all_feedback()
    total = len(entries)
    if total == 0:
        return {"total": 0, "confirmed": 0, "rejected": 0, "agreement_rate": None}

    confirmed = sum(1 for e in entries if e["analyst_verdict"] == CONFIRMED)
    rejected = sum(1 for e in entries if e["analyst_verdict"] == REJECTED)

    agreements = sum(
        1 for e in entries
        if (e["system_verdict"] == "Validated" and e["analyst_verdict"] == CONFIRMED)
        or (e["system_verdict"] == "Refuted" and e["analyst_verdict"] == REJECTED)
    )
    agreement_rate = round(agreements / total, 2) if total else None

    return {
        "total": total,
        "confirmed": confirmed,
        "rejected": rejected,
        "agreement_rate": agreement_rate,
    }