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
    # --- ML feature snapshot, captured at feedback time ---------------
    # Stored so model_trainer.py can retrain a real classifier from this
    # file directly, without needing to re-derive anomaly/hypothesis state
    # that may no longer exist (e.g. after the source data changes).
    evidence_count: int = 0
    base_confidence_score: float = 0.5   # evidence-derived score, before any learning adjustment
    deviation_pct: float = 0.0
    modified_z_score: float = 0.0
    severity: str = "MEDIUM"
    nature: str = "RISK"


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


def _finite_or_capped(value: float, cap: float = 50.0) -> float:
    """
    Clamps +/-inf (which can legitimately occur - see anomaly_detector.py's
    zero-MAD / zero-baseline edge cases) to a large finite number, so this
    value can be safely stored in JSON and fed into a numeric ML model
    without producing NaN/Inf downstream.
    """
    if value == float("inf"):
        return cap
    if value == float("-inf"):
        return -cap
    return float(value)


def record_feedback(
    anomaly, hypothesis, system_verdict: str, system_score: float,
    analyst_verdict: str, note: str = "",
) -> None:
    """
    Saves (or overwrites) an analyst's confirm/reject decision.

    `hypothesis` is the full Hypothesis object (not just its id/title) so
    this function can snapshot the feature values model_trainer.py needs
    to train a real classifier later: how much evidence backed this
    candidate, what the anomaly looked like statistically, etc. Capturing
    these NOW, at feedback time, means training data stays valid even if
    the underlying anomaly/evidence data later changes or is regenerated.
    """
    key = make_feedback_key(anomaly, hypothesis.hypothesis_id)
    entry = FeedbackEntry(
        key=key,
        metric=anomaly.metric,
        dimension_value=anomaly.dimension_value,
        week=anomaly.week,
        hypothesis_id=hypothesis.hypothesis_id,
        hypothesis_title=hypothesis.title,
        system_verdict=system_verdict,
        system_score=system_score,
        analyst_verdict=analyst_verdict,
        note=note,
        timestamp=datetime.now(timezone.utc).isoformat(),
        evidence_count=len(hypothesis.supporting_evidence_ids),
        base_confidence_score=float(hypothesis.confidence_score),
        deviation_pct=_finite_or_capped(anomaly.deviation_pct, cap=5.0),
        modified_z_score=_finite_or_capped(anomaly.modified_z_score, cap=50.0),
        severity=anomaly.severity,
        nature=getattr(anomaly, "nature", "RISK"),
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


MIN_LEARNING_SAMPLES = 3          # need at least this many past decisions before adjusting
MAX_CONFIDENCE_ADJUSTMENT = 0.12  # cap so learning nudges scores, never overrides evidence


def get_confidence_adjustment(metric: str, hypothesis_title: str) -> dict:
    """
    THE SELF-LEARNING LOOP.

    Looks at every past analyst CONFIRM/REJECT decision recorded for this
    exact (metric, hypothesis_title) pair - e.g. "Supply Chain Disruption &
    Late Deliveries" hypotheses raised for "Avg Delivery Days" anomalies,
    across every past session/week, not just the current run - and turns
    the historical confirm rate into a small adjustment applied to that
    hypothesis type's confidence score going forward.

    This is deliberately NOT a trained model. It is a transparent,
    auditable recalibration: if analysts have confirmed this hypothesis
    type 8 times out of 10 in the past, the system should be a bit more
    confident next time it proposes it; if it's been rejected repeatedly,
    it should be less confident. feedback.json (already written by
    record_feedback()) is the system's entire "memory" of past sessions -
    no separate database needed, since the existing feedback store already
    persists across restarts.

    Scoped to (metric, hypothesis_title) rather than title alone, because
    "Supply Chain Disruption" explaining a Revenue anomaly and the same
    title explaining a Delivery Days anomaly are different real-world
    patterns and shouldn't share one learned adjustment.

    Below MIN_LEARNING_SAMPLES, returns a neutral (0.0) adjustment rather
    than overreacting to 1-2 data points - same "don't manufacture
    statistically-shaky numbers" principle used elsewhere in this project.

    Returns:
        {
            "adjustment": float,       # in [-MAX_CONFIDENCE_ADJUSTMENT, +MAX_CONFIDENCE_ADJUSTMENT]
            "sample_count": int,       # how many past decisions this is based on
            "confirm_rate": float|None,
            "note": str,               # human-readable, meant for a UI badge
        }
    """
    entries = [
        e for e in get_all_feedback()
        if e.get("metric") == metric and e.get("hypothesis_title") == hypothesis_title
    ]
    sample_count = len(entries)

    if sample_count < MIN_LEARNING_SAMPLES:
        return {
            "adjustment": 0.0,
            "sample_count": sample_count,
            "confirm_rate": None,
            "note": (
                f"🧠 Learning: only {sample_count}/{MIN_LEARNING_SAMPLES} past decisions on record "
                f"for this hypothesis type — not enough history yet to adjust confidence."
            ),
        }

    confirmed = sum(1 for e in entries if e["analyst_verdict"] == CONFIRMED)
    confirm_rate = confirmed / sample_count

    # Centered at 0.5 confirm rate = no adjustment; scales linearly to the
    # cap. E.g. 100% confirm rate -> +0.12, 0% confirm rate -> -0.12.
    adjustment = round((confirm_rate - 0.5) * 2 * MAX_CONFIDENCE_ADJUSTMENT, 3)

    if adjustment > 0:
        direction = f"boosted by {adjustment:+.2f}"
    elif adjustment < 0:
        direction = f"reduced by {adjustment:.2f}"
    else:
        direction = "left unchanged"

    note = (
        f"🧠 Learned from {sample_count} past analyst decision(s) on this exact hypothesis type "
        f"for {metric}: confirmed {confirmed}/{sample_count} times ({confirm_rate:.0%}) — "
        f"confidence {direction}."
    )

    return {
        "adjustment": adjustment,
        "sample_count": sample_count,
        "confirm_rate": round(confirm_rate, 2),
        "note": note,
    }


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