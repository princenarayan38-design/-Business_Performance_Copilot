"""
hypothesis_generator.py

Drafts structured CONTRIBUTING-FACTOR CANDIDATES for a detected anomaly -
deliberately not called "root causes". Each candidate cites specific
evidence record IDs pulled by evidence_retriever.py, which are matched by
TIME WINDOW ONLY (the anomaly's week, +/- a buffer). That is temporal
correlation, not verified causation - an analyst still has to confirm or
reject each candidate against what actually happened (see the
confirm/reject feedback workflow wired up in app.py).

WHY CONFIDENCE IS DERIVED, NOT HARDCODED
------------------------------------------
Earlier versions of this file assigned fixed confidence_score constants
(0.85, 0.75, 0.70) to every hypothesis regardless of the actual evidence
found for a given anomaly. That's a real credibility problem: the number
never moved no matter how much or how little supporting evidence existed,
so it conveyed false precision - it looked like a calibrated probability
but was actually a decoration.

_derive_confidence() below computes confidence from two things that
actually vary per anomaly:
  1. How much of the total retrieved evidence supports this specific
     candidate (a candidate backed by 4 of 5 evidence records is more
     credible than one backed by 1 of 5).
  2. How many evidence records support it in absolute terms (a candidate
     with zero matching evidence should never score high, no matter what
     ratio math would otherwise produce).

This is still a heuristic, not a statistically validated model - real
calibration requires comparing these scores against confirmed outcomes
over time (see ai/feedback_store.py). But it at least reacts to the
actual data instead of being fixed at design time.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .evidence_retriever import AnomalyEvidencePackage


@dataclass
class Hypothesis:
    hypothesis_id: str
    title: str
    description: str
    supporting_evidence_ids: List[str]
    confidence_score: float = 0.5
    evidence_basis: str = ""  # human-readable note on what the confidence is based on


@dataclass
class HypothesisGenerationResult:
    anomaly_summary: str
    hypotheses: List[Hypothesis]


def _derive_confidence(supporting_ids: list[str], total_evidence_count: int) -> tuple[float, str]:
    """
    Confidence = f(how much of the available evidence backs this
    candidate). NOT a claim of causal probability - just a data-driven
    substitute for what used to be a hardcoded number.

    Returns (score, human-readable basis string) so the UI can show WHY
    a score is what it is, rather than presenting it as an opaque fact.
    """
    supporting_count = len(supporting_ids)

    if total_evidence_count == 0:
        return 0.30, "No evidence records were retrieved for this anomaly at all."

    if supporting_count == 0:
        return 0.30, "No evidence records directly support this candidate."

    ratio = supporting_count / total_evidence_count
    # Base 0.40 (weak default), scaled up by both the ratio of evidence
    # this candidate accounts for AND the absolute count (so 1 record out
    # of 1 doesn't score as high as 4 records out of 5).
    score = 0.40 + (ratio * 0.35) + (min(supporting_count, 4) * 0.06)
    score = round(min(0.92, score), 2)  # cap below 1.0 - never claim certainty

    basis = (
        f"{supporting_count} of {total_evidence_count} retrieved evidence records "
        f"({ratio:.0%}) are time-correlated with this candidate."
    )
    return score, basis


def generate_hypotheses_for_anomaly(evidence_package: AnomalyEvidencePackage) -> HypothesisGenerationResult:
    anomaly = evidence_package.anomaly
    is_opportunity = getattr(anomaly, "nature", "RISK") == "OPPORTUNITY"
    total_evidence = evidence_package.total_evidence_count

    delivery_ids = [item.evidence_id for item in evidence_package.delivery_records]
    feedback_ids = [item.evidence_id for item in evidence_package.feedback_records]
    event_ids = [item.evidence_id for item in evidence_package.event_records]

    def make(hyp_id, title, description, ids):
        score, basis = _derive_confidence(ids, total_evidence)
        return Hypothesis(
            hypothesis_id=hyp_id, title=title, description=description,
            supporting_evidence_ids=ids, confidence_score=score, evidence_basis=basis,
        )

    if is_opportunity:
        # Framing flips from "what went wrong" to "what went right, and can
        # we replicate it" - same evidence sources, different question.
        hyp_ids = delivery_ids[:2] if delivery_ids else event_ids[:1]
        hyp1 = make("H1", "Operational Excellence & Faster Fulfillment",
                     f"Stronger delivery/operational performance in the {anomaly.dimension_value} region "
                     f"is time-correlated with the gain during week {anomaly.week} — worth confirming "
                     f"with the region before assuming it's replicable.",
                     hyp_ids)
        hyp_ids2 = feedback_ids[:2] if feedback_ids else delivery_ids[:1]
        hyp2 = make("H2", "Positive Customer Sentiment / Word-of-Mouth",
                     f"Positive feedback during week {anomaly.week} is time-correlated with the uplift "
                     f"in the {anomaly.dimension_value} region.",
                     hyp_ids2)
        hyp_ids3 = event_ids[:2] if event_ids else feedback_ids[:1]
        hyp3 = make("H3", "Favorable External Event or Market Timing",
                     f"A promotion, local event, or market shift around week {anomaly.week} "
                     f"is time-correlated with the extra demand — check if it's repeatable.",
                     hyp_ids3)
        hypotheses = [hyp1, hyp2, hyp3]
        summary = (
            f"Generated {len(hypotheses)} growth-driver candidates (time-correlated, not yet confirmed) "
            f"for the {anomaly.metric} opportunity in {anomaly.dimension_value} (Week {anomaly.week})."
        )
    else:
        hyp_ids = delivery_ids[:2] if delivery_ids else event_ids[:1]
        hyp1 = make("H1", "Supply Chain Disruption & Late Deliveries",
                     f"Operational delays in the {anomaly.dimension_value} region are time-correlated "
                     f"with the deviation during week {anomaly.week}.",
                     hyp_ids)
        hyp_ids2 = feedback_ids[:2] if feedback_ids else delivery_ids[:1]
        hyp2 = make("H2", "Customer Sentiment Shift",
                     f"Negative feedback during week {anomaly.week} is time-correlated with the "
                     f"deviation in the {anomaly.dimension_value} region.",
                     hyp_ids2)
        hyp_ids3 = event_ids[:2] if event_ids else feedback_ids[:1]
        hyp3 = make("H3", "External Market or Regional Event Impact",
                     f"External events around week {anomaly.week} are time-correlated with a "
                     f"demand/supply shift.",
                     hyp_ids3)
        hypotheses = [hyp1, hyp2, hyp3]
        summary = (
            f"Generated {len(hypotheses)} contributing-factor candidates (time-correlated, not yet "
            f"confirmed) for the {anomaly.metric} anomaly in {anomaly.dimension_value} (Week {anomaly.week})."
        )

    return HypothesisGenerationResult(
        anomaly_summary=summary,
        hypotheses=hypotheses
    )