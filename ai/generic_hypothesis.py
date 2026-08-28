"""
generic_hypothesis.py

The Universal Mode counterpart to hypothesis_generator.py. That module
emits 3 fixed, domain-specific hypothesis templates (supply chain /
sentiment / external event) - meaningful for the demo's specific business
scenario, meaningless for an arbitrary uploaded dataset.

This module instead generates ONE candidate PER user-supplied evidence
source that actually has matching records for the anomaly, named after
whatever the user called that source. If a source has zero matching
evidence, no hypothesis is generated for it (no point speculating about
a source with nothing to show). If NO source has any evidence at all, a
single low-confidence "Unexplained Statistical Deviation" candidate is
returned instead of inventing a story.

Confidence scoring reuses the exact same evidence-ratio logic as the
demo pipeline (see hypothesis_generator._derive_confidence) - a
candidate backed by more of the total retrieved evidence scores higher,
capped below 1.0, same honesty caveats apply.
"""

from __future__ import annotations

from .hypothesis_generator import Hypothesis, HypothesisGenerationResult, _derive_confidence
from .generic_evidence import GenericEvidencePackage


def generate_generic_hypotheses(evidence_package: GenericEvidencePackage) -> HypothesisGenerationResult:
    anomaly = evidence_package.anomaly
    is_opportunity = getattr(anomaly, "nature", "RISK") == "OPPORTUNITY"
    total_evidence = evidence_package.total_evidence_count

    hypotheses = []
    for i, (source_label, items) in enumerate(evidence_package.records_by_source.items(), start=1):
        if not items:
            continue  # no evidence from this source for this week - nothing to hypothesize

        ids = [item.evidence_id for item in items[:3]]
        score, basis = _derive_confidence(ids, total_evidence)

        verb = "is time-correlated with the gain" if is_opportunity else "is time-correlated with the deviation"
        title = f"Factor Linked To: {source_label}"
        description = (
            f"{len(items)} record(s) from '{source_label}' fall within the anomaly window for "
            f"{anomaly.dimension_value}, week {anomaly.week}. This {verb} — "
            f"worth checking with the team before treating it as a confirmed cause."
        )

        hypotheses.append(Hypothesis(
            hypothesis_id=f"H{i}",
            title=title,
            description=description,
            supporting_evidence_ids=ids,
            confidence_score=score,
            evidence_basis=basis,
        ))

    if not hypotheses:
        hypotheses.append(Hypothesis(
            hypothesis_id="H1",
            title="Unexplained Statistical Deviation",
            description=(
                f"No uploaded evidence source had records falling within the anomaly window for "
                f"{anomaly.dimension_value}, week {anomaly.week}. The deviation is statistically "
                f"confirmed, but no contributing factor could be identified from the data provided."
            ),
            supporting_evidence_ids=[],
            confidence_score=0.30,
            evidence_basis="No evidence sources had matching records for this time window.",
        ))

    nature_word = "growth-driver" if is_opportunity else "contributing-factor"
    summary = (
        f"Generated {len(hypotheses)} {nature_word} candidate(s) (time-correlated, not yet confirmed) "
        f"for the {anomaly.metric} {'opportunity' if is_opportunity else 'anomaly'} in "
        f"{anomaly.dimension_value} (Week {anomaly.week})."
    )

    return HypothesisGenerationResult(anomaly_summary=summary, hypotheses=hypotheses)