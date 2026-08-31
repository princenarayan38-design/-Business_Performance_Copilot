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
from .feedback_store import get_confidence_adjustment
from . import model_trainer


def generate_generic_hypotheses(evidence_package: GenericEvidencePackage) -> HypothesisGenerationResult:
    anomaly = evidence_package.anomaly
    is_opportunity = getattr(anomaly, "nature", "RISK") == "OPPORTUNITY"
    total_evidence = evidence_package.total_evidence_count

    hypotheses = []
    for i, (source_label, items) in enumerate(evidence_package.records_by_source.items(), start=1):
        if not items:
            continue  # no evidence from this source for this week - nothing to hypothesize

        ids = [item.evidence_id for item in items[:3]]
        base_score, basis = _derive_confidence(ids, total_evidence)

        verb = "is time-correlated with the gain" if is_opportunity else "is time-correlated with the deviation"
        title = f"Factor Linked To: {source_label}"
        description = (
            f"{len(items)} record(s) from '{source_label}' fall within the anomaly window for "
            f"{anomaly.dimension_value}, week {anomaly.week}. This {verb} — "
            f"worth checking with the team before treating it as a confirmed cause."
        )

        ml_result = model_trainer.predict_confidence(
            evidence_count=len(ids),
            base_confidence_score=base_score,
            deviation_pct=getattr(anomaly, "deviation_pct", 0.0),
            modified_z_score=getattr(anomaly, "modified_z_score", 0.0),
            severity=getattr(anomaly, "severity", "MEDIUM"),
            nature=getattr(anomaly, "nature", "RISK"),
        )

        if ml_result.get("available"):
            ml_prob = ml_result["probability"]
            adjusted_score = round(min(0.95, max(0.05, (0.5 * base_score) + (0.5 * ml_prob))), 2)
            learning_note = (
                f"🤖 Trained model ({ml_result['trained_on_samples']} labeled examples, "
                f"{ml_result['train_accuracy']:.0%} train accuracy) predicts {ml_prob:.0%} "
                f"chance this hypothesis type would be confirmed."
            )
            hyp_kwargs = dict(
                confidence_score=adjusted_score,
                learning_note=learning_note,
                confidence_adjustment=round(adjusted_score - base_score, 3),
                learning_sample_count=ml_result["trained_on_samples"],
                ml_used=True,
                ml_probability=ml_prob,
            )
        else:
            learning = get_confidence_adjustment(metric=anomaly.metric, hypothesis_title=title)
            adjusted_score = round(min(0.95, max(0.05, base_score + learning["adjustment"])), 2)
            hyp_kwargs = dict(
                confidence_score=adjusted_score,
                learning_note=learning["note"],
                confidence_adjustment=learning["adjustment"],
                learning_sample_count=learning["sample_count"],
            )

        hypotheses.append(Hypothesis(
            hypothesis_id=f"H{i}",
            title=title,
            description=description,
            supporting_evidence_ids=ids,
            evidence_basis=basis,
            **hyp_kwargs,
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