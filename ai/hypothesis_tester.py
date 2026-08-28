"""
hypothesis_tester.py

Takes generated hypotheses and an AnomalyEvidencePackage, then evaluates
each hypothesis against the underlying evidence to compute a verification
score, supporting metrics, and a final verdict (Validated, Refuted, or
Inconclusive).

Field names used downstream (app.py, reporter.py) MUST match exactly:
  - HypothesisTestResult.final_score   (NOT .score)
  - HypothesisTestResult.verdict is one of "Validated" / "Inconclusive" / "Refuted"
    (NOT "SUPPORTED")
  - test_hypotheses() returns an EvaluationReport, whose actual list of
    per-hypothesis results lives at `.test_results`.
"""

from __future__ import annotations

from dataclasses import dataclass
from .evidence_retriever import AnomalyEvidencePackage
from .hypothesis_generator import HypothesisGenerationResult, Hypothesis


@dataclass
class HypothesisTestResult:
    hypothesis_id: str
    title: str
    verdict: str  # "Validated", "Refuted", "Inconclusive"
    final_score: float  # 0.0 to 1.0
    supporting_evidence_count: int
    rationale: str


@dataclass
class EvaluationReport:
    anomaly_summary: str
    test_results: list[HypothesisTestResult]


def test_hypotheses(evidence_package: AnomalyEvidencePackage, gen_result: HypothesisGenerationResult) -> EvaluationReport:
    """
    Evaluates each generated hypothesis by checking the volume and strength
    of the supporting evidence records provided in the evidence package.
    """
    test_results = []

    for hyp in gen_result.hypotheses:
        cited_ids = set(hyp.supporting_evidence_ids)

        matched_records = 0
        for record in evidence_package.all_records:
            if record.evidence_id in cited_ids:
                matched_records += 1

        citation_ratio = (matched_records / len(cited_ids)) if cited_ids else 0.0
        final_score = round(min(1.0, (hyp.confidence_score * 0.6) + (citation_ratio * 0.4)), 2)

        if final_score >= 0.70:
            verdict = "Validated"
            rationale = (
                f"Strong alignment with underlying data. Found {matched_records} valid supporting "
                f"records out of {len(cited_ids)} cited references, confirming root-cause pattern."
            )
        elif final_score >= 0.45:
            verdict = "Inconclusive"
            rationale = (
                f"Partial alignment. Found {matched_records} supporting records, but additional "
                f"downstream cross-validation is recommended."
            )
        else:
            verdict = "Refuted"
            rationale = (
                f"Insufficient data support. Only {matched_records} records matched the citations, "
                f"suggesting this hypothesis does not explain the core anomaly."
            )

        test_results.append(
            HypothesisTestResult(
                hypothesis_id=hyp.hypothesis_id,
                title=hyp.title,
                verdict=verdict,
                final_score=final_score,
                supporting_evidence_count=matched_records,
                rationale=rationale
            )
        )

    return EvaluationReport(
        anomaly_summary=gen_result.anomaly_summary,
        test_results=test_results
    )


if __name__ == "__main__":
    from anomaly_detector import detect_all_anomalies
    from evidence_retriever import retrieve_evidence_for_anomaly
    from hypothesis_generator import generate_hypotheses_for_anomaly

    anomalies = detect_all_anomalies()
    if anomalies:
        top_anomaly = anomalies[0]
        print(f"Testing Hypothesis Tester on top anomaly:\n{top_anomaly}\n")

        package = retrieve_evidence_for_anomaly(top_anomaly)
        gen_result = generate_hypotheses_for_anomaly(package)
        evaluation = test_hypotheses(package, gen_result)

        print(f"Evaluation Report for: {evaluation.anomaly_summary}")
        for res in evaluation.test_results:
            print(f"\n[{res.hypothesis_id}] {res.title}")
            print(f"Verdict: {res.verdict} (Score: {res.final_score})")
            print(f"Supporting Records: {res.supporting_evidence_count}")
            print(f"Rationale: {res.rationale}")
    else:
        print("No anomalies detected.")