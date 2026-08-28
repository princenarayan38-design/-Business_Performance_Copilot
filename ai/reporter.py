"""
ai/reporter.py
Synthesizes the final executive report and recommendations based on
hypothesis testing results.
"""
import os
import json
from typing import Any, List
from pydantic import BaseModel


class ExecutiveReport(BaseModel):
    executive_summary: str
    key_recommendations: List[str]


def generate_executive_report(evidence_package: Any, gen_result: Any, evaluations: Any) -> ExecutiveReport:
    """
    Synthesizes final insights using Gemini or falls back offline.

    `evaluations` is the EvaluationReport returned by test_hypotheses() —
    its actual per-hypothesis results live at `evaluations.test_results`.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    # Accept either an EvaluationReport (has .test_results) or a bare list,
    # so this function stays robust if a caller passes either shape.
    results = getattr(evaluations, "test_results", evaluations)

    if not api_key:
        return _fallback_report(results)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        eval_summary = "\n".join([
            f"- Hypothesis {r.hypothesis_id} ({r.title}): Verdict={r.verdict}, "
            f"Score={r.final_score}, Rationale={r.rationale}"
            for r in results
        ])

        prompt = f"""
        You are an expert Chief Operations Officer. Write a short executive briefing for this supply chain anomaly:
        - Anomaly: {gen_result.anomaly_summary}
        - Test Evaluations:
        {eval_summary}

        Return valid raw JSON matching this structure:
        {{
            "executive_summary": "1-2 sentence high-level synthesis",
            "key_recommendations": ["Recommendation 1", "Recommendation 2"]
        }}
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

        data = json.loads(response.text)
        return ExecutiveReport(
            executive_summary=data.get("executive_summary", "Summary generated."),
            key_recommendations=data.get("key_recommendations", [])
        )
    except Exception as e:
        print(f"Reporter API error: {e}")
        return _fallback_report(results, error=str(e))


def _fallback_report(results, error: str | None = None) -> ExecutiveReport:
    """
    Deterministic, offline fallback that still reflects the actual
    hypothesis verdicts rather than a generic canned message.
    """
    validated = [r for r in results if r.verdict == "Validated"]
    inconclusive = [r for r in results if r.verdict == "Inconclusive"]

    if validated:
        top = max(validated, key=lambda r: r.final_score)
        summary = (
            f"Root-cause analysis points to '{top.title}' as the most strongly "
            f"supported explanation (score {top.final_score})."
        )
        recs = [
            f"Prioritize investigation into: {top.title}.",
            "Cross-check supporting evidence IDs cited in the validated hypothesis before acting.",
        ]
        if len(validated) > 1:
            recs.append("Other validated hypotheses may be contributing factors — review in parallel.")
    elif inconclusive:
        summary = (
            "No hypothesis was strongly validated; the leading candidates remain inconclusive "
            "given the available evidence."
        )
        recs = [
            "Gather additional evidence (extend the time window or add data sources) before acting.",
            "Treat inconclusive hypotheses as leads, not conclusions.",
        ]
    else:
        summary = "Available hypotheses were refuted by the evidence; the true root cause remains unidentified."
        recs = [
            "Re-examine the anomaly window for data quality issues.",
            "Consider generating alternative hypotheses outside the current evidence set.",
        ]

    if error:
        recs.append(f"(Note: AI-generated synthesis unavailable — {error})")

    return ExecutiveReport(executive_summary=summary, key_recommendations=recs)