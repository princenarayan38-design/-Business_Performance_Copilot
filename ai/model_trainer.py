"""
model_trainer.py

The actual TRAINED MODEL behind the "gets more accurate each session"
claim. Where feedback_store.get_confidence_adjustment() is a transparent
heuristic recalibration, this module is a real supervised classifier:

  Features (per hypothesis, snapshotted at feedback time):
    - evidence_count            how many evidence records backed it
    - base_confidence_score     the evidence-derived score before any
                                 learning adjustment
    - deviation_pct             how far the anomaly moved from baseline
    - modified_z_score          statistical severity of the anomaly
    - severity_is_high          1 if HIGH, else 0
    - nature_is_opportunity     1 if OPPORTUNITY, else 0

  Label:
    - 1 if the analyst CONFIRMED the hypothesis, 0 if REJECTED

  Model:
    - LogisticRegression (scikit-learn) - deliberately simple: the
      dataset is small (one row per analyst decision), so a linear model
      that's fast to retrain and easy to explain beats anything fancier.

WHY THIS COUNTS AS "SELF-LEARNING"
------------------------------------
train_model() is called again after every single record_feedback() call
(see app.py). Each analyst decision immediately becomes another labeled
training example, and the classifier is refit from scratch on the full
accumulated history. No manual retraining step, no separate offline
pipeline - the model that scores the NEXT hypothesis is never more than
one analyst decision stale.

WHY LOGISTIC REGRESSION AND NOT SOMETHING BIGGER
----------------------------------------------------
With a handful to a few hundred feedback rows (this is a single
analyst's session history, not a web-scale dataset), a high-capacity
model would overfit immediately and produce a confidence number that
LOOKS precise but means nothing - the exact anti-pattern this project's
_derive_confidence() docstring already warns against. Logistic
regression's coefficients are also directly inspectable (see
get_model_status()), which matters for a system whose whole premise is
explainable, auditable scoring.

GRACEFUL DEGRADATION
----------------------
If scikit-learn/joblib aren't installed, or there isn't enough labeled
history yet (MIN_TRAINING_SAMPLES, and at least one example of each
class), predict_confidence() returns None and callers fall back to the
heuristic learning loop in feedback_store.py - the same "optional
dependency, always-available fallback" pattern already used for Prophet
in forecasting.py.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from .feedback_store import get_all_feedback, CONFIRMED

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(MODEL_DIR, "confidence_model.joblib")

MIN_TRAINING_SAMPLES = 8   # need at least this many labeled examples...
MIN_PER_CLASS = 2          # ...with at least this many of EACH verdict, or
                            # the classifier has nothing to discriminate.

FEATURE_NAMES = [
    "evidence_count",
    "base_confidence_score",
    "deviation_pct_abs",
    "modified_z_abs",
    "severity_is_high",
    "nature_is_opportunity",
]


def _build_feature_row(
    evidence_count: int,
    base_confidence_score: float,
    deviation_pct: float,
    modified_z_score: float,
    severity: str,
    nature: str,
) -> list[float]:
    return [
        float(evidence_count),
        float(base_confidence_score),
        min(abs(float(deviation_pct)), 5.0),
        min(abs(float(modified_z_score)), 50.0),
        1.0 if severity == "HIGH" else 0.0,
        1.0 if nature == "OPPORTUNITY" else 0.0,
    ]


def _build_training_data():
    """Turns every stored feedback entry into an (X row, y label) pair."""
    X, y = [], []
    for e in get_all_feedback():
        # Older feedback entries (recorded before this module existed)
        # won't have the ML feature fields - skip them rather than
        # training on incomplete/zeroed-out rows.
        if "evidence_count" not in e or "base_confidence_score" not in e:
            continue
        X.append(_build_feature_row(
            evidence_count=e["evidence_count"],
            base_confidence_score=e["base_confidence_score"],
            deviation_pct=e["deviation_pct"],
            modified_z_score=e["modified_z_score"],
            severity=e["severity"],
            nature=e["nature"],
        ))
        y.append(1 if e["analyst_verdict"] == CONFIRMED else 0)
    return X, y


def train_model() -> dict:
    """
    Retrains the classifier from the FULL accumulated feedback history
    and saves it to disk. Call this after every record_feedback() - it's
    cheap (a few dozen to a few hundred rows, logistic regression) so
    retraining on every single new label is not wasteful.

    Returns a status dict describing what happened - used both for
    logging and to drive the sidebar's model-status panel.
    """
    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
    except ImportError:
        return {
            "trained": False,
            "reason": "scikit-learn / joblib not installed (pip install scikit-learn joblib).",
        }

    X, y = _build_training_data()
    n_total = len(y)
    n_confirmed = sum(y)
    n_rejected = n_total - n_confirmed

    if n_total < MIN_TRAINING_SAMPLES or n_confirmed < MIN_PER_CLASS or n_rejected < MIN_PER_CLASS:
        return {
            "trained": False,
            "reason": (
                f"Not enough labeled feedback yet ({n_total} total, {n_confirmed} confirmed / "
                f"{n_rejected} rejected). Need >= {MIN_TRAINING_SAMPLES} total and >= "
                f"{MIN_PER_CLASS} of each verdict before training a classifier."
            ),
            "sample_count": n_total,
        }

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)

    # Train-set accuracy, shown plainly labeled as such (NOT held-out
    # validation accuracy - with this little data, a real train/test
    # split would be too noisy to be meaningful; see model status note).
    train_accuracy = round(accuracy_score(y, model.predict(X)), 3)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": FEATURE_NAMES,
        "n_samples": n_total,
        "train_accuracy": train_accuracy,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }, MODEL_PATH)

    return {
        "trained": True,
        "sample_count": n_total,
        "confirmed": n_confirmed,
        "rejected": n_rejected,
        "train_accuracy": train_accuracy,
    }


def get_model_status() -> dict:
    """Loads the saved model's metadata (if any) for display - no scoring."""
    try:
        import joblib
    except ImportError:
        return {"trained": False, "reason": "scikit-learn / joblib not installed."}

    if not os.path.exists(MODEL_PATH):
        return {"trained": False, "reason": "No model has been trained yet."}

    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        return {"trained": False, "reason": f"Saved model could not be loaded ({e})."}

    return {
        "trained": True,
        "sample_count": bundle["n_samples"],
        "train_accuracy": bundle["train_accuracy"],
        "trained_at": bundle["trained_at"],
    }


def predict_confidence(
    evidence_count: int,
    base_confidence_score: float,
    deviation_pct: float,
    modified_z_score: float,
    severity: str,
    nature: str,
) -> dict:
    """
    Returns the trained classifier's predicted probability that a
    hypothesis with these characteristics would be CONFIRMED by an
    analyst, or a "not available" status if no model has been trained
    yet (or scikit-learn isn't installed) - callers should fall back to
    the heuristic loop in feedback_store.py in that case.
    """
    try:
        import joblib
    except ImportError:
        return {"available": False, "reason": "scikit-learn / joblib not installed."}

    if not os.path.exists(MODEL_PATH):
        return {"available": False, "reason": "Model not trained yet."}

    try:
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        row = [_build_feature_row(
            evidence_count, base_confidence_score, deviation_pct, modified_z_score, severity, nature,
        )]
        probability = float(model.predict_proba(row)[0][1])
    except Exception as e:
        return {"available": False, "reason": f"Prediction failed ({e})."}

    return {
        "available": True,
        "probability": round(probability, 3),
        "trained_on_samples": bundle["n_samples"],
        "train_accuracy": bundle["train_accuracy"],
        "trained_at": bundle["trained_at"],
    }