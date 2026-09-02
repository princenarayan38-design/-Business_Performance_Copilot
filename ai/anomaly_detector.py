"""
anomaly_detector.py

Flags weeks where a metric deviates enough from its recent baseline to
count as a meaningful signal, not noise.

WHY MODIFIED Z-SCORE (MEDIAN + MAD), NOT MEAN + STD
------------------------------------------------------
data_processor.py's rolling_mean/rolling_std (Phase 2) use a plain 4-week
trailing window. That's fine in the steady state, but during a multi-week
anomaly, later weeks end up with EARLIER anomalous weeks inside their own
trailing window. That contaminates rolling_std - it grows sharply - and a
bigger std in the denominator of a z-score SHRINKS the score right when
the anomaly is at its worst.

The fix: the MEDIAN and MEDIAN ABSOLUTE DEVIATION (MAD) of the trailing
clean history, instead of the mean and standard deviation - a standard,
well-established technique (Iglewicz & Hoaglin, 1993), not a novel
invention.

    modified_z = 0.6745 * (actual - baseline_median) / baseline_MAD

WHY THE BASELINE IS BUILT CAUSALLY, EXCLUDING FLAGGED WEEKS
--------------------------------------------------------------
Median and MAD are robust up to a 50% "breakdown point" - they resist
being dragged by outliers only as long as outliers are a MINORITY of the
window. In this project's demo data, once a 4-week disruption is 2 weeks
in, a plain 4-observation trailing window is already 50% anomalous, which
breaks that guarantee.

In practice this showed up as: North week 17 was 46% above its true
baseline, but scored a modified z of only ~1.0 (nowhere near the 3.5
threshold) - because weeks 15-16 (also anomalous) were sitting inside
its own baseline window, dragging the median and MAD up with it.

The fix used here: process weeks in order, and maintain a baseline made
ONLY of historical observations that were NOT themselves flagged
anomalous. When a week gets flagged, its value is excluded from every
future week's baseline - so a persistent anomaly can't drag its own
comparison point along with it.

This is causal (a week's score never depends on future weeks, which
matters for a system meant to score live/incoming data) and adaptive:
once the metric returns to normal, subsequent normal weeks are added
back into the clean history, allowing the baseline to adapt to the
new normal over time.

THRESHOLD
----------
|modified_z| >= 3.5 is the threshold used here as the anomaly cutoff.
A week must ALSO deviate by at least MIN_MEANINGFUL_DEVIATION in real
terms - see detect_anomalies() for why a purely statistical threshold
isn't enough on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_processor import build_processed_dataset, ROLLING_WINDOW

MAD_SCALE_CONSTANT = 0.6745
ANOMALY_THRESHOLD = 3.5   # |modified_z| above this = anomaly (now a REFERENCE anchor — see dynamic threshold engine below)
HIGH_SEVERITY_THRESHOLD = 6.0  # |modified_z| above this = HIGH severity
MIN_MEANINGFUL_DEVIATION = 0.05  # a move must also be >=5% to count as an anomaly,
                                  # not just statistically unusual - see detect_anomalies()


# ===========================================================================
# DYNAMIC THRESHOLD ENGINE
# ===========================================================================
# WHY A FIXED 3.5 THRESHOLD FAILS ACROSS DIFFERENT STREAMS
# ------------------------------------------------------------------------
# A single hardcoded |modified_z| >= 3.5 cutoff implicitly assumes every
# metric/region combination has the same "shape" of noise. In practice:
#   - A high-revenue, high-volatility stream (e.g. a big region with daily
#     promo activity) naturally swings a lot week to week. A fixed 3.5
#     bar gets crossed by routine noise -> false positives, and analysts
#     start ignoring the alerts ("alert fatigue").
#   - A small, stable, low-volume stream barely moves at all under normal
#     conditions. A real, business-relevant problem there might only
#     produce a modified z of 2.5-3.0 - genuinely meaningful, but the
#     fixed 3.5 bar lets it slip through as "not an anomaly."
#
# THE FIX: treat 3.5 as a REFERENCE ANCHOR, not an absolute cutoff, and
# scale it per (metric, dimension_value) baseline window using three
# independent, multiplicative factors:
#
#   dynamic_threshold = BASE_THRESHOLD * cv_factor * tier_factor * user_multiplier
#
#   1. cv_factor      - stretches the bar for naturally noisy streams and
#                        relaxes it for naturally stable ones, using the
#                        Coefficient of Variation (CV = stddev / mean) of
#                        the stream's own recent baseline. CV is unitless,
#                        so it fairly compares a $2M revenue stream against
#                        a $20K one on the same noise-relative-to-scale
#                        footing - this is what makes the whole approach
#                        "universal" rather than tied to one metric's units.
#   2. tier_factor    - a coarser, business-facing adjustment: which
#                        revenue/volume tier this stream's baseline falls
#                        into (Low / Medium / High). This lets a business
#                        stakeholder reason about sensitivity in terms
#                        they recognize ("our small regions should be
#                        watched more closely") independently of the raw
#                        CV math.
#   3. user_multiplier - a direct, UI-controllable knob (0.7x-1.5x) so an
#                        analyst can bias the WHOLE system toward
#                        aggressive (catch more, tolerate more false
#                        positives) or conservative (fewer alerts, higher
#                        bar) without touching code.
#
# The result is clamped to [MIN_DYNAMIC_THRESHOLD, MAX_DYNAMIC_THRESHOLD]
# so no combination of factors can produce a threshold so low it flags
# routine noise, or so high it can never fire at all.
# ===========================================================================

# --- Tunable constants for the dynamic threshold engine --------------------

# Reference CV: the "typical" volatility this whole system was originally
# tuned around (roughly what the demo dataset's stable streams look like).
# Streams noisier than this get a higher bar; calmer streams get a lower one.
REFERENCE_CV = 0.20

# How strongly CV above/below the reference should move the threshold.
# 1.0 means "a CV that's 0.10 above reference adds 0.10 * CV_SENSITIVITY
# to the multiplicative factor" - tune this down for a gentler response.
CV_SENSITIVITY = 1.2

# Guardrails on the CV factor itself, applied before combining with the
# other factors - prevents one extremely volatile or extremely flat
# stream from producing a nonsensical multiplier on its own.
MIN_CV_FACTOR = 0.75
MAX_CV_FACTOR = 2.5

# Revenue/volume tiering: classifies a stream by the mean absolute value
# of its own recent baseline window. Cutoffs are in the same units as
# whatever metric is being scored (e.g. weekly revenue, weekly units) -
# override these per-deployment via `tier_cutoffs` if a business's real
# revenue scale differs a lot from this project's demo data.
DEFAULT_TIER_CUTOFFS = {
    "LOW": 0,          # baseline mean below MEDIUM cutoff -> LOW tier
    "MEDIUM": 50_000,  # baseline mean below HIGH cutoff, at/above this -> MEDIUM tier
    "HIGH": 250_000,   # baseline mean at/above this -> HIGH tier
}

# Business rationale per tier (see module docstring above):
#   LOW    - small/low-volume streams get MORE sensitive (lower factor):
#            these are easy to overlook otherwise, and small-dollar swings
#            are still meaningful relative to the business unit's size.
#   MEDIUM - unchanged (factor 1.0) - this is the "typical" case the
#            original fixed 3.5 threshold was implicitly tuned for.
#   HIGH   - large/high-revenue streams get MORE conservative (higher
#            factor): naturally larger absolute swings are common there
#            and don't automatically mean something is wrong.
DEFAULT_TIER_FACTORS = {
    "LOW": 0.85,
    "MEDIUM": 1.0,
    "HIGH": 1.20,
}

# Absolute floor/ceiling on the final dynamic threshold, regardless of how
# extreme the CV/tier/user-multiplier combination gets. Keeps the system
# from ever becoming unusable in either direction.
MIN_DYNAMIC_THRESHOLD = 2.5
MAX_DYNAMIC_THRESHOLD = 8.0

# User sensitivity multiplier bounds - exposed directly as a UI slider.
MIN_USER_MULTIPLIER = 0.7   # aggressive / most sensitive
MAX_USER_MULTIPLIER = 1.5   # conservative / least sensitive


def compute_coefficient_of_variation(baseline_values: list[float]) -> float:
    """
    CV = stddev / |mean| of a stream's recent baseline window. Unitless,
    so a stream with mean=2,000,000 and std=400,000 (CV=0.20) is treated
    identically to one with mean=20,000 and std=4,000 (also CV=0.20) -
    this is what lets one formula generalize across wildly different
    revenue scales instead of needing per-metric hand-tuning.

    Returns 0.0 for a degenerate/empty baseline or a zero mean (a flat
    zero baseline has no meaningful "relative" volatility to measure).
    """
    if not baseline_values:
        return 0.0

    values = np.array(baseline_values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))

    if mean == 0:
        return 0.0

    return abs(std / mean)


def classify_revenue_tier(baseline_mean: float, tier_cutoffs: dict | None = None) -> str:
    """
    Classifies a stream's scale as LOW / MEDIUM / HIGH based on the mean
    of its recent baseline window, using `tier_cutoffs` (defaults to
    DEFAULT_TIER_CUTOFFS - override per deployment if a business's real
    revenue/volume scale differs significantly from this project's demo
    data).
    """
    cutoffs = tier_cutoffs or DEFAULT_TIER_CUTOFFS
    abs_mean = abs(baseline_mean)

    if abs_mean >= cutoffs["HIGH"]:
        return "HIGH"
    elif abs_mean >= cutoffs["MEDIUM"]:
        return "MEDIUM"
    else:
        return "LOW"


def compute_dynamic_threshold(
    baseline_values: list[float],
    base_threshold: float = ANOMALY_THRESHOLD,
    user_multiplier: float = 1.0,
    tier_cutoffs: dict | None = None,
    tier_factors: dict | None = None,
) -> dict:
    """
    THE CORE OF THE DYNAMIC THRESHOLD ENGINE.

    Computes a per-stream anomaly threshold from the stream's OWN recent
    baseline window, replacing the single fixed ANOMALY_THRESHOLD used
    everywhere in the original detector.

    Args:
        baseline_values:  the same trailing clean-history window
                           detect_anomalies() already builds for its
                           median/MAD baseline - reused here so the
                           dynamic threshold and the anomaly score are
                           always computed from identical, causal data.
        base_threshold:    the reference anchor (default: the project's
                           original fixed value, 3.5).
        user_multiplier:   UI-controlled sensitivity knob. <1.0 = more
                           sensitive (lower threshold, catches more);
                           >1.0 = more conservative (higher threshold,
                           fewer alerts). Clamped to
                           [MIN_USER_MULTIPLIER, MAX_USER_MULTIPLIER].
        tier_cutoffs / tier_factors: override the default revenue/volume
                           tiering if a deployment's scale differs from
                           this project's demo data.

    Returns a dict (not just a number) so the UI can show WHY a given
    threshold was chosen - same "explainable scoring, not a black box"
    principle used throughout this project's confidence scoring:
        {
            "threshold": float,       # the final, clamped threshold to use
            "cv": float,              # coefficient of variation used
            "cv_factor": float,
            "tier": str,              # "LOW" / "MEDIUM" / "HIGH"
            "tier_factor": float,
            "user_multiplier": float, # the clamped multiplier actually applied
        }
    """
    user_multiplier = float(min(max(user_multiplier, MIN_USER_MULTIPLIER), MAX_USER_MULTIPLIER))

    if not baseline_values:
        # No history yet - fall back to the base threshold, unscaled, so
        # callers with genuinely no data still get a sane number instead
        # of a divide-by-zero or a nonsensical clamp.
        return {
            "threshold": base_threshold,
            "cv": 0.0,
            "cv_factor": 1.0,
            "tier": "MEDIUM",
            "tier_factor": 1.0,
            "user_multiplier": user_multiplier,
        }

    cv = compute_coefficient_of_variation(baseline_values)
    cv_factor = 1.0 + CV_SENSITIVITY * (cv - REFERENCE_CV)
    cv_factor = min(max(cv_factor, MIN_CV_FACTOR), MAX_CV_FACTOR)

    baseline_mean = float(np.mean(baseline_values))
    tier = classify_revenue_tier(baseline_mean, tier_cutoffs)
    tier_factor = (tier_factors or DEFAULT_TIER_FACTORS)[tier]

    raw_threshold = base_threshold * cv_factor * tier_factor * user_multiplier
    threshold = min(max(raw_threshold, MIN_DYNAMIC_THRESHOLD), MAX_DYNAMIC_THRESHOLD)

    return {
        "threshold": round(threshold, 2),
        "cv": round(cv, 3),
        "cv_factor": round(cv_factor, 3),
        "tier": tier,
        "tier_factor": tier_factor,
        "user_multiplier": user_multiplier,
    }


# ===========================================================================
# END DYNAMIC THRESHOLD ENGINE
# ===========================================================================


# ---------------------------------------------------------------------------
# Metric directionality — a statistical anomaly is not automatically a
# business risk. Whether a deviation is good or bad news depends on the
# metric: higher revenue is a growth opportunity, but higher delivery
# delay is a risk. Metrics not listed default to "higher_is_better" since
# most business KPIs (revenue, units, satisfaction) work that way; add a
# metric here explicitly if higher is worse for it (costs, delays, churn,
# complaint rates, defect rates, etc.).
# ---------------------------------------------------------------------------
LOWER_IS_BETTER_METRICS = {
    "Avg Delivery Days",
    "Delivery Days",
    "Late Delivery Rate",
    "Negative Feedback Rate",
    "Churn Rate",
    "Defect Rate",
    "Return Rate",
}

OPPORTUNITY = "OPPORTUNITY"
RISK = "RISK"


def classify_nature(metric_label: str, deviation_pct: float, lower_is_better: bool | None = None) -> str:
    """
    Classifies an anomaly's business nature, independent of its
    statistical severity. A HIGH-severity anomaly can still be great news
    (e.g. a huge, unexpected revenue jump) - severity says "how unusual",
    nature says "is this good or bad for the business."

    higher_is_better metric + positive deviation  -> OPPORTUNITY (growth)
    higher_is_better metric + negative deviation   -> RISK (a drop/loss)
    lower_is_better metric  + positive deviation   -> RISK (things got worse)
    lower_is_better metric  + negative deviation   -> OPPORTUNITY (things improved)

    `lower_is_better`, when explicitly passed (True/False), overrides the
    LOWER_IS_BETTER_METRICS name lookup - this lets callers working with
    user-uploaded / arbitrary metric names (where the metric_label isn't
    one of the known built-in names) specify direction directly instead
    of silently defaulting to "higher is better".
    """
    if lower_is_better is None:
        lower_is_better = metric_label in LOWER_IS_BETTER_METRICS

    if not np.isfinite(deviation_pct):
        # An infinite deviation (baseline was exactly zero) is always
        # treated conservatively as a risk needing investigation, since
        # direction can't be meaningfully compared against a zero baseline
        # in the usual way.
        return RISK

    went_up = deviation_pct > 0

    if lower_is_better:
        return RISK if went_up else OPPORTUNITY
    else:
        return OPPORTUNITY if went_up else RISK


@dataclass
class Anomaly:
    metric: str
    dimension: str          # e.g. "region"
    dimension_value: str    # e.g. "North"
    week: int
    actual: float
    expected: float         # baseline median at the time - the robust "normal" value
    deviation_pct: float
    modified_z_score: float
    anomaly_score: float    # normalized 0-1, see compute_anomaly_score()
    severity: str            # "MEDIUM" or "HIGH" (below-threshold weeks aren't Anomaly objects at all)
    nature: str = RISK       # "OPPORTUNITY" or "RISK" - see classify_nature()
    # --- Dynamic threshold transparency (see compute_dynamic_threshold()) ---
    # The threshold this specific anomaly was actually judged against, plus
    # the inputs that produced it - shown in the UI so an analyst can see
    # WHY a stream's bar was raised/lowered, not just that it was.
    threshold_used: float = ANOMALY_THRESHOLD
    threshold_cv: float = 0.0
    threshold_tier: str = "MEDIUM"

    def __str__(self) -> str:
        sign = "+" if self.deviation_pct >= 0 else ""
        nature_label = "📈 Growth Opportunity" if self.nature == OPPORTUNITY else "⚠️ Risk"
        return (
            f"Metric: {self.metric} ({self.dimension}={self.dimension_value}, week {self.week})\n"
            f"Nature: {nature_label}\n"
            f"Actual: {self.actual:,.1f}\n"
            f"Expected: {self.expected:,.1f}\n"
            f"Deviation: {sign}{self.deviation_pct:.1%}\n"
            f"Modified Z-Score: {self.modified_z_score:.2f}\n"
            f"Anomaly Score: {self.anomaly_score:.2f}\n"
            f"Severity: {self.severity}\n"
            f"Threshold used: {self.threshold_used:.2f} (CV={self.threshold_cv:.2f}, tier={self.threshold_tier})"
        )


# ---------------------------------------------------------------------------
# Anomaly scoring helpers
# ---------------------------------------------------------------------------

def compute_anomaly_score(modified_z: float) -> float:
    """
    Normalizes |modified_z| to a 0-1 "anomaly score" for display purposes
    (e.g. "Anomaly Score: 0.91" as shown in the project spec).

    Formula: 1 - exp(-|z| / 5), a saturating curve approaching 1 as |z|
    grows. Deliberately continuous and independent of ANOMALY_THRESHOLD,
    so scores stay comparable across metrics regardless of what threshold
    was used to flag them.
    """

    if not np.isfinite(modified_z):
        return 1.0

    return round(1 - np.exp(-abs(modified_z) / 5.0), 2)


def classify_severity(modified_z: float) -> str:
    if not np.isfinite(modified_z):
        return "HIGH"

    return "HIGH" if abs(modified_z) >= HIGH_SEVERITY_THRESHOLD else "MEDIUM"


def _median_and_mad(values: list[float]) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(np.array(values) - median)))
    return median, mad


# ---------------------------------------------------------------------------
# Core detection: causal, self-correcting
# ---------------------------------------------------------------------------

def detect_anomalies(
    weekly_df: pd.DataFrame,
    value_col: str,
    metric_label: str,
    dimension: str,
    window: int = ROLLING_WINDOW,
    threshold: float = ANOMALY_THRESHOLD,
    min_meaningful_deviation: float = MIN_MEANINGFUL_DEVIATION,
    lower_is_better: bool | None = None,
    use_dynamic_threshold: bool = True,
    user_sensitivity_multiplier: float = 1.0,
    tier_cutoffs: dict | None = None,
    tier_factors: dict | None = None,
) -> list[Anomaly]:
    """
    Runs anomaly detection on one metric column, independently within
    each value of `dimension` (e.g. each region).

    Processes weeks in chronological order; the baseline for each week is
    the median/MAD of the last `window` clean historical observations that
    were NOT themselves flagged anomalous.

    `lower_is_better`, when given, overrides classify_nature()'s built-in
    name lookup - needed for arbitrary/user-uploaded metric names where
    the metric_label isn't one of the pipeline's known built-in metrics.

    DYNAMIC THRESHOLD (default on): instead of comparing every stream's
    |modified_z| against the same fixed `threshold`, each week's baseline
    window is fed through compute_dynamic_threshold() (see the "DYNAMIC
    THRESHOLD ENGINE" section near the top of this file) to produce a
    threshold scaled to THAT stream's own volatility (Coefficient of
    Variation) and revenue/volume tier, further adjusted by
    `user_sensitivity_multiplier`. `threshold` is still used - as the
    reference anchor the dynamic engine scales up or down from, not as
    an absolute cutoff. Set `use_dynamic_threshold=False` to restore the
    original fixed-threshold behavior exactly (e.g. for A/B comparison).

    Returns only weeks that cross the (dynamic or fixed) threshold - per
    the project spec, "do NOT blindly label every change as an anomaly."
    Most weeks produce no Anomaly at all.
    """

    anomalies: list[Anomaly] = []

    for dim_value, group in weekly_df.groupby(dimension):
        group = group.sort_values("week")
        clean_history: list[float] = []  # values from weeks judged "normal" so far

        for _, row in group.iterrows():
            week = int(row["week"])
            actual = row[value_col]

            # -----------------------------------------------------------
            # Handle missing values
            # -----------------------------------------------------------
            if pd.isna(actual):
                continue

            actual = float(actual)

            if len(clean_history) < window:
                # Not enough clean history yet to establish a baseline -
                # same "no statistic on insufficient data" policy as
                # data_processor.py. Build up history, don't score.
                clean_history.append(actual)
                continue

            baseline_values = clean_history[-window:]
            median, mad = _median_and_mad(baseline_values)

            # -----------------------------------------------------------
            # Compute this week's effective threshold from ITS OWN
            # baseline window - see DYNAMIC THRESHOLD ENGINE above.
            # -----------------------------------------------------------
            if use_dynamic_threshold:
                threshold_info = compute_dynamic_threshold(
                    baseline_values,
                    base_threshold=threshold,
                    user_multiplier=user_sensitivity_multiplier,
                    tier_cutoffs=tier_cutoffs,
                    tier_factors=tier_factors,
                )
            else:
                threshold_info = {"threshold": threshold, "cv": 0.0, "tier": "MEDIUM"}
            effective_threshold = threshold_info["threshold"]

            # -----------------------------------------------------------
            # Calculate percentage deviation from baseline
            # -----------------------------------------------------------
            if median != 0:
                deviation_pct = (actual - median) / median
            else:
                # If baseline median is zero:
                # 0 -> 0 means no deviation.
                # 0 -> non-zero is treated as an infinite relative change.
                deviation_pct = (
                    0.0
                    if actual == 0
                    else (float("inf") if actual > 0 else float("-inf"))
                )

            # -----------------------------------------------------------
            # Handle MAD == 0
            # -----------------------------------------------------------
            if mad == 0:
                # A zero MAD means the historical baseline is perfectly flat.
                # If the new value differs meaningfully from that flat baseline,
                # it should be treated as an anomaly rather than silently
                # accepted as normal.

                if (
                    actual != median
                    and abs(deviation_pct) >= min_meaningful_deviation
                ):
                    modified_z = (
                        float("inf")
                        if actual > median
                        else float("-inf")
                    )

                    anomalies.append(
                        Anomaly(
                            metric=metric_label,
                            dimension=dimension,
                            dimension_value=dim_value,
                            week=week,
                            actual=actual,
                            expected=median,
                            deviation_pct=deviation_pct,
                            modified_z_score=modified_z,
                            anomaly_score=1.0,
                            severity="HIGH",
                            nature=classify_nature(metric_label, deviation_pct, lower_is_better=lower_is_better),
                            threshold_used=effective_threshold,
                            threshold_cv=threshold_info["cv"],
                            threshold_tier=threshold_info["tier"],
                        )
                    )

                    # Deliberately NOT added to clean_history.
                    continue

                # No meaningful deviation from the flat baseline.
                clean_history.append(actual)
                continue

            # -----------------------------------------------------------
            # Calculate Modified Z-Score
            # -----------------------------------------------------------
            modified_z = (
                MAD_SCALE_CONSTANT * (actual - median) / mad
            )

            # -----------------------------------------------------------
            # Determine whether this is an anomaly, against THIS week's
            # dynamic (or fixed, if disabled) threshold.
            # -----------------------------------------------------------
            is_anomaly = (
                abs(modified_z) >= effective_threshold
                and abs(deviation_pct) >= min_meaningful_deviation
            )

            if is_anomaly:
                anomalies.append(
                    Anomaly(
                        metric=metric_label,
                        dimension=dimension,
                        dimension_value=dim_value,
                        week=week,
                        actual=actual,
                        expected=median,
                        deviation_pct=deviation_pct,
                        modified_z_score=round(modified_z, 2),
                        anomaly_score=compute_anomaly_score(modified_z),
                        severity=classify_severity(modified_z),
                        nature=classify_nature(metric_label, deviation_pct, lower_is_better=lower_is_better),
                        threshold_used=effective_threshold,
                        threshold_cv=threshold_info["cv"],
                        threshold_tier=threshold_info["tier"],
                    )
                )

                # Deliberately NOT added to clean_history - this is the
                # fix: an anomalous week must not become part of the
                # baseline that future weeks are judged against.

            else:
                clean_history.append(actual)

    return sorted(
        anomalies,
        key=lambda a: abs(a.modified_z_score),
        reverse=True
    )


# ---------------------------------------------------------------------------
# Run detection across the project's key metrics
# ---------------------------------------------------------------------------

def detect_all_anomalies(user_sensitivity_multiplier: float = 1.0) -> list[Anomaly]:
    """
    Runs detection across the metrics this project cares about most:
    revenue, units sold, and average delivery days (each per region).

    `user_sensitivity_multiplier` is passed straight through to
    detect_anomalies()'s dynamic threshold engine for every metric - this
    is the single knob a UI slider needs to bind to make the WHOLE
    dashboard more aggressive or more conservative at once.

    Returns a single combined, ranked list.
    """

    processed = build_processed_dataset()

    revenue_anomalies = detect_anomalies(
        processed["sales_weekly"],
        value_col="revenue",
        metric_label="Revenue",
        dimension="region",
        user_sensitivity_multiplier=user_sensitivity_multiplier,
    )

    units_anomalies = detect_anomalies(
        processed["sales_weekly"],
        value_col="units_sold",
        metric_label="Units Sold",
        dimension="region",
        user_sensitivity_multiplier=user_sensitivity_multiplier,
    )

    delivery_anomalies = detect_anomalies(
        processed["delivery_weekly"],
        value_col="avg_delivery_days",
        metric_label="Avg Delivery Days",
        dimension="region",
        user_sensitivity_multiplier=user_sensitivity_multiplier,
    )

    all_anomalies = (
        revenue_anomalies
        + units_anomalies
        + delivery_anomalies
    )

    return sorted(
        all_anomalies,
        key=lambda a: abs(a.modified_z_score),
        reverse=True
    )


if __name__ == "__main__":
    anomalies = detect_all_anomalies()

    print(
        f"Detected {len(anomalies)} anomalies "
        f"(|modified z| >= {ANOMALY_THRESHOLD}) "
        f"across all metrics.\n"
    )

    for a in anomalies:
        print(a)
        print("-" * 40)