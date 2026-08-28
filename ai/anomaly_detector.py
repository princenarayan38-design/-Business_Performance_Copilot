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
ANOMALY_THRESHOLD = 3.5   # |modified_z| above this = anomaly
HIGH_SEVERITY_THRESHOLD = 6.0  # |modified_z| above this = HIGH severity
MIN_MEANINGFUL_DEVIATION = 0.05  # a move must also be >=5% to count as an anomaly,
                                  # not just statistically unusual - see detect_anomalies()

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
            f"Severity: {self.severity}"
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

    Returns only weeks that cross the threshold - per the project spec,
    "do NOT blindly label every change as an anomaly." Most weeks produce
    no Anomaly at all.
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
            # Determine whether this is an anomaly
            # -----------------------------------------------------------
            is_anomaly = (
                abs(modified_z) >= threshold
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

def detect_all_anomalies() -> list[Anomaly]:
    """
    Runs detection across the metrics this project cares about most:
    revenue, units sold, and average delivery days (each per region).

    Returns a single combined, ranked list.
    """

    processed = build_processed_dataset()

    revenue_anomalies = detect_anomalies(
        processed["sales_weekly"],
        value_col="revenue",
        metric_label="Revenue",
        dimension="region"
    )

    units_anomalies = detect_anomalies(
        processed["sales_weekly"],
        value_col="units_sold",
        metric_label="Units Sold",
        dimension="region"
    )

    delivery_anomalies = detect_anomalies(
        processed["delivery_weekly"],
        value_col="avg_delivery_days",
        metric_label="Avg Delivery Days",
        dimension="region"
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