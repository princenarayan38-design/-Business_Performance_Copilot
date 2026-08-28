"""
data_processor.py

Turns the raw, event-level CSVs (one row per date x region x product) into
weekly-aggregated tables with rolling mean/std, which is what
anomaly_detector.py (Phase 3) needs to compute z-scores.

WHY WEEKLY, NOT DAILY
----------------------
Daily units_sold has meaningful day-to-day noise (see generate_data.py -
+/-8% random daily fluctuation is intentional). Comparing single days against
each other would trigger constant false anomalies. Weekly aggregation smooths
that noise while still being fine-grained enough to isolate a multi-week
event like the North region disruption.

WHY A MINIMUM WINDOW (no shrinking window)
--------------------------------------------
A rolling mean/std computed from only 1-2 prior weeks is statistically
unstable (or undefined, for std with n=1). Per the project's rule against
manufacturing statistically-valid-looking numbers without justification,
this module leaves rolling_mean/rolling_std as NaN until ROLLING_WINDOW
full weeks of prior history exist, rather than silently returning a
low-confidence number that looks the same as a high-confidence one.
"""

from __future__ import annotations

import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
ROLLING_WINDOW = 4  # weeks of trailing history required before scoring


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_data() -> dict[str, pd.DataFrame]:
    """Load the four raw CSVs and add a 1-indexed `week` column to each."""
    sales = pd.read_csv(os.path.join(RAW_DIR, "sales.csv"), parse_dates=["date"])
    delivery = pd.read_csv(os.path.join(RAW_DIR, "delivery.csv"), parse_dates=["date"])
    feedback = pd.read_csv(os.path.join(RAW_DIR, "customer_feedback.csv"), parse_dates=["date"])
    events = pd.read_csv(os.path.join(RAW_DIR, "events.csv"), parse_dates=["date"])

    min_date = sales["date"].min()
    for df in (sales, delivery, feedback):
        df["week"] = ((df["date"] - min_date).dt.days // 7) + 1

    return {"sales": sales, "delivery": delivery, "feedback": feedback, "events": events}


# ---------------------------------------------------------------------------
# Weekly aggregation
# ---------------------------------------------------------------------------

def aggregate_sales_weekly(sales: pd.DataFrame, dimension: str | None = "region") -> pd.DataFrame:
    """
    Aggregate sales to weekly grain, optionally split by a dimension
    (e.g. "region" or "product_id"). Pass dimension=None for an overall
    (all-regions, all-products) weekly total.

    Returns columns: week, [dimension], units_sold, revenue, avg_price,
    avg_marketing_spend, avg_discount
    """
    group_cols = ["week"] if dimension is None else ["week", dimension]

    agg = sales.groupby(group_cols).agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        avg_price=("price", "mean"),
        avg_marketing_spend=("marketing_spend", "mean"),
        avg_discount=("discount", "mean"),
    ).reset_index()

    return agg.sort_values(group_cols).reset_index(drop=True)


def aggregate_delivery_weekly(delivery: pd.DataFrame, dimension: str | None = "region") -> pd.DataFrame:
    """
    Returns columns: week, [dimension], avg_delivery_days, avg_late_rate
    """
    group_cols = ["week"] if dimension is None else ["week", dimension]

    agg = delivery.groupby(group_cols).agg(
        avg_delivery_days=("average_delivery_days", "mean"),
        avg_late_rate=("late_delivery_rate", "mean"),
    ).reset_index()

    return agg.sort_values(group_cols).reset_index(drop=True)


def aggregate_feedback_weekly(feedback: pd.DataFrame, dimension: str | None = "region") -> pd.DataFrame:
    """
    Returns columns: week, [dimension], category, total_feedback,
    negative_feedback, negative_rate

    Kept split by `category` (e.g. "delivery", "product_quality") in
    addition to the requested dimension, since evidence_retriever.py
    (Phase 4) needs to distinguish complaint types, not just totals.
    """
    group_cols = ["week"] if dimension is None else ["week", dimension]
    group_cols_with_category = group_cols + ["category"]

    total = feedback.groupby(group_cols_with_category).size().rename("total_feedback")
    negative = (
        feedback[feedback["sentiment"] == "negative"]
        .groupby(group_cols_with_category)
        .size()
        .rename("negative_feedback")
    )

    agg = pd.concat([total, negative], axis=1).fillna(0).reset_index()
    agg["negative_feedback"] = agg["negative_feedback"].astype(int)
    agg["negative_rate"] = (agg["negative_feedback"] / agg["total_feedback"]).round(3)

    return agg.sort_values(group_cols_with_category).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rolling statistics (the input anomaly_detector.py needs)
# ---------------------------------------------------------------------------

def add_rolling_stats(
    weekly_df: pd.DataFrame,
    value_col: str,
    group_cols: list[str] | None = None,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Adds `rolling_mean` and `rolling_std` columns computed over the
    trailing `window` weeks BEFORE the current week (i.e. the current
    week's own value is never included in its own expected-value
    calculation - that would be circular).

    If group_cols is given (e.g. ["region"]), rolling stats are computed
    independently within each group, so North's baseline is never
    contaminated by South's numbers and vice versa.

    Weeks with fewer than `window` prior weeks of history get NaN -
    see module docstring for why.
    """
    df = weekly_df.sort_values((group_cols or []) + ["week"]).copy()

    if group_cols:
        # Two-step: (1) shift within each group so the current week's own
        # value never leaks into its own "expected" baseline, (2) roll the
        # shifted series within each group. Using transform() (not apply())
        # avoids pandas' version-dependent handling of grouping columns
        # inside apply() - transform always returns a same-length Series
        # aligned to df's index, nothing to lose track of.
        shifted = df.groupby(group_cols)[value_col].shift(1)
        df["_shifted"] = shifted
        grouped_shifted = df.groupby(group_cols)["_shifted"]
        df["rolling_mean"] = grouped_shifted.transform(
            lambda s: s.rolling(window=window, min_periods=window).mean()
        )
        df["rolling_std"] = grouped_shifted.transform(
            lambda s: s.rolling(window=window, min_periods=window).std()
        )
        df = df.drop(columns=["_shifted"])
    else:
        shifted = df[value_col].shift(1)
        df["rolling_mean"] = shifted.rolling(window=window, min_periods=window).mean()
        df["rolling_std"] = shifted.rolling(window=window, min_periods=window).std()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience: run the full processing pipeline
# ---------------------------------------------------------------------------

def build_processed_dataset() -> dict[str, pd.DataFrame]:
    """
    Loads raw data and returns weekly-aggregated, rolling-stat-enriched
    tables ready for anomaly_detector.py. This is the main entry point
    other modules should import.
    """
    raw = load_raw_data()

    sales_weekly = aggregate_sales_weekly(raw["sales"], dimension="region")
    sales_weekly = add_rolling_stats(sales_weekly, value_col="units_sold", group_cols=["region"])

    delivery_weekly = aggregate_delivery_weekly(raw["delivery"], dimension="region")
    delivery_weekly = add_rolling_stats(delivery_weekly, value_col="avg_delivery_days", group_cols=["region"])

    feedback_weekly = aggregate_feedback_weekly(raw["feedback"], dimension="region")

    return {
        "sales_weekly": sales_weekly,
        "delivery_weekly": delivery_weekly,
        "feedback_weekly": feedback_weekly,
        "events": raw["events"],
        "min_date": raw["sales"]["date"].min(),
    }


if __name__ == "__main__":
    processed = build_processed_dataset()

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data", "processed"), exist_ok=True)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    for name, df in processed.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        print(f"{name}: {len(df)} rows -> data/processed/{name}.csv")

    print("\nSample: North region, weeks 13-19 (sales_weekly):")
    north = processed["sales_weekly"]
    print(north[(north.region == "North") & (north.week.between(13, 19))].to_string(index=False))
