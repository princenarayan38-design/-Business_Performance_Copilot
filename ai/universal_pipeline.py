"""
universal_pipeline.py

Turns an ARBITRARY user-uploaded table into the shape the existing
anomaly_detector.detect_anomalies() already expects: a weekly-aggregated
dataframe with a numeric value column and (optionally) a category/
dimension column.

This is deliberately separate from data_processor.py, which stays as the
curated demo pipeline for the project's own sample CSVs. This module is
the productized path: no assumptions about column names, file names, or
what the business actually measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ColumnMapping:
    date_col: str
    value_col: str
    dimension_col: str | None      # None = analyze as a single overall series
    lower_is_better: bool           # False = higher values are good news (revenue-like)
    aggregation: str = "sum"        # "sum" | "mean" - how to roll multiple rows/week together


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Reads a Streamlit UploadedFile (CSV or Excel) into a DataFrame.
    Raises a ValueError with a clear message on unsupported formats
    instead of letting pandas throw an opaque error.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: '{uploaded_file.name}'. Please upload a .csv or .xlsx file.")


def validate_mapping(df: pd.DataFrame, mapping: ColumnMapping) -> list[str]:
    """
    Sanity-checks a proposed column mapping against the actual data
    BEFORE running the pipeline, so the user gets a clear message instead
    of a stack trace. Returns a list of problem descriptions (empty = OK).
    """
    problems = []

    if mapping.date_col not in df.columns:
        problems.append(f"Date/time column '{mapping.date_col}' not found in the file.")
    if mapping.value_col not in df.columns:
        problems.append(f"Metric column '{mapping.value_col}' not found in the file.")
    if mapping.dimension_col and mapping.dimension_col not in df.columns:
        problems.append(f"Category column '{mapping.dimension_col}' not found in the file.")

    if mapping.value_col in df.columns:
        non_numeric = pd.to_numeric(df[mapping.value_col], errors="coerce").isna().sum()
        if non_numeric > 0:
            problems.append(
                f"Metric column '{mapping.value_col}' has {non_numeric} non-numeric value(s) "
                "that will be dropped from analysis."
            )

    if mapping.date_col in df.columns:
        parsed_dates = pd.to_datetime(df[mapping.date_col], errors="coerce")
        bad_dates = parsed_dates.isna().sum()
        if bad_dates > 0:
            problems.append(
                f"Date column '{mapping.date_col}' has {bad_dates} value(s) that couldn't be "
                "parsed as dates and will be dropped."
            )
        elif parsed_dates.nunique() < 8:
            problems.append(
                f"Only {parsed_dates.nunique()} distinct dates found — the anomaly detector needs "
                "at least ~8 weeks of history per category to build a reliable baseline. Results may "
                "be sparse or empty."
            )

    return problems


def build_weekly_series(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    """
    Aggregates the raw user table into the weekly (week, [dimension_value],
    value) shape that detect_anomalies() consumes — mirroring what
    data_processor.aggregate_sales_weekly() does for the demo data, but
    driven entirely by the user's column choices instead of hardcoded
    'region'/'revenue' names.
    """
    work = df.copy()

    work[mapping.date_col] = pd.to_datetime(work[mapping.date_col], errors="coerce")
    work[mapping.value_col] = pd.to_numeric(work[mapping.value_col], errors="coerce")
    work = work.dropna(subset=[mapping.date_col, mapping.value_col])

    if work.empty:
        raise ValueError("After cleaning, no valid rows remain — check the date and metric column mapping.")

    min_date = work[mapping.date_col].min()
    work["week"] = ((work[mapping.date_col] - min_date).dt.days // 7) + 1

    if mapping.dimension_col:
        work["dimension_value"] = work[mapping.dimension_col].astype(str)
        group_cols = ["week", "dimension_value"]
    else:
        work["dimension_value"] = "Overall"
        group_cols = ["week", "dimension_value"]

    agg_func = "mean" if mapping.aggregation == "mean" else "sum"
    weekly = (
        work.groupby(group_cols)[mapping.value_col]
        .agg(agg_func)
        .reset_index()
        .rename(columns={mapping.value_col: "value"})
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    return weekly


def suggest_column_roles(df: pd.DataFrame) -> dict:
    """
    Best-effort auto-suggestions for the mapping dropdowns, based on
    dtypes and common naming patterns. The UI still lets the user
    override every suggestion — this just saves clicks for the common case.
    """
    suggestions = {"date_col": None, "value_col": None, "dimension_col": None}

    date_keywords = {"date", "time", "day", "timestamp", "period"}
    value_keywords = {"revenue", "sales", "amount", "value", "units", "count", "price", "cost", "score"}
    dimension_keywords = {"region", "category", "segment", "product", "store", "group", "department", "type"}

    for col in df.columns:
        col_lower = col.lower()

        if suggestions["date_col"] is None:
            is_datetime_dtype = pd.api.types.is_datetime64_any_dtype(df[col])
            if is_datetime_dtype or any(k in col_lower for k in date_keywords):
                suggestions["date_col"] = col
                continue

        if suggestions["value_col"] is None:
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            if is_numeric and any(k in col_lower for k in value_keywords):
                suggestions["value_col"] = col
                continue

        if suggestions["dimension_col"] is None:
            is_low_cardinality_text = df[col].dtype == object and df[col].nunique() <= max(20, len(df) // 5)
            if any(k in col_lower for k in dimension_keywords) and is_low_cardinality_text:
                suggestions["dimension_col"] = col
                continue

    # Fallback: if no value_col matched by keyword, just take the first
    # purely numeric column that isn't obviously an ID.
    if suggestions["value_col"] is None:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and "id" not in col.lower():
                suggestions["value_col"] = col
                break

    return suggestions