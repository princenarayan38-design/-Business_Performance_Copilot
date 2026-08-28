"""
forecasting.py

Projects a metric forward N weeks with a confidence band, for the
"what's likely to happen next" view alongside the historical trend chart.

Two backends:
  - Prophet, if installed (pip install prophet) - handles trend +
    seasonality properly and is the better choice for real deployments.
  - A lightweight linear-trend fallback (numpy polyfit + residual-based
    confidence band) that needs no extra dependencies - always available,
    so the feature never breaks if Prophet isn't installed.

Both return the SAME output shape (a DataFrame with week, forecast,
lower, upper) so the Streamlit/Plotly code that consumes this doesn't
need to know or care which backend produced it.

HONESTY NOTE: this is a simple trend projection, not a validated
forecasting model. With ~20-30 weeks of history and no seasonality
modeling in the fallback, treat the confidence band as a rough
guide, not a statistical guarantee - see the caption shown in the UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _linear_trend_forecast(series: pd.DataFrame, horizon: int, confidence_z: float = 1.96) -> pd.DataFrame:
    """
    Fallback forecaster: fits a straight line to the historical weekly
    values via least squares, projects it forward `horizon` weeks, and
    builds a confidence band from the in-sample residual standard
    deviation (widening slightly with distance into the future, since
    uncertainty compounds the further out you project).
    """
    weeks = series["week"].to_numpy(dtype=float)
    values = series["value"].to_numpy(dtype=float)

    if len(weeks) < 3:
        raise ValueError("Need at least 3 historical weeks to fit a trend forecast.")

    slope, intercept = np.polyfit(weeks, values, 1)
    fitted = slope * weeks + intercept
    residual_std = float(np.std(values - fitted, ddof=1)) if len(weeks) > 2 else 0.0

    last_week = int(weeks.max())
    future_weeks = np.arange(last_week + 1, last_week + horizon + 1)
    point_forecast = slope * future_weeks + intercept

    # Widen the band the further out we project - uncertainty compounds.
    steps_ahead = np.arange(1, horizon + 1)
    band_width = confidence_z * residual_std * np.sqrt(steps_ahead)

    return pd.DataFrame({
        "week": future_weeks,
        "forecast": point_forecast,
        "lower": point_forecast - band_width,
        "upper": point_forecast + band_width,
    })


def _prophet_forecast(series: pd.DataFrame, horizon: int, min_date: pd.Timestamp) -> pd.DataFrame:
    """Prophet-backed forecaster, used only if the `prophet` package is installed."""
    from prophet import Prophet  # local import - optional dependency

    prophet_df = pd.DataFrame({
        "ds": min_date + pd.to_timedelta((series["week"] - 1) * 7, unit="D"),
        "y": series["value"],
    })

    model = Prophet(interval_width=0.95)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon, freq="W")
    forecast = model.predict(future)

    future_only = forecast.tail(horizon).copy()
    future_only["week"] = series["week"].max() + np.arange(1, horizon + 1)

    return future_only.rename(columns={"yhat": "forecast", "yhat_lower": "lower", "yhat_upper": "upper"})[
        ["week", "forecast", "lower", "upper"]
    ]


def generate_forecast(
    series: pd.DataFrame,
    horizon: int,
    min_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    series: DataFrame with columns ["week", "value"] (one row per week,
            already filtered to a single dimension/category if relevant).
    horizon: number of future weeks to project.
    min_date: needed only for the Prophet backend, to convert week numbers
              back into real calendar dates.

    Returns (forecast_df, backend_used) where backend_used is "prophet"
    or "linear_trend" - shown in the UI so users know which method ran.
    """
    if len(series) < 3:
        raise ValueError("Need at least 3 weeks of historical data to forecast.")

    if min_date is not None:
        try:
            return _prophet_forecast(series, horizon, min_date), "prophet"
        except ImportError:
            pass
        except Exception:
            # Prophet installed but failed for some data-specific reason -
            # fall back rather than breaking the whole forecast feature.
            pass

    return _linear_trend_forecast(series, horizon), "linear_trend"