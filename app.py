"""
app.py
Streamlit Dashboard for Business Intelligence AI Pipeline with Local Ollama Integration.
Team Scubacats - Accenture Innovation Challenge 2026
"""
import re
import streamlit as st
import os
import ollama
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

from ai.anomaly_detector import detect_anomalies, detect_all_anomalies
from ai.data_processor import build_processed_dataset
from ai.evidence_retriever import retrieve_evidence_for_anomaly
from ai.hypothesis_generator import generate_hypotheses_for_anomaly
from ai.hypothesis_tester import test_hypotheses
from ai.reporter import generate_executive_report
from ai.universal_pipeline import (
    load_uploaded_file, validate_mapping, build_weekly_series,
    suggest_column_roles, ColumnMapping,
)
from ai.feedback_store import record_feedback, get_feedback, get_calibration_stats, CONFIRMED, REJECTED
from ai.model_trainer import train_model, get_model_status
from ai.event_context import (
    load_events_from_dataframe, load_events_from_csv, get_events_for_week, format_events_for_prompt,
)
from ai.forecasting import generate_forecast
from ai.generic_evidence import GenericEvidenceSource, retrieve_generic_evidence
from ai.generic_hypothesis import generate_generic_hypotheses

st.set_page_config(page_title="KPI Storytelling Agent", layout="wide")

# ---------------------------------------------------------------------------
# High-tech "command center" theme — glowing cards, gradient banners,
# animated status pulses. Pure CSS injected once; no external assets.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
:root {
    --accent-cyan: #00e5ff;
    --accent-purple: #a855f7;
    --accent-pink: #ec4899;
    --accent-green: #22ffb0;
    --accent-orange: #ff9d3d;
    --accent-red: #ff4d6d;
    --panel-bg: rgba(255,255,255,0.035);
}

/* Page background: subtle radial glow grid */
.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(0,229,255,0.07), transparent 40%),
        radial-gradient(circle at 85% 0%, rgba(168,85,247,0.08), transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(236,72,153,0.05), transparent 50%),
        #0a0e17;
}

/* Section headers get a gradient underline */
h1, h2, h3 { letter-spacing: 0.3px; }
h2 { border-bottom: 2px solid transparent; border-image: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple)) 1; padding-bottom: 6px; }

/* Glowing metric / stat cards */
.glow-card {
    background: var(--panel-bg);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 18px 20px;
    backdrop-filter: blur(6px);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glow-card:hover { transform: translateY(-3px); }
.glow-card::before {
    content: "";
    position: absolute; inset: 0;
    padding: 1px;
    border-radius: 16px;
    background: linear-gradient(135deg, var(--card-glow, var(--accent-cyan)), transparent 60%);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    opacity: 0.55;
}
.glow-card .glow-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.2px; color: #9aa4b2; margin-bottom: 6px; }
.glow-card .glow-value { font-size: 1.9rem; font-weight: 800; color: #f5f7fa; line-height: 1.1; }
.glow-card .glow-sub { font-size: 0.82rem; color: #7f8a99; margin-top: 4px; }

/* Severity-tinted glow variants */
.glow-high   { --card-glow: var(--accent-red); box-shadow: 0 0 24px -8px rgba(255,77,109,0.45); }
.glow-medium { --card-glow: var(--accent-orange); box-shadow: 0 0 24px -8px rgba(255,157,61,0.4); }
.glow-good   { --card-glow: var(--accent-green); box-shadow: 0 0 24px -8px rgba(34,255,176,0.4); }
.glow-info   { --card-glow: var(--accent-cyan); box-shadow: 0 0 24px -8px rgba(0,229,255,0.4); }
.glow-purple { --card-glow: var(--accent-purple); box-shadow: 0 0 24px -8px rgba(168,85,247,0.4); }

/* Sleek notification banner replacing plain st.success/st.warning look */
.notif-banner {
    display: flex; align-items: center; gap: 14px;
    background: linear-gradient(90deg, rgba(0,229,255,0.12), rgba(168,85,247,0.10));
    border-left: 4px solid var(--accent-cyan);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0 18px 0;
}
.notif-banner.danger { border-left-color: var(--accent-red); background: linear-gradient(90deg, rgba(255,77,109,0.14), rgba(255,157,61,0.08)); }
.notif-banner .pulse-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--accent-cyan); flex-shrink: 0;
    box-shadow: 0 0 0 0 rgba(0,229,255,0.6);
    animation: pulse 1.8s infinite;
}
.notif-banner.danger .pulse-dot { background: var(--accent-red); box-shadow: 0 0 0 0 rgba(255,77,109,0.6); }
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(0,229,255,0.55); }
    70%  { box-shadow: 0 0 0 10px rgba(0,229,255,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,229,255,0); }
}

/* Verdict pill badges */
.pill { display:inline-block; padding: 3px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.4px; }
.pill-validated   { background: rgba(34,255,176,0.15); color: var(--accent-green); border: 1px solid rgba(34,255,176,0.4); }
.pill-inconclusive{ background: rgba(255,157,61,0.15); color: var(--accent-orange); border: 1px solid rgba(255,157,61,0.4); }
.pill-refuted     { background: rgba(255,77,109,0.15); color: var(--accent-red); border: 1px solid rgba(255,77,109,0.4); }

/* Sidebar tint */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1220, #0a0e17); border-right: 1px solid rgba(255,255,255,0.06); }

/* Sidebar section headers read like nav group labels, not paragraph text */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    color: #7f8a99 !important;
    border: none !important;
    margin: 18px 0 6px 0 !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] .stButton button {
    border-radius: 10px;
    border: 1px solid rgba(0,229,255,0.35);
}

/* Flatten st.expander into list-row cards: compact, rounded, no default chrome */
div[data-testid="stExpander"] {
    background: var(--panel-bg);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    margin-bottom: 10px;
    overflow: hidden;
}
div[data-testid="stExpander"] summary { padding: 12px 16px !important; font-weight: 600; }
div[data-testid="stExpander"] summary:hover { background: rgba(255,255,255,0.03); }
div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] { padding: 4px 16px 16px 16px; }

/* Top tab bar (taskbar) styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px 10px 0 0;
    padding: 10px 18px;
    color: #9aa4b2;
}
.stTabs [aria-selected="true"] {
    background: rgba(255,255,255,0.04);
    color: #f5f7fa !important;
    border-bottom: 2px solid var(--accent-cyan);
}

/* Sub-tab bar (AI Hypotheses / Validation & Evidence / Executive Briefing):
   higher-contrast headers with a distinct highlight on the active tab. */
.stTabs .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02);
    border-radius: 10px;
    padding: 4px;
    border-bottom: none;
}
.stTabs .stTabs [data-baseweb="tab"] {
    color: #b7c0cc;
    font-weight: 600;
    border-radius: 8px;
    padding: 9px 16px;
}
.stTabs .stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(0,229,255,0.22), rgba(168,85,247,0.18));
    color: #ffffff !important;
    border-bottom: 2px solid var(--accent-purple);
    box-shadow: 0 0 14px -4px rgba(0,229,255,0.5);
}

/* Confidence percentage badge (Validation & Evidence tab) */
.confidence-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.3px;
}
.confidence-high   { background: rgba(34,255,176,0.15); color: var(--accent-green); border: 1px solid rgba(34,255,176,0.4); }
.confidence-medium { background: rgba(255,157,61,0.15); color: var(--accent-orange); border: 1px solid rgba(255,157,61,0.4); }
.confidence-low    { background: rgba(255,77,109,0.15); color: var(--accent-red); border: 1px solid rgba(255,77,109,0.4); }

/* Small inline info-tooltip trigger for raw-stat details */
.stat-tooltip {
    display: inline-block; margin-left: 6px; cursor: help;
    color: #7f8a99; border: 1px solid rgba(255,255,255,0.25);
    border-radius: 50%; width: 15px; height: 15px; line-height: 13px;
    text-align: center; font-size: 0.68rem; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


def glow_card(label: str, value: str, sub: str = "", variant: str = "info"):
    """Renders one glowing stat card. variant: info|high|medium|good|purple"""
    st.markdown(f"""
    <div class="glow-card glow-{variant}">
        <div class="glow-label">{label}</div>
        <div class="glow-value">{value}</div>
        <div class="glow-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def notif_banner(text: str, danger: bool = False):
    cls = "notif-banner danger" if danger else "notif-banner"
    st.markdown(f"""
    <div class="{cls}"><span class="pulse-dot"></span><span>{text}</span></div>
    """, unsafe_allow_html=True)


def mini_gauge(value_pct: float, label: str, color: str = "#a855f7", height: int = 140):
    """Small ring gauge for a 0-100 value, used for calibration agreement rate
    and confidence-model train accuracy so those read as glanceable visuals."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_pct,
        number={"suffix": "%", "font": {"size": 20, "color": "#f5f7fa"}},
        gauge={
            "axis": {"range": [0, 100], "visible": False},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.06)",
            "borderwidth": 0,
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6e9ef"),
        margin=dict(l=10, r=10, t=10, b=0),
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"gauge_{label}")
    st.caption(label)


def score_badge(score: float) -> str:
    """Same visual treatment as confidence_badge, but tolerant of scores
    already expressed on a 0-100 scale (as opposed to 0-1)."""
    normalized = score / 100.0 if score > 1 else score
    return confidence_badge(normalized)


def confidence_badge(score: float) -> str:
    """Turns a raw 0-1 confidence score into an exec-readable percentage badge."""
    pct = max(0.0, min(1.0, score)) * 100
    if pct >= 70:
        cls, tag = "confidence-high", "High Confidence"
    elif pct >= 40:
        cls, tag = "confidence-medium", "Medium Confidence"
    else:
        cls, tag = "confidence-low", "Low Confidence"
    return f'<span class="confidence-badge {cls}">{pct:.0f}% · {tag}</span>'


def severity_impact_label(anomaly) -> tuple:
    """Converts the raw modified Z-score into a simplified 0-10 impact
    reading with a plain-English tier, for the exec-facing 'Anomaly
    Severity' card. Raw statistics are kept available separately for the
    tooltip rather than shown here."""
    z = anomaly.modified_z_score
    z_abs = abs(z) if z not in (float("inf"), float("-inf")) else 10.0
    impact_score = min(10.0, z_abs)
    if anomaly.severity == "HIGH" or impact_score >= 7:
        tier = "High Impact"
    elif anomaly.severity == "MEDIUM" or impact_score >= 4:
        tier = "Moderate Impact"
    else:
        tier = "Low Impact"
    return tier, impact_score


def clean_source_label(filename: str) -> str:
    """Maps a raw source filename to an executive-friendly label, e.g.
    'delivery.csv' -> 'Delivery Records'."""
    mapping = {
        "delivery.csv": "Delivery Records",
        "customer_feedback.csv": "Customer Feedback",
        "events.csv": "Calendar Events",
    }
    if filename in mapping:
        return mapping[filename]
    # Fallback for any other/uploaded source: strip extension, title-case.
    base = re.sub(r"\.(csv|xlsx|xls)$", "", filename, flags=re.IGNORECASE)
    return base.replace("_", " ").replace("-", " ").title()


def hypothesis_number_label(hypothesis_id: str, fallback_index: int) -> str:
    """Converts developer-facing hypothesis IDs like 'H1' into a clean
    numeral ('1.') for exec display."""
    m = re.search(r"(\d+)", hypothesis_id or "")
    n = m.group(1) if m else str(fallback_index)
    return f"{n}."


def verdict_pill(verdict: str) -> str:
    cls = {"Validated": "pill-validated", "Inconclusive": "pill-inconclusive", "Refuted": "pill-refuted"}.get(verdict, "pill-inconclusive")
    return f'<span class="pill {cls}">{verdict.upper()}</span>'


def plotly_dark_layout(fig, height=340):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6e9ef", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
    )
    return fig


def build_forecast_chart(historical: pd.DataFrame, forecast: pd.DataFrame, title: str, y_axis_label: str = "Revenue ($)") -> go.Figure:
    """
    Builds a combined historical + forecast Plotly figure: solid line for
    actuals, dashed line for the point forecast, and a shaded confidence
    band (fill between lower/upper) using Plotly's tonexty fill trick.
    The last historical point is prepended to the forecast series so the
    solid cyan line and the dashed purple forecast line join up exactly
    where the shaded confidence interval begins, instead of leaving a gap.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=historical["week"], y=historical["value"], mode="lines+markers",
        name="Historical", line=dict(color="#00e5ff", width=2),
    ))

    last_week = historical["week"].iloc[-1]
    last_value = historical["value"].iloc[-1]
    bridge_x = pd.concat([pd.Series([last_week]), forecast["week"]], ignore_index=True)
    bridge_upper = pd.concat([pd.Series([last_value]), forecast["upper"]], ignore_index=True)
    bridge_lower = pd.concat([pd.Series([last_value]), forecast["lower"]], ignore_index=True)
    bridge_forecast = pd.concat([pd.Series([last_value]), forecast["forecast"]], ignore_index=True)

    # Confidence band: draw upper bound invisibly, then lower bound with
    # fill='tonexty' to shade the area between them. Starting the band at
    # the last historical point anchors the shading exactly at the
    # transition from actuals to projection.
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_upper, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_lower, mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor="rgba(168,85,247,0.18)",
        name="95% Confidence Interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_forecast, mode="lines+markers",
        name="Forecast", line=dict(color="#a855f7", width=2, dash="dash"),
    ))

    fig.update_layout(title=title, xaxis_title="Week", yaxis_title=y_axis_label)
    return fig


# ---------------------------------------------------------------------------
# Interactive Evidence Inspector — proves a cited evidence ID (e.g. "DEL-1792")
# corresponds to a real row in the raw source CSV, not a hallucinated citation.
# ---------------------------------------------------------------------------
RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

# Maps an evidence ID's prefix -> (source CSV filename, friendly label)
EVIDENCE_SOURCE_MAP = {
    "DEL": ("delivery.csv", "Delivery Records"),
    "FB": ("customer_feedback.csv", "Customer Feedback"),
    "EVT": ("events.csv", "Calendar Events"),
}


def resolve_evidence_id(evidence_id: str):
    """Splits 'DEL-1792' into ('DEL', 1792). Returns (None, None) if malformed."""
    prefix, sep, idx_str = evidence_id.partition("-")
    if not sep or not idx_str.isdigit():
        return None, None
    return prefix, int(idx_str)


@st.cache_data(show_spinner=False)
def _load_raw_source(filename: str) -> pd.DataFrame:
    """Loads a raw source CSV fresh from disk (cached per filename)."""
    path = os.path.join(RAW_DIR, filename)
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except (ValueError, KeyError):
        return pd.read_csv(path)


def fetch_evidence_row(evidence_id: str):
    """
    Given a cited evidence ID, resolves which raw CSV it points to and
    returns the exact matching row (by original DataFrame index — the
    same index evidence_retriever.py used to mint the ID in the first
    place, so this is a direct, auditable round-trip back to source data).

    Returns (dataframe_row_as_df, source_filename) or (None, error_message).
    """
    prefix, idx = resolve_evidence_id(evidence_id)
    if prefix is None or prefix not in EVIDENCE_SOURCE_MAP:
        return None, f"Unrecognized evidence ID format: '{evidence_id}'"

    filename, _ = EVIDENCE_SOURCE_MAP[prefix]
    df = _load_raw_source(filename)

    if idx not in df.index:
        return None, f"Row index {idx} not found in {filename} (file may have changed since this citation was generated)."

    return df.loc[[idx]], filename


def render_evidence_inspector(evidence_id: str):
    """
    Renders the looked-up row as a styled dataframe with a glowing
    highlight tint, so the citation is visibly proven against real data.
    """
    row_df, source = fetch_evidence_row(evidence_id)

    if row_df is None:
        st.error(f"{source}")
        return

    st.markdown(
        f"**Evidence `{evidence_id}`** — matched row in `{source}`:"
    )

    def glow_highlight(_):
        return [
            "background-color: rgba(0, 229, 255, 0.16); "
            "border: 1px solid rgba(0, 229, 255, 0.55); "
            "color: #f5f7fa; font-weight: 600;"
        ] * len(row_df.columns)

    styled = row_df.style.apply(glow_highlight, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=False)


def evidence_id_buttons(evidence_ids: list[str], key_prefix: str):
    """
    Renders one small button per evidence ID. Clicking sets the selected
    ID in session state so the inspector panel below can render it.
    """
    if not evidence_ids:
        st.caption("No evidence IDs cited.")
        return
    cols = st.columns(len(evidence_ids))
    for col, eid in zip(cols, evidence_ids):
        if col.button(f"{eid}", key=f"{key_prefix}_{eid}", use_container_width=True):
            st.session_state.selected_evidence_id = eid



st.title("Business Performance Copilot")
st.markdown("Powered by **Local Llama 3 (Ollama)** & Automated Root-Cause Analytics.")

# Stable scroll target for the headline focus area. The flag is set by an
# "Investigate" action and consumed on the following rerun so the scroll
# happens after Streamlit has rendered the updated anomaly panel.
st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)
if st.session_state.get("scroll_to_top", False):
    st.markdown(
        """
        <script>
        (function () {
            function scrollToFocus() {
                const target = window.parent.document.getElementById("top-anchor");
                if (target) {
                    target.scrollIntoView({ behavior: "smooth", block: "start" });
                } else {
                    window.parent.scrollTo({ top: 0, behavior: "smooth" });
                }
            }
            setTimeout(scrollToFocus, 50);
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.scroll_to_top = False

notif_banner("System Ready (Business anomaly engine and AI briefing active)")

# ---------------------------------------------------------------------------
# Session state — keeps the last pipeline run around so the chat assistant
# can answer questions grounded in the actual anomaly/evidence/hypotheses,
# instead of just generic chit-chat.
# ---------------------------------------------------------------------------
if "pipeline_result_growth" not in st.session_state:
    st.session_state.pipeline_result_growth = None
if "scroll_to_top" not in st.session_state:
    st.session_state.scroll_to_top = False
if "pipeline_result_risk" not in st.session_state:
    st.session_state.pipeline_result_risk = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content)
if "selected_evidence_id" not in st.session_state:
    st.session_state.selected_evidence_id = None
if "datasets" not in st.session_state:
    st.session_state.datasets = None
if "all_detected_anomalies" not in st.session_state:
    st.session_state.all_detected_anomalies = []
if "revenue_anomalies" not in st.session_state:
    st.session_state.revenue_anomalies = []
if "custom_anomalies" not in st.session_state:
    st.session_state.custom_anomalies = None
if "custom_weekly" not in st.session_state:
    st.session_state.custom_weekly = None
if "custom_metric_name" not in st.session_state:
    st.session_state.custom_metric_name = None
if "custom_events" not in st.session_state:
    st.session_state.custom_events = []
if "custom_evidence_sources" not in st.session_state:
    st.session_state.custom_evidence_sources = []
if "custom_pipeline_result" not in st.session_state:
    st.session_state.custom_pipeline_result = None

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")


def get_week_context(week: int) -> str:
    """
    Looks up raw weekly-aggregated numbers (sales, delivery, feedback) for
    a specific week across all regions, plus whether that week was flagged
    as anomalous by the detector. Returns a formatted text block to inject
    into the LLM prompt so it can answer about ANY week, not just the one
    top anomaly the pipeline highlighted.
    """
    datasets = st.session_state.get("datasets")
    if datasets is None:
        return f"(No dataset loaded yet — run the pipeline first to enable week {week} lookups.)"

    sales = datasets["sales_weekly"]
    delivery = datasets["delivery_weekly"]

    lines = [f"Requested data for week {week} (all regions):"]

    sales_wk = sales[sales["week"] == week]
    if sales_wk.empty:
        lines.append(f"  No sales data found for week {week} (dataset covers weeks "
                      f"{int(sales['week'].min())}-{int(sales['week'].max())}).")
    else:
        for _, row in sales_wk.iterrows():
            lines.append(
                f"  [{row['region']}] revenue={row['revenue']:.0f}, units_sold={row['units_sold']:.0f}"
            )

    delivery_wk = delivery[delivery["week"] == week]
    for _, row in delivery_wk.iterrows():
        lines.append(
            f"  [{row['region']}] avg_delivery_days={row['avg_delivery_days']:.2f}, "
            f"avg_late_rate={row['avg_late_rate']:.1%}"
        )

    # Was this week flagged anomalous by any metric in the last pipeline run?
    all_anomalies = st.session_state.get("all_detected_anomalies", [])
    matches = [a for a in all_anomalies if a.week == week]
    if matches:
        for a in matches:
            lines.append(
                f"  ANOMALY FLAGGED: {a.metric} in {a.dimension_value} "
                f"(deviation {a.deviation_pct:+.1%}, severity {a.severity})"
            )
    else:
        lines.append("  No anomaly was flagged for this week in the last pipeline run.")

    # --- Real-world event context (holidays, disruptions, promotions) ---
    min_date = datasets.get("min_date")
    events_source = datasets.get("events")
    demo_events = load_events_from_dataframe(events_source) if events_source is not None else []
    all_events = demo_events + st.session_state.get("custom_events", [])
    if all_events and min_date is not None:
        week_events = get_events_for_week(all_events, week, min_date)
        lines.append("")
        lines.append(format_events_for_prompt(week_events))

    return "\n".join(lines)


def build_ollama_system_prompt(user_question: str = "") -> str:
    """
    Builds a system prompt for the local Ollama assistant. If a pipeline
    run is available in session state, the real anomaly/evidence/hypothesis
    data is embedded so answers are grounded instead of generic.
    """
    base = (
        "You are an expert Chief Operations Officer assistant analyzing business "
        "intelligence metrics, supply chain data anomalies, and root causes for "
        "the Scubacats team. Be concise and specific. When the context below "
        "contains real findings, ground your answer in them and cite hypothesis "
        "IDs (H1/H2/H3) or evidence IDs (DEL-/FB-/EVT-) where relevant. If asked "
        "something the context doesn't cover, say so plainly instead of guessing."
    )

    growth_result = st.session_state.pipeline_result_growth
    risk_result = st.session_state.pipeline_result_risk
    if not growth_result and not risk_result:
        return base + "\n\nNo pipeline run has been executed yet — no anomaly data is available."

    def _format_result_block(result, label):
        anomaly = result["anomaly"]
        gen_result = result["gen_result"]
        evaluation = result["evaluation"]
        report = result["report"]

        hyp_lines = []
        for hyp, test in zip(gen_result.hypotheses, evaluation.test_results):
            hyp_lines.append(
                f"  [{hyp.hypothesis_id}] {hyp.title} — Verdict: {test.verdict} "
                f"(score {test.final_score}). {hyp.description} "
                f"Cited evidence: {', '.join(hyp.supporting_evidence_ids) or 'none'}."
            )

        return f"""
{label} finding:
- Anomaly: {anomaly.metric} in {anomaly.dimension_value} ({anomaly.dimension}), week {anomaly.week}
  Actual={anomaly.actual:.1f}, Expected={anomaly.expected:.1f}, Deviation={anomaly.deviation_pct:+.1%}, Severity={anomaly.severity}
- Hypotheses tested:
{chr(10).join(hyp_lines)}
- Executive summary: {report.executive_summary}
- Recommendations: {', '.join(report.key_recommendations)}
"""

    context = "\nCurrent pipeline findings (two separate headlines — a growth opportunity and a risk):\n"
    if growth_result:
        context += _format_result_block(growth_result, "Growth Opportunity")
    if risk_result:
        context += _format_result_block(risk_result, "Risk Anomaly")

    # If the user's question mentions a specific week (e.g. "week 16"),
    # pull that week's real numbers from the full dataset and inject them
    # too — this is what lets the assistant answer beyond the one
    # highlighted anomaly.
    week_match = re.search(r"\bweek\s*(\d+)\b", user_question, re.IGNORECASE)
    if week_match:
        requested_week = int(week_match.group(1))
        context += "\n" + get_week_context(requested_week)

    return base + "\n" + context


def ask_ollama(question: str, week_facts: str = "") -> str:
    system_prompt = build_ollama_system_prompt(user_question=question)
    if week_facts:
        system_prompt += (
            "\n\nIMPORTANT: The exact facts for the requested week are given above "
            "and have already been shown to the user verbatim. Do NOT contradict them "
            "or claim 'no anomalies' if an ANOMALY FLAGGED line is present above — "
            "acknowledge it explicitly and build your analysis on top of it."
        )
    messages = [{"role": "system", "content": system_prompt}]
    # Include recent chat history for continuity (last 4 turns — kept short
    # so an earlier week's context doesn't bleed into / override the
    # current question's facts).
    for role, content in st.session_state.chat_history[-4:]:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Sidebar status
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Business Performance Copilot")
st.sidebar.caption("Executive Insights & Anomaly Tracker")

st.sidebar.markdown("### AI Engine & Model Status")
# Technical execution details (which LLM backend, API connectivity, etc.)
# are intentionally not surfaced here — an executive reader only needs to
# know insights are being generated, not which model/service produced them.
st.sidebar.markdown(
    f"<div class='glow-card glow-info' style='padding:10px 14px;'>"
    f"<div class='glow-sub'>Insight Engine</div><div style='font-weight:700;color:#f5f7fa'>Standalone Mode (Local AI)</div></div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Calibration panel — the closest thing this system has to ground truth.
# Shows whether system verdicts (Validated/Refuted) actually match what
# analysts confirmed in reality, based on accumulated feedback.
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Decision History")
calib = get_calibration_stats()
if calib["total"] == 0:
    st.sidebar.caption(
        "Decision History: 0 Reviewed. Verify or dismiss key findings in the AI Hypothesis "
        "tab to tailor future recommendations."
    )
else:
    with st.sidebar:
        if calib["agreement_rate"] is not None:
            mini_gauge(calib["agreement_rate"] * 100, "Recommendation accuracy", color="#22ffb0", height=130)
        c1, c2 = st.columns(2)
        c1.metric("Verified", calib["confirmed"])
        c2.metric("Dismissed", calib["rejected"])
    st.sidebar.caption(
        f"Decision History: {calib['total']} Reviewed. Every verified or dismissed finding "
        "helps tailor future recommendations to your business."
    )

# ---------------------------------------------------------------------------
# Trained model status — the literal "self-learning" component. Shows
# whether enough analyst feedback exists to train a real classifier yet,
# and if so, how big its training set is and what its train accuracy is.
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Confidence Model")
model_status = get_model_status()
if not model_status["trained"]:
    st.sidebar.caption(
        "Currently follows Standard Trends. Custom scoring will activate after "
        "reviewing 10+ business insights."
    )
else:
    with st.sidebar:
        mini_gauge(model_status["train_accuracy"] * 100, f"Personalized accuracy · {model_status['sample_count']} insights reviewed", color="#00e5ff", height=130)
    st.sidebar.caption(
        "Custom scoring is active and refines itself automatically every time you "
        "verify or dismiss a finding — tailored to your business, not generic trends."
    )

def run_pipeline_for_anomaly(anomaly, revenue_anomalies):
    """
    Runs evidence retrieval -> hypothesis generation -> testing -> reporting
    for ONE specific anomaly, and stores the result under the matching
    "growth" or "risk" headline slot based on the anomaly's nature. Shared
    by the initial pipeline run, week-retargeting, and the Growth/Risk
    "Investigate" buttons, so all paths behave identically.
    """
    evidence = retrieve_evidence_for_anomaly(anomaly)
    gen_result = generate_hypotheses_for_anomaly(evidence)
    evaluation = test_hypotheses(evidence, gen_result)
    report = generate_executive_report(evidence, gen_result, evaluation)

    result = {
        "anomaly": anomaly,
        "evidence": evidence,
        "gen_result": gen_result,
        "evaluation": evaluation,
        "report": report,
        "all_anomalies": revenue_anomalies,
    }

    slot = "pipeline_result_growth" if getattr(anomaly, "nature", "RISK") == "OPPORTUNITY" else "pipeline_result_risk"
    st.session_state[slot] = result
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Anomaly Filters")

# Business-facing labels map to the same underlying sensitivity multiplier
# that the dynamic threshold engine (ai/anomaly_detector.py) already scales
# per-stream — only the presentation changes, not the mechanism.
ALERT_THRESHOLD_OPTIONS = {
    "Conservative (Focus on Critical & High-Priority Anomalies)": 1.5,
    "Balanced (Include Medium & High-Priority Anomalies)": 1.0,
    "Aggressive (Capture Low-Threshold & Early Warning Signals)": 0.7,
}

st.sidebar.markdown("**Alert Threshold**")
selected_alert_label = st.sidebar.selectbox(
    "Alert Threshold",
    options=list(ALERT_THRESHOLD_OPTIONS.keys()),
    index=1,  # defaults to "Balanced"
    label_visibility="collapsed",
    help=(
        "Sets how sensitive anomaly alerts are across every metric. Conservative "
        "surfaces only the most significant issues; Aggressive surfaces earlier, "
        "subtler warning signs at the cost of more alerts. Each stream's bar still "
        "adapts to its own volatility and revenue scale — this choice shifts that "
        "bar up or down."
    ),
)
sensitivity_multiplier = ALERT_THRESHOLD_OPTIONS[selected_alert_label]
st.sidebar.caption(f"Alert Threshold: {selected_alert_label}")

if st.sidebar.button("Run Analysis", use_container_width=True, type="primary"):
    with st.spinner("Analyzing data streams and processing pipeline..."):
        datasets = build_processed_dataset()
        sales_weekly = datasets.get("sales_weekly")

        # Keep the full processed dataset around so the chat assistant can
        # look up ANY week on request, not just the top anomaly below.
        st.session_state.datasets = datasets

        # Detect across revenue/units/delivery (all metrics), not just
        # revenue, so week-specific questions have full anomaly context.
        # The dynamic threshold engine (see ai/anomaly_detector.py) scales
        # each stream's own bar by its volatility and revenue tier; the
        # sensitivity slider above multiplies that per-stream threshold.
        all_detected = detect_all_anomalies(user_sensitivity_multiplier=sensitivity_multiplier)
        st.session_state.all_detected_anomalies = all_detected
        st.session_state.sensitivity_multiplier = sensitivity_multiplier

        revenue_anomalies = detect_anomalies(
            sales_weekly, value_col="revenue", metric_label="Sales", dimension="region",
            user_sensitivity_multiplier=sensitivity_multiplier,
        )
        st.session_state.revenue_anomalies = revenue_anomalies

        # Two separate headlines: the single biggest growth opportunity and
        # the single biggest risk, each across ALL metrics (not just revenue).
        # all_detected is already sorted by |z-score| descending, so the
        # first match in each filtered list is that group's biggest anomaly.
        growth_candidates = [a for a in all_detected if getattr(a, "nature", "RISK") == "OPPORTUNITY"]
        risk_candidates = [a for a in all_detected if getattr(a, "nature", "RISK") != "OPPORTUNITY"]

        if not growth_candidates and not risk_candidates:
            st.session_state.pipeline_result_growth = None
            st.session_state.pipeline_result_risk = None
            st.warning("No anomalies detected in the dataset.")
        else:
            if growth_candidates:
                run_pipeline_for_anomaly(growth_candidates[0], revenue_anomalies)
            else:
                st.session_state.pipeline_result_growth = None
            if risk_candidates:
                run_pipeline_for_anomaly(risk_candidates[0], revenue_anomalies)
            else:
                st.session_state.pipeline_result_risk = None

# ---------------------------------------------------------------------------
# Universal / Bring-Your-Own-Data mode — upload any CSV/Excel file, map its
# columns to the pipeline's required roles, and run the same statistical
# detection engine on it. Fully independent of the built-in demo dataset.
# ---------------------------------------------------------------------------
tab_overview, tab_upload, tab_chat, tab_week_lookup = st.tabs(
    ["Overview", "Upload Your Data", "Assistant", "Week Lookup"]
)

with tab_upload:
    with st.expander("Analyze Your Own Data (upload CSV / Excel)", expanded=False):
        st.caption(
            "Works with any table that has a date/time column and a numeric metric column. "
            "A category column (region, product, store, etc.) is optional but lets the detector "
            "flag anomalies per-category instead of on one combined total."
        )

        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

        with st.expander("Add real-world event context (optional)"):
            st.caption(
                "Upload a small CSV of known real-world events (holidays, promotions, supply-chain "
                "disruptions, weather, etc.) with at least a 'date' and 'description' column, and "
                "optionally 'category' and 'region'. The chat assistant will factor these in when "
                "discussing anomalies that fall near these dates."
            )
            events_file = st.file_uploader("Upload event annotations CSV", type=["csv"], key="events_uploader")
            if events_file is not None:
                try:
                    st.session_state.custom_events = load_events_from_csv(events_file)
                    st.success(f"Loaded {len(st.session_state.custom_events)} event(s).")
                except Exception as e:
                    st.error(f"Couldn't parse event file: {e}")
            if st.session_state.custom_events:
                st.caption(f"{len(st.session_state.custom_events)} custom event(s) currently loaded.")
                if st.button("Clear custom events"):
                    st.session_state.custom_events = []
                    st.rerun()

        with st.expander("Add supporting evidence sources (optional, for full pipeline)"):
            st.caption(
                "Upload up to 3 supporting datasets (e.g. operations/delivery logs, customer feedback, "
                "support tickets) so the pipeline can generate contributing-factor hypotheses for your "
                "own data — matched by time window, same as the built-in demo. Without any of these, "
                "you'll still get anomaly detection, but no hypotheses."
            )
            n_sources = st.number_input("How many evidence source files?", min_value=0, max_value=3, value=len(st.session_state.custom_evidence_sources))
            new_sources = []
            for i in range(int(n_sources)):
                st.markdown(f"**Source {i + 1}**")
                ev_file = st.file_uploader(f"File for source {i + 1}", type=["csv", "xlsx", "xls"], key=f"evidence_file_{i}")
                if ev_file is not None:
                    try:
                        ev_df = load_uploaded_file(ev_file)
                    except ValueError as e:
                        st.error(str(e))
                        ev_df = None
                    if ev_df is not None:
                        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
                        with ecol1:
                            label = st.text_input(f"Label (source {i + 1})", value=f"Source {i + 1}", key=f"ev_label_{i}")
                        with ecol2:
                            ev_date_col = st.selectbox(f"Date column (source {i + 1})", options=list(ev_df.columns), key=f"ev_date_{i}")
                        with ecol3:
                            dim_opts = ["— None —"] + list(ev_df.columns)
                            ev_dim_choice = st.selectbox(f"Dimension column (source {i + 1})", options=dim_opts, key=f"ev_dim_{i}")
                            ev_dim_col = None if ev_dim_choice == "— None —" else ev_dim_choice
                        with ecol4:
                            id_prefix = st.text_input(f"ID prefix (source {i + 1})", value=f"EV{i + 1}", key=f"ev_prefix_{i}", max_chars=6)
                        new_sources.append(GenericEvidenceSource(
                            label=label, id_prefix=id_prefix, df=ev_df,
                            date_col=ev_date_col, dimension_col=ev_dim_col,
                        ))
            if st.button("Save evidence sources"):
                st.session_state.custom_evidence_sources = new_sources
                st.success(f"Saved {len(new_sources)} evidence source(s).")

        if uploaded_file is not None:
            try:
                user_df = load_uploaded_file(uploaded_file)
            except ValueError as e:
                st.error(str(e))
                user_df = None

            if user_df is not None:
                st.write(f"Preview ({len(user_df)} rows):")
                st.dataframe(user_df.head(10), use_container_width=True)

                suggestions = suggest_column_roles(user_df)
                columns = list(user_df.columns)
                no_dimension_option = "— None (analyze as one overall series) —"

                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    date_col = st.selectbox(
                        "Date / time column",
                        options=columns,
                        index=columns.index(suggestions["date_col"]) if suggestions["date_col"] in columns else 0,
                    )
                with mcol2:
                    value_col = st.selectbox(
                        "Target metric column (what to analyze)",
                        options=columns,
                        index=columns.index(suggestions["value_col"]) if suggestions["value_col"] in columns else 0,
                    )
                with mcol3:
                    dim_options = [no_dimension_option] + columns
                    default_dim = suggestions["dimension_col"] if suggestions["dimension_col"] in columns else no_dimension_option
                    dimension_choice = st.selectbox(
                        "Category / dimension column (optional)",
                        options=dim_options,
                        index=dim_options.index(default_dim),
                    )
                    dimension_col = None if dimension_choice == no_dimension_option else dimension_choice

                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    direction = st.radio(
                        "For this metric, is HIGHER better or worse?",
                        options=["Higher is better (e.g. revenue, sales)", "Lower is better (e.g. delays, costs, complaints)"],
                        horizontal=False,
                    )
                    lower_is_better = direction.startswith("Lower")
                with dcol2:
                    aggregation = st.radio(
                        "How should multiple rows in the same week be combined?",
                        options=["Sum (totals, e.g. revenue, units)", "Average (rates, e.g. ratings, days)"],
                        horizontal=False,
                    )
                    agg_mode = "mean" if aggregation.startswith("Average") else "sum"

                mapping = ColumnMapping(
                    date_col=date_col, value_col=value_col, dimension_col=dimension_col,
                    lower_is_better=lower_is_better, aggregation=agg_mode,
                )

                issues = validate_mapping(user_df, mapping)
                for issue in issues:
                    st.warning(f"{issue}")

                if st.button("Run Anomaly Detection on My Data", type="primary"):
                    try:
                        with st.spinner("Aggregating and analyzing your data..."):
                            weekly = build_weekly_series(user_df, mapping)
                            metric_display_name = value_col
                            detected = detect_anomalies(
                                weekly, value_col="value", metric_label=metric_display_name,
                                dimension="dimension_value", lower_is_better=mapping.lower_is_better,
                                user_sensitivity_multiplier=sensitivity_multiplier,
                            )
                        st.session_state.custom_weekly = weekly
                        st.session_state.custom_anomalies = detected
                        st.session_state.custom_metric_name = metric_display_name
                    except ValueError as e:
                        st.error(str(e))

        # --- Results for the uploaded dataset (separate from the demo pipeline results) ---
        if st.session_state.get("custom_anomalies") is not None:
            detected = st.session_state.custom_anomalies
            weekly = st.session_state.custom_weekly
            metric_name = st.session_state.custom_metric_name

            st.markdown("---")
            st.subheader(f"Results — {metric_name}")

            if not detected:
                st.info("No statistically significant anomalies were found in this data with the current settings.")
            else:
                opp_count = sum(1 for a in detected if a.nature == "OPPORTUNITY")
                risk_count = len(detected) - opp_count
                rcol1, rcol2, rcol3 = st.columns(3)
                with rcol1:
                    glow_card("Anomalies Found", str(len(detected)), f"across {weekly['dimension_value'].nunique()} categor(y/ies)", "info")
                with rcol2:
                    glow_card("Growth Opportunities", str(opp_count), "positive deviations", "good")
                with rcol3:
                    glow_card("Risks Flagged", str(risk_count), "negative deviations", "high" if risk_count else "medium")

                labels = [f"{a.dimension_value} · W{a.week}" for a in detected]
                zscores = [abs(a.modified_z_score) if a.modified_z_score not in (float("inf"), float("-inf")) else 10 for a in detected]
                colors = ["#22ffb0" if a.nature == "OPPORTUNITY" else "#ff4d6d" for a in detected]
                bar = go.Figure(go.Bar(
                    x=zscores, y=labels, orientation="h", marker=dict(color=colors),
                    text=[f"{a.deviation_pct:+.1%}" for a in detected], textposition="outside",
                ))
                bar.update_layout(title="Detected Anomalies — |Modified Z-Score|", yaxis=dict(autorange="reversed"))
                st.plotly_chart(plotly_dark_layout(bar, height=max(280, 40 * len(detected))), use_container_width=True, key="custom_anomaly_bar")

                with st.expander("View anomaly details"):
                    for a in sorted(detected, key=lambda x: x.week):
                        tag = "Opportunity" if a.nature == "OPPORTUNITY" else "Risk"
                        st.markdown(
                            f"**{tag} — {a.dimension_value}, Week {a.week}**: "
                            f"{a.actual:,.2f} vs expected {a.expected:,.2f} "
                            f"({a.deviation_pct:+.1%}, |Z|={abs(a.modified_z_score):.2f}, severity {a.severity})"
                        )

                with st.expander("Forecast a category forward"):
                    categories = sorted(weekly["dimension_value"].unique().tolist())
                    chosen_cat = st.selectbox("Category to forecast", options=categories, key="custom_forecast_cat")
                    horizon2 = st.slider("Weeks to forecast ahead", min_value=2, max_value=12, value=4, key="custom_forecast_horizon")
                    if st.button("Generate Forecast", key="custom_forecast_btn"):
                        cat_series = weekly[weekly["dimension_value"] == chosen_cat][["week", "value"]].sort_values("week")
                        try:
                            forecast_df, backend = generate_forecast(cat_series, horizon2, min_date=None)
                            fc_fig = build_forecast_chart(cat_series, forecast_df, title=f"{chosen_cat} — Historical + Forecast")
                            st.plotly_chart(plotly_dark_layout(fc_fig, height=380), use_container_width=True, key="custom_forecast_chart")
                            st.caption("Model: Prophet Predictive Pipeline | Shaded area represents 95% Confidence Interval widening over time.")
                        except ValueError as e:
                            st.warning(str(e))

                # --- Full pipeline: hypotheses + testing + report, on YOUR data ---
                st.markdown("---")
                st.subheader("Run Full Root-Cause Pipeline on a Detected Anomaly")
                anomaly_labels = [f"Week {a.week} · {a.dimension_value} · {a.deviation_pct:+.1%}" for a in detected]
                chosen_idx = st.selectbox(
                    "Choose an anomaly to investigate", options=range(len(detected)),
                    format_func=lambda i: anomaly_labels[i], key="custom_anomaly_picker",
                )
                if not st.session_state.custom_evidence_sources:
                    st.info("No evidence sources uploaded above — the pipeline will still run, but will report 'Unexplained Statistical Deviation' since there's nothing to correlate against.")

                if st.button("Run Full Pipeline", type="primary", key="custom_full_pipeline_btn"):
                    target_anomaly = detected[chosen_idx]
                    min_date_custom = user_df[date_col].min() if uploaded_file is not None else pd.Timestamp.now()
                    with st.spinner("Retrieving evidence and generating hypotheses..."):
                        ev_package = retrieve_generic_evidence(
                            target_anomaly, st.session_state.custom_evidence_sources, min_date=pd.to_datetime(min_date_custom),
                        )
                        gen_result = generate_generic_hypotheses(ev_package)
                        evaluation = test_hypotheses(ev_package, gen_result)
                        report = generate_executive_report(ev_package, gen_result, evaluation)

                    st.session_state.custom_pipeline_result = {
                        "anomaly": target_anomaly, "evidence": ev_package,
                        "gen_result": gen_result, "evaluation": evaluation, "report": report,
                    }

                cpr = st.session_state.custom_pipeline_result
                if cpr:
                    a = cpr["anomaly"]
                    st.markdown(f"#### Results for Week {a.week} · {a.dimension_value}")
                    is_opp = getattr(a, "nature", "RISK") == "OPPORTUNITY"
                    st.caption("Correlated factors based on timeline. Validate with regional managers to confirm root cause.")
                    for idx, h in enumerate(cpr["gen_result"].hypotheses, start=1):
                        test_result = next((t for t in cpr["evaluation"].test_results if t.hypothesis_id == h.hypothesis_id), None)
                        num_label = hypothesis_number_label(h.hypothesis_id, idx)
                        with st.expander(f"{num_label} {h.title}" + (f" — {verdict_pill(test_result.verdict)}" if test_result else ""), expanded=False):
                            st.write(h.description)
                            st.markdown(confidence_badge(h.confidence_score), unsafe_allow_html=True)
                            st.caption(h.evidence_basis)
                            if h.confidence_adjustment != 0:
                                st.success(h.learning_note)
                            elif h.learning_sample_count > 0:
                                st.caption(h.learning_note)
                            if test_result:
                                st.markdown(f"**Verdict:** {verdict_pill(test_result.verdict)} &nbsp; Score: **{test_result.final_score}**", unsafe_allow_html=True)
                                st.caption(test_result.rationale)
                            if h.supporting_evidence_ids:
                                st.markdown("**Cited evidence:**")
                                for eid in h.supporting_evidence_ids:
                                    item = next((it for it in cpr["evidence"].all_records if it.evidence_id == eid), None)
                                    st.markdown(f"- `{eid}`: " + (item.summary if item else "(details unavailable)"))

                            prior_fb = get_feedback(a, h.hypothesis_id)
                            if prior_fb:
                                tag = "Confirmed" if prior_fb["analyst_verdict"] == CONFIRMED else "Rejected"
                                st.info(f"**Analyst feedback on record:** {tag}")

                            note_key = f"custom_note_{h.hypothesis_id}_{a.week}"
                            note = st.text_input("Optional note", key=note_key, label_visibility="collapsed", placeholder="Optional note")
                            fcol1, fcol2 = st.columns(2)
                            with fcol1:
                                if st.button("Confirm", key=f"custom_confirm_{h.hypothesis_id}_{a.week}", use_container_width=True):
                                    record_feedback(a, h,
                                                     system_verdict=test_result.verdict if test_result else "Unknown",
                                                     system_score=test_result.final_score if test_result else h.confidence_score,
                                                     analyst_verdict=CONFIRMED, note=note)
                                    with st.spinner("Retraining confidence model on updated feedback..."):
                                        train_model()
                                    st.rerun()
                            with fcol2:
                                if st.button("Reject", key=f"custom_reject_{h.hypothesis_id}_{a.week}", use_container_width=True):
                                    record_feedback(a, h,
                                                     system_verdict=test_result.verdict if test_result else "Unknown",
                                                     system_score=test_result.final_score if test_result else h.confidence_score,
                                                     analyst_verdict=REJECTED, note=note)
                                    with st.spinner("Retraining confidence model on updated feedback..."):
                                        train_model()
                                    st.rerun()

                    st.markdown("##### Executive Summary")
                    st.caption("Primary takeaway based on multi-source data synthesis.")
                    st.write(cpr["report"].executive_summary)
                    for rec in cpr["report"].key_recommendations:
                        st.markdown(f"- **Action Item:** {rec}")

        st.caption(
            "Note: hypothesis generation and evidence citation currently require the curated "
            "delivery/feedback/events CSVs from the demo dataset. Uploaded-data mode runs the "
            "same statistical detection engine, but root-cause hypotheses aren't generated for it yet."
        )

with tab_overview:

    def render_headline_section(result, key_prefix):
        """Renders one full headline drill-down (banner, KPI cards, gauge,
        trend, forecast, hypothesis tabs) for a single pipeline result.
        Called once for the biggest growth opportunity and once for the
        biggest risk anomaly, each with its own widget-key namespace so
        the two sections never collide."""
        anomaly = result["anomaly"]
        gen_result = result["gen_result"]
        evaluation = result["evaluation"]
        report = result["report"]
        all_anomalies = result["all_anomalies"]
        sales_all = st.session_state.datasets["sales_weekly"] if st.session_state.datasets is not None else None

        is_opportunity = getattr(anomaly, "nature", "RISK") == "OPPORTUNITY"

        if is_opportunity:
            notif_banner(
                f"Growth Opportunity Detected: <b>{anomaly.metric}</b> in <b>{anomaly.dimension_value}</b> "
                f"(Week {anomaly.week}) — {anomaly.deviation_pct:+.1%} vs. expected. Investigating what drove it, so it can be replicated.",
                danger=False,
            )
        else:
            notif_banner(
                f"Detected Anomaly: <b>{anomaly.metric}</b> in <b>{anomaly.dimension_value}</b> "
                f"(Week {anomaly.week}) — {anomaly.deviation_pct:+.1%} deviation, severity {anomaly.severity}",
                danger=(anomaly.severity == "HIGH"),
            )

        # --- Glowing KPI card row ---
        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            card_variant = "good" if is_opportunity else ("high" if anomaly.severity == "HIGH" else "medium")
            glow_card("Anomaly Metric", anomaly.metric, f"{anomaly.dimension_value} · Week {anomaly.week}", card_variant)
        with kc2:
            glow_card("Deviation", f"{anomaly.deviation_pct:+.1%}", f"vs expected {anomaly.expected:,.0f}", "purple")
        with kc3:
            impact_tier, impact_score = severity_impact_label(anomaly)
            raw_stats_tooltip = (
                f"Modified Z-Score: {anomaly.modified_z_score:.2f} | "
                f"Dynamic threshold ≥ {anomaly.threshold_used:.2f} | "
                f"Coefficient of Variation: {anomaly.threshold_cv:.2f} | "
                f"Volatility tier: {anomaly.threshold_tier}"
            )
            st.markdown(f"""
            <div class="glow-card glow-info">
                <div class="glow-label">Anomaly Severity <span class="stat-tooltip" title="{raw_stats_tooltip}">i</span></div>
                <div class="glow-value">{impact_tier}</div>
                <div class="glow-sub">{impact_score:.1f} out of 10 impact score</div>
            </div>
            """, unsafe_allow_html=True)
        with kc4:
            nature_label = "Growth Driver" if is_opportunity else "Risk"
            glow_card("Nature", nature_label, "opportunity vs. risk classification", "good" if is_opportunity else "medium")

        label = "biggest growth opportunity" if is_opportunity else "biggest risk"
        st.caption(f"This is the {label} — {anomaly.metric} in {anomaly.dimension_value}, Week {anomaly.week}. Use 'Investigate' in the list below to switch to a different anomaly.")

        # --- Geometric visual: anomaly score gauge + weekly trend ---
        gcol1, gcol2 = st.columns([1, 2])
        with gcol1:
            # Target line marks the "expected volume" threshold (60%) —
            # readings above it represent a meaningful deviation from
            # normal business-as-usual performance.
            expected_volume_threshold = 60
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=anomaly.anomaly_score * 100,
                number={"suffix": "%", "font": {"size": 34}},
                title={"text": "Deviation from Expected Volume"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ec4899"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(34,255,176,0.25)"},
                        {"range": [40, 75], "color": "rgba(255,157,61,0.25)"},
                        {"range": [75, 100], "color": "rgba(255,77,109,0.3)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f5f7fa", "width": 3},
                        "thickness": 0.85,
                        "value": expected_volume_threshold,
                    },
                },
            ))
            gauge.update_layout(
                annotations=[dict(
                    text="⬥ Expected-volume target line · Bars beyond it flag a significant deviation",
                    x=0.5, y=-0.06, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=11, color="#9aa4b2"),
                )],
                margin=dict(b=40),
            )
            st.plotly_chart(plotly_dark_layout(gauge, height=300), use_container_width=True, key=f"{key_prefix}_anomaly_gauge")
        with gcol2:
            region_series = sales_all[sales_all["region"] == anomaly.dimension_value].sort_values("week") if sales_all is not None else None
            if region_series is not None and not region_series.empty:
                trend = px.line(
                    region_series, x="week", y="revenue", markers=True,
                    title=f"Revenue Trend — {anomaly.dimension_value}",
                    color_discrete_sequence=["#00e5ff"],
                )
                trend.add_scatter(
                    x=[anomaly.week], y=[anomaly.actual], mode="markers",
                    marker=dict(
                        size=20,
                        color="#22ffb0" if is_opportunity else "#ff4d6d",
                        symbol="star",
                        line=dict(width=1, color="#ffffff"),
                    ),
                    name=f"Week {anomaly.week} (studied)",
                    hovertemplate=(
                        f"Week {anomaly.week}<br>Deviation: {anomaly.deviation_pct:+.1%}<br>"
                        f"|Z|={abs(anomaly.modified_z_score):.2f}<extra></extra>"
                    ),
                )
                st.plotly_chart(plotly_dark_layout(trend), use_container_width=True, key=f"{key_prefix}_revenue_trend")

                # --- Forecast panel ---
                with st.expander(f"Forecast future {anomaly.dimension_value} revenue"):
                    horizon = st.slider("Weeks to forecast ahead", min_value=2, max_value=12, value=4, key=f"{key_prefix}_forecast_horizon")
                    if st.button("Generate Forecast", key=f"{key_prefix}_forecast_btn"):
                        fc_series = region_series[["week", "revenue"]].rename(columns={"revenue": "value"})
                        try:
                            min_date = st.session_state.datasets.get("min_date") if st.session_state.datasets else None
                            forecast_df, backend = generate_forecast(fc_series, horizon, min_date=min_date)
                            fc_fig = build_forecast_chart(
                                fc_series, forecast_df,
                                title=f"{anomaly.dimension_value} Revenue — Historical + Forecast",
                                y_axis_label="Revenue ($)",
                            )
                            st.plotly_chart(plotly_dark_layout(fc_fig, height=380), use_container_width=True, key=f"{key_prefix}_forecast_chart")
                            st.caption("Model: Prophet Predictive Pipeline | Shaded area represents 95% Confidence Interval widening over time.")
                        except ValueError as e:
                            st.warning(str(e))
        if len(all_anomalies) > 1:
            st.caption(f"{len(all_anomalies) - 1} additional anomal{'y' if len(all_anomalies) == 2 else 'ies'} detected in the Sales/revenue metric alone.")

        tab1, tab2, tab3 = st.tabs(["AI Hypotheses", "Validation & Evidence", "Executive Briefing"])

        with tab1:
            header = "Growth-Driver Candidates (What Went Right)" if is_opportunity else "Contributing-Factor Candidates (Time-Correlated)"
            st.subheader(header)
            st.caption("Correlated factors based on timeline. Validate with regional managers to confirm root cause.")
            st.write(gen_result.anomaly_summary)

            for idx, h in enumerate(gen_result.hypotheses, start=1):
                # Find the matching test result for verdict/score used in feedback logging
                test_result = next((t for t in evaluation.test_results if t.hypothesis_id == h.hypothesis_id), None)
                num_label = hypothesis_number_label(h.hypothesis_id, idx)

                with st.expander(f"{num_label} {h.title}"):
                    st.write(h.description)
                    st.markdown(confidence_badge(h.confidence_score), unsafe_allow_html=True)
                    st.caption(h.evidence_basis)
                    if h.confidence_adjustment != 0:
                        st.success(h.learning_note)
                    elif h.learning_sample_count > 0:
                        st.caption(h.learning_note)
                    st.markdown("**Cited Evidence (co-occurring in time) — click an ID to inspect the exact raw row:**")
                    evidence_id_buttons(h.supporting_evidence_ids, key_prefix=f"{key_prefix}_{h.hypothesis_id}")

                    st.markdown("---")
                    prior_feedback = get_feedback(anomaly, h.hypothesis_id)
                    if prior_feedback:
                        tag = "Confirmed" if prior_feedback["analyst_verdict"] == CONFIRMED else "Rejected"
                        st.info(f"**Analyst feedback on record:** {tag}" + (f" — _{prior_feedback['note']}_" if prior_feedback.get("note") else ""))

                    st.markdown("**Analyst review:** did you verify this against what actually happened?")
                    note_key = f"{key_prefix}_note_{h.hypothesis_id}_{anomaly.week}"
                    note = st.text_input("Optional note (what you found)", key=note_key, label_visibility="collapsed", placeholder="Optional note (what you found)")
                    fb_col1, fb_col2 = st.columns(2)
                    with fb_col1:
                        if st.button("Confirm this factor", key=f"{key_prefix}_confirm_{h.hypothesis_id}_{anomaly.week}", use_container_width=True):
                            record_feedback(
                                anomaly, h,
                                system_verdict=test_result.verdict if test_result else "Unknown",
                                system_score=test_result.final_score if test_result else h.confidence_score,
                                analyst_verdict=CONFIRMED, note=note,
                            )
                            with st.spinner("Retraining confidence model on updated feedback..."):
                                train_model()
                            st.rerun()
                    with fb_col2:
                        if st.button("Reject this factor", key=f"{key_prefix}_reject_{h.hypothesis_id}_{anomaly.week}", use_container_width=True):
                            record_feedback(
                                anomaly, h,
                                system_verdict=test_result.verdict if test_result else "Unknown",
                                system_score=test_result.final_score if test_result else h.confidence_score,
                                analyst_verdict=REJECTED, note=note,
                            )
                            with st.spinner("Retraining confidence model on updated feedback..."):
                                train_model()
                            st.rerun()

            if st.session_state.selected_evidence_id:
                st.markdown("---")
                render_evidence_inspector(st.session_state.selected_evidence_id)
                if st.button("Close inspector", key=f"{key_prefix}_close_inspector"):
                    st.session_state.selected_evidence_id = None
                    st.rerun()

        with tab2:
            st.subheader("Hypothesis Testing & Scoring")
            for idx, test in enumerate(evaluation.test_results, start=1):
                num_label = hypothesis_number_label(test.hypothesis_id, idx)
                st.markdown(
                    f"**{num_label} {test.title}** — {verdict_pill(test.verdict)} &nbsp; {score_badge(test.final_score)}",
                    unsafe_allow_html=True,
                )
                st.caption(test.rationale)

            st.markdown("---")
            st.subheader("Supporting Data & Audit Trail")
            ev = result["evidence"]
            ecol1, ecol2 = st.columns([1, 1])
            with ecol1:
                glow_card("Delivery Records", str(len(ev.delivery_records)), "Source: Delivery Records", "info")
                glow_card("Feedback Records", str(len(ev.feedback_records)), "Source: Customer Feedback", "purple")
                glow_card("Event Records", str(len(ev.event_records)), "Source: Calendar Events", "medium")
            with ecol2:
                counts = {
                    "Delivery": len(ev.delivery_records),
                    "Feedback": len(ev.feedback_records),
                    "Events": len(ev.event_records),
                }
                if sum(counts.values()) > 0:
                    donut = px.pie(
                        names=list(counts.keys()), values=list(counts.values()),
                        hole=0.55, title="Evidence Mix",
                        color_discrete_sequence=["#00e5ff", "#a855f7", "#ff9d3d"],
                    )
                    st.plotly_chart(plotly_dark_layout(donut, height=300), use_container_width=True, key=f"{key_prefix}_evidence_donut")
            with st.expander("View raw evidence details"):
                for item in ev.delivery_records + ev.feedback_records + ev.event_records:
                    st.markdown(f"**{item.evidence_id}** (Source: {clean_source_label(item.source_file)}) — {item.summary}")

        with tab3:
            st.subheader("Executive Summary & Action Plan")
            st.caption("Primary takeaway based on multi-source data synthesis.")
            st.write(report.executive_summary)

            # Surface the strongest-scoring, validated hypothesis as an
            # explicit "Primary Driver" callout rather than leaving it
            # buried in a list of equally-weighted candidates.
            validated = [t for t in evaluation.test_results if t.verdict == "Validated"]
            top_test = max(validated, key=lambda t: t.final_score, default=None) or \
                max(evaluation.test_results, key=lambda t: t.final_score, default=None)
            if top_test:
                driver_verb = "directly correlated with" if is_opportunity else "was the leading contributing factor behind"
                st.markdown(
                    f"**Primary Driver:** {top_test.title} {driver_verb} the "
                    f"{anomaly.deviation_pct:+.1%} change in {anomaly.metric} for {anomaly.dimension_value}."
                )

            st.markdown("### Key Recommendations")
            for rec in report.key_recommendations:
                st.markdown(f"- **Action Item:** {rec}")

    growth_result = st.session_state.pipeline_result_growth
    risk_result = st.session_state.pipeline_result_risk

    if not growth_result and not risk_result:
        st.info("Click **Run Analysis** in the sidebar to get started, or ask the local AI assistant a question below.")
    else:
        if growth_result:
            st.markdown("## Growth Opportunity")
            render_headline_section(growth_result, key_prefix="growth")

        if risk_result:
            st.markdown("---")
            st.markdown("## Risk Anomaly")
            render_headline_section(risk_result, key_prefix="risk")

        _headline_anomalies = [r["anomaly"] for r in (growth_result, risk_result) if r]

        # --- Growth vs Risk split — every detected anomaly, grouped by nature.
        # The list is executive-facing: clean metric headers, business urgency
        # labels, aligned actions, and a stable scroll target back to the focus panel. ---
        st.markdown("---")
        st.markdown("### Growth Opportunities & Risk Anomalies")

        PRIORITY_TIERS = {
            "HIGH": ("Critical Alert", "#ff4d6d"),
            "MEDIUM": ("High Priority", "#ff9d3d"),
            "LOW": ("Monitor", "#9aa4b2"),
        }

        all_metric_anomalies_split = st.session_state.get("all_detected_anomalies", [])
        growth_list = [
            a for a in all_metric_anomalies_split
            if getattr(a, "nature", "RISK") == "OPPORTUNITY"
        ]
        risk_list = [
            a for a in all_metric_anomalies_split
            if getattr(a, "nature", "RISK") != "OPPORTUNITY"
        ]

        gcol_growth, gcol_risk = st.columns(2)

        def _standardized_header(a) -> str:
            """Return '[Region] Region: [Metric Name] (W[Week])'."""
            dimension = str(a.dimension_value).strip()
            metric_clean = re.sub(r"[_\-]+", " ", str(a.metric)).strip()

            # Remove region/week fragments if the detector's metric label already
            # contains them, so we never display duplicates such as
            # 'North Region: Avg Delivery Days North Week 16 (W16)'.
            metric_clean = re.sub(
                rf"\s*[-–—]?\s*{re.escape(dimension)}\s*[-–—]?\s*week\s*{int(a.week)}\s*$",
                "",
                metric_clean,
                flags=re.IGNORECASE,
            )
            metric_clean = re.sub(
                rf"\s*[-–—]?\s*week\s*{int(a.week)}\s*$",
                "",
                metric_clean,
                flags=re.IGNORECASE,
            )
            metric_clean = re.sub(
                rf"\s*[-–—]?\s*{re.escape(dimension)}\s*$",
                "",
                metric_clean,
                flags=re.IGNORECASE,
            )

            metric_clean = re.sub(r"\s{2,}", " ", metric_clean).strip(" -–—")
            metric_clean = metric_clean.title()

            return f"**{dimension} Region: {metric_clean} (W{int(a.week)})**"

        def _render_anomaly_row(a, key_prefix):
            row_l, row_r = st.columns([4, 1], vertical_alignment="center")

            with row_l:
                tier_label, tier_color = PRIORITY_TIERS.get(
                    str(getattr(a, "severity", "LOW")).upper(),
                    ("Monitor", "#9aa4b2"),
                )
                st.markdown(_standardized_header(a))
                st.markdown(
                    f"<span style='color:{tier_color};font-weight:700;font-size:0.85rem'>"
                    f"{a.deviation_pct:+.1%} Deviation · {tier_label}</span>",
                    unsafe_allow_html=True,
                )

            with row_r:
                is_current = any(
                    a.metric == h.metric
                    and a.dimension_value == h.dimension_value
                    and a.week == h.week
                    for h in _headline_anomalies
                )

                if is_current:
                    # Keep the active state the same visual height as an
                    # Investigate button so both columns remain aligned.
                    st.markdown(
                        """
                        <div style="
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            min-height:38.4px;
                            box-sizing:border-box;
                            padding:0 8px;
                            border-radius:8px;
                            background:rgba(0,229,255,0.12);
                            border:1px solid rgba(0,229,255,0.45);
                            color:#00e5ff;
                            font-weight:700;
                            font-size:0.82rem;
                            letter-spacing:0.3px;
                            text-align:center;
                            line-height:1.15;
                        ">Currently Viewing</div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif st.button(
                    "Investigate",
                    key=f"{key_prefix}_{a.metric}_{a.dimension_value}_{a.week}",
                    use_container_width=True,
                ):
                    with st.spinner(f"Running root-cause analysis for week {a.week}..."):
                        run_pipeline_for_anomaly(
                            a, st.session_state.revenue_anomalies
                        )

                    # The script is rendered on the next run, after Streamlit
                    # has rebuilt the updated headline/focus panel.
                    st.session_state.scroll_to_top = True
                    st.rerun()

        with gcol_growth:
            st.markdown(
                f"<div class='glow-card glow-good' style='margin-bottom:10px;'>"
                f"<div class='glow-label'>Growth Opportunities</div>"
                f"<div class='glow-value'>{len(growth_list)}</div>"
                f"<div class='glow-sub'>positive deviations across all metrics</div></div>",
                unsafe_allow_html=True,
            )
            if not growth_list:
                st.caption("No growth opportunities detected in the last run.")
            else:
                for a in sorted(growth_list, key=lambda x: abs(x.modified_z_score) if x.modified_z_score not in (float("inf"), float("-inf")) else 1e9, reverse=True):
                    _render_anomaly_row(a, "growth_row")

        with gcol_risk:
            st.markdown(
                f"<div class='glow-card glow-high' style='margin-bottom:10px;'>"
                f"<div class='glow-label'>Risk Anomalies</div>"
                f"<div class='glow-value'>{len(risk_list)}</div>"
                f"<div class='glow-sub'>negative deviations across all metrics</div></div>",
                unsafe_allow_html=True,
            )
            if not risk_list:
                st.caption("No risk anomalies detected in the last run.")
            else:
                for a in sorted(risk_list, key=lambda x: abs(x.modified_z_score) if x.modified_z_score not in (float("inf"), float("-inf")) else 1e9, reverse=True):
                    _render_anomaly_row(a, "risk_row")

        # --- View all anomalies — the detailed bar chart, now the final
        # item in the tab, after both headline drill-downs and the
        # growth/risk overview above it. ---
        all_metric_anomalies = st.session_state.get("all_detected_anomalies", [])
        if len(all_metric_anomalies) > 1:
            with st.expander(f"View all {len(all_metric_anomalies)} anomalies detected across every metric (revenue, units sold, delivery)"):
                labels = [f"{a.metric} · {a.dimension_value} · W{a.week}" for a in all_metric_anomalies]
                zscores = [abs(a.modified_z_score) if a.modified_z_score not in (float("inf"), float("-inf")) else 10 for a in all_metric_anomalies]
                colors = ["#ff4d6d" if a.severity == "HIGH" else "#ff9d3d" for a in all_metric_anomalies]
                bar = go.Figure(go.Bar(
                    x=zscores, y=labels, orientation="h",
                    marker=dict(color=colors),
                    text=[f"{a.deviation_pct:+.1%}" for a in all_metric_anomalies],
                    textposition="outside",
                ))
                bar.update_layout(title="All Detected Anomalies — |Modified Z-Score|", yaxis=dict(autorange="reversed"))
                st.plotly_chart(plotly_dark_layout(bar, height=max(280, 40 * len(all_metric_anomalies))), use_container_width=True, key="all_metric_bar")

with tab_week_lookup:
    st.subheader("Week Lookup")
    st.caption(
        "Look up the exact recorded data and anomaly context for any week."
    )

    if st.session_state.datasets is not None:
        sales_all = st.session_state.datasets["sales_weekly"]

        min_week = int(sales_all["week"].min())
        max_week = int(sales_all["week"].max())

        chosen_week = st.number_input(
            f"Week number ({min_week}-{max_week}):",
            min_value=min_week,
            max_value=max_week,
            value=min_week,
            step=1,
            key="week_lookup_tab",
        )

        if st.button(
            "Show Week Data",
            use_container_width=True,
            type="primary",
            key="week_lookup_show_btn",
        ):
            st.markdown(f"### Week {int(chosen_week)}")
            st.code(
                get_week_context(int(chosen_week)),
                language=None
            )
    else:
        st.info("Run Analysis first to enable week-level data lookup.")


with tab_chat:
    st.subheader("Executive AI Assistant")

    growth_result = st.session_state.pipeline_result_growth
    risk_result = st.session_state.pipeline_result_risk

    if growth_result or risk_result:
        st.caption("Context-aware intelligence powered by synthesized root-cause data and operational evidence.")
        suggested_questions = []
        if growth_result:
            ga = growth_result["anomaly"]
            suggested_questions.append(f"Why did {ga.metric} grow in {ga.dimension_value}?")
        if risk_result:
            ra = risk_result["anomaly"]
            suggested_questions.append(f"Why did the {ra.metric} anomaly happen in {ra.dimension_value}?")
        suggested_questions += [
            "Which hypothesis has the strongest evidence, and why?",
            "What should leadership do first based on this analysis?",
        ]
    else:
        st.caption("Run the pipeline first for grounded answers, or ask a general question now.")
        suggested_questions = [
            "What kinds of anomalies can this system detect?",
            "How does the root-cause hypothesis testing work?",
            "What data sources feed into this analysis?",
        ]

    st.markdown("**Suggested questions:**")
    cols = st.columns(len(suggested_questions))
    for col, q in zip(cols, suggested_questions):
        if col.button(q, use_container_width=True):
            # Write directly into the text_input's own session-state key rather
            # than passing value= below — mixing value= and key= on a widget
            # that has already been rendered once causes Streamlit to ignore
            # the new value, so clicking a suggestion silently did nothing.
            st.session_state.user_query_input = q

    user_query = st.text_input(
        "Ask a custom business question:",
        key="user_query_input",
    )

    ask_col, clear_col = st.columns([1, 1])
    ask_clicked = ask_col.button("Ask Copilot", type="primary")
    clear_clicked = clear_col.button("Clear chat")

    if clear_clicked:
        st.session_state.chat_history = []
        # Can't assign to st.session_state.user_query_input directly here -
        # the text_input widget above has already been instantiated with this
        # key for the current script run, and Streamlit forbids overwriting a
        # widget-bound key post-instantiation (StreamlitAPIException). Deleting
        # the key + rerunning is the supported way to clear it: on the next
        # run the widget hasn't been created yet, so it reinitializes empty.
        if "user_query_input" in st.session_state:
            del st.session_state.user_query_input
        st.rerun()

    if ask_clicked and user_query:
        with st.spinner(f"{OLLAMA_MODEL} is analyzing locally..."):
            try:
                # If the question mentions a specific week, resolve and show the
                # exact facts FIRST, deterministically — the LLM only adds
                # interpretation on top and is instructed not to contradict them.
                week_match = re.search(r"\bweek\s*(\d+)\b", user_query, re.IGNORECASE)
                week_facts = ""
                if week_match and st.session_state.datasets is not None:
                    week_facts = get_week_context(int(week_match.group(1)))

                answer = ask_ollama(user_query, week_facts=week_facts)

                st.session_state.chat_history.append(("user", user_query))
                if week_facts:
                    # Store the deterministic facts as their own message so
                    # they're always visible in the thread, not just buried in
                    # a system prompt the user can't see.
                    st.session_state.chat_history.append(
                        ("assistant", f"**Data on record for this week:**\n```\n{week_facts}\n```")
                    )
                st.session_state.chat_history.append(("assistant", answer))
            except Exception as e:
                st.error(
                    f"Error connecting to Ollama. Make sure the Ollama service is running "
                    f"and `{OLLAMA_MODEL}` is pulled (`ollama pull {OLLAMA_MODEL}`). Details: {e}"
                )

    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(content)