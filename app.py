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

st.set_page_config(page_title="Business Intelligence AI", page_icon="📊", layout="wide")

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


def build_forecast_chart(historical: pd.DataFrame, forecast: pd.DataFrame, title: str) -> go.Figure:
    """
    Builds a combined historical + forecast Plotly figure: solid line for
    actuals, dashed line for the point forecast, and a shaded confidence
    band (fill between lower/upper) using Plotly's tonexty fill trick.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=historical["week"], y=historical["value"], mode="lines+markers",
        name="Historical", line=dict(color="#00e5ff", width=2),
    ))

    # Confidence band: draw upper bound invisibly, then lower bound with
    # fill='tonexty' to shade the area between them.
    fig.add_trace(go.Scatter(
        x=forecast["week"], y=forecast["upper"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["week"], y=forecast["lower"], mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor="rgba(168,85,247,0.18)",
        name="Confidence band", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["week"], y=forecast["forecast"], mode="lines+markers",
        name="Forecast", line=dict(color="#a855f7", width=2, dash="dash"),
    ))

    fig.update_layout(title=title, xaxis_title="Week", yaxis_title="Value")
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
        st.error(f"⚠️ {source}")
        return

    st.markdown(
        f"**🔎 Evidence `{evidence_id}`** — matched row in `{source}`:"
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
        if col.button(f"🔗 {eid}", key=f"{key_prefix}_{eid}", use_container_width=True):
            st.session_state.selected_evidence_id = eid



st.title("📊 Business Intelligence AI Assistant")
st.markdown("Powered by **Local Llama 3 (Ollama)** & Automated Root-Cause Analytics.")
notif_banner("🟢 System online — pipeline, evidence retrieval, and local LLM assistant are ready.")

# ---------------------------------------------------------------------------
# Session state — keeps the last pipeline run around so the chat assistant
# can answer questions grounded in the actual anomaly/evidence/hypotheses,
# instead of just generic chit-chat.
# ---------------------------------------------------------------------------
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
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

    result = st.session_state.pipeline_result
    if not result:
        return base + "\n\nNo pipeline run has been executed yet — no anomaly data is available."

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

    context = f"""
Current pipeline findings:
- Anomaly: {anomaly.metric} in {anomaly.dimension_value} ({anomaly.dimension}), week {anomaly.week}
  Actual={anomaly.actual:.1f}, Expected={anomaly.expected:.1f}, Deviation={anomaly.deviation_pct:+.1%}, Severity={anomaly.severity}
- Hypotheses tested:
{chr(10).join(hyp_lines)}
- Executive summary: {report.executive_summary}
- Recommendations: {', '.join(report.key_recommendations)}
"""

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
st.sidebar.header("Configuration Status")
api_key_status = "🟢 Connected (Gemini Active)" if os.environ.get("GEMINI_API_KEY") else "🟡 Offline Fallback Mode"
st.sidebar.markdown(f"**Reporter LLM:** {api_key_status}")
st.sidebar.markdown(f"**Local LLM:** 🟢 Ollama (`{OLLAMA_MODEL}`)")

# ---------------------------------------------------------------------------
# Calibration panel — the closest thing this system has to ground truth.
# Shows whether system verdicts (Validated/Refuted) actually match what
# analysts confirmed in reality, based on accumulated feedback.
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Analyst Calibration")
calib = get_calibration_stats()
if calib["total"] == 0:
    st.sidebar.caption("No analyst feedback recorded yet. Confirm or reject hypotheses in the AI Hypotheses tab to start building a track record.")
else:
    st.sidebar.metric("Feedback recorded", calib["total"])
    st.sidebar.metric("Confirmed vs Rejected", f"{calib['confirmed']} / {calib['rejected']}")
    if calib["agreement_rate"] is not None:
        st.sidebar.metric("System-vs-analyst agreement", f"{calib['agreement_rate']:.0%}")
        st.sidebar.caption(
            "Agreement = system said Validated and analyst Confirmed, OR system said Refuted "
            "and analyst Rejected. Low agreement means the scoring thresholds need retuning."
        )
    st.sidebar.caption(
        "🧠 This feedback isn't just displayed — it retrains a real classifier "
        "(see 🤖 Confidence Model panel below) and, until that model has enough data, "
        "a transparent heuristic recalibrates scores for matching hypothesis types."
    )

# ---------------------------------------------------------------------------
# Trained model status — the literal "self-learning" component. Shows
# whether enough analyst feedback exists to train a real classifier yet,
# and if so, how big its training set is and what its train accuracy is.
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Confidence Model")
model_status = get_model_status()
if not model_status["trained"]:
    st.sidebar.caption(f"Not trained yet — {model_status['reason']}")
    st.sidebar.caption(
        "Falls back to the heuristic recalibration above until there's enough "
        "labeled feedback (both Confirmed AND Rejected examples) to fit a classifier."
    )
else:
    st.sidebar.metric("Trained on", f"{model_status['sample_count']} examples")
    st.sidebar.metric("Train accuracy", f"{model_status['train_accuracy']:.0%}")
    st.sidebar.caption(
        f"Last retrained: {model_status['trained_at'][:19].replace('T', ' ')} UTC. "
        "Retrains automatically after every Confirm/Reject click — this is train-set "
        "accuracy, not held-out validation, since the dataset is still small."
    )

def run_pipeline_for_anomaly(anomaly, revenue_anomalies):
    """
    Runs evidence retrieval -> hypothesis generation -> testing -> reporting
    for ONE specific anomaly, and stores the result as the dashboard's
    current headline. Shared by the initial pipeline run and by
    week-retargeting, so both paths behave identically.
    """
    evidence = retrieve_evidence_for_anomaly(anomaly)
    gen_result = generate_hypotheses_for_anomaly(evidence)
    evaluation = test_hypotheses(evidence, gen_result)
    report = generate_executive_report(evidence, gen_result, evaluation)

    st.session_state.pipeline_result = {
        "anomaly": anomaly,
        "evidence": evidence,
        "gen_result": gen_result,
        "evaluation": evaluation,
        "report": report,
        "all_anomalies": revenue_anomalies,
    }
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
if st.sidebar.button("▶️ Run Full Intelligence Pipeline"):
    with st.spinner("Analyzing data streams and processing pipeline..."):
        datasets = build_processed_dataset()
        sales_weekly = datasets.get("sales_weekly")

        # Keep the full processed dataset around so the chat assistant can
        # look up ANY week on request, not just the top anomaly below.
        st.session_state.datasets = datasets

        # Detect across revenue/units/delivery (all metrics), not just
        # revenue, so week-specific questions have full anomaly context.
        st.session_state.all_detected_anomalies = detect_all_anomalies()

        revenue_anomalies = detect_anomalies(sales_weekly, value_col="revenue", metric_label="Sales", dimension="region")
        st.session_state.revenue_anomalies = revenue_anomalies

        if not revenue_anomalies:
            st.session_state.pipeline_result = None
            st.warning("No anomalies detected in the dataset.")
        else:
            run_pipeline_for_anomaly(revenue_anomalies[0], revenue_anomalies)

# ---------------------------------------------------------------------------
# Universal / Bring-Your-Own-Data mode — upload any CSV/Excel file, map its
# columns to the pipeline's required roles, and run the same statistical
# detection engine on it. Fully independent of the built-in demo dataset.
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("🗂️ Analyze Your Own Data (upload CSV / Excel)", expanded=False):
    st.caption(
        "Works with any table that has a date/time column and a numeric metric column. "
        "A category column (region, product, store, etc.) is optional but lets the detector "
        "flag anomalies per-category instead of on one combined total."
    )

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

    with st.expander("📅 Add real-world event context (optional)"):
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

    with st.expander("🧾 Add supporting evidence sources (optional, for full pipeline)"):
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
                st.warning(f"⚠️ {issue}")

            if st.button("🚀 Run Anomaly Detection on My Data", type="primary"):
                try:
                    with st.spinner("Aggregating and analyzing your data..."):
                        weekly = build_weekly_series(user_df, mapping)
                        metric_display_name = value_col
                        detected = detect_anomalies(
                            weekly, value_col="value", metric_label=metric_display_name,
                            dimension="dimension_value", lower_is_better=mapping.lower_is_better,
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
            st.plotly_chart(plotly_dark_layout(bar, height=max(280, 40 * len(detected))), use_container_width=True)

            with st.expander("View anomaly details"):
                for a in sorted(detected, key=lambda x: x.week):
                    tag = "📈 Opportunity" if a.nature == "OPPORTUNITY" else "⚠️ Risk"
                    st.markdown(
                        f"**{tag} — {a.dimension_value}, Week {a.week}**: "
                        f"{a.actual:,.2f} vs expected {a.expected:,.2f} "
                        f"({a.deviation_pct:+.1%}, |Z|={abs(a.modified_z_score):.2f}, severity {a.severity})"
                    )

            with st.expander("🔮 Forecast a category forward"):
                categories = sorted(weekly["dimension_value"].unique().tolist())
                chosen_cat = st.selectbox("Category to forecast", options=categories, key="custom_forecast_cat")
                horizon2 = st.slider("Weeks to forecast ahead", min_value=2, max_value=12, value=4, key="custom_forecast_horizon")
                if st.button("Generate Forecast", key="custom_forecast_btn"):
                    cat_series = weekly[weekly["dimension_value"] == chosen_cat][["week", "value"]].sort_values("week")
                    try:
                        forecast_df, backend = generate_forecast(cat_series, horizon2, min_date=None)
                        fc_fig = build_forecast_chart(cat_series, forecast_df, title=f"{chosen_cat} — Historical + Forecast")
                        st.plotly_chart(plotly_dark_layout(fc_fig, height=380), use_container_width=True)
                        st.caption(f"Backend: **{backend}**. Simple trend projection — treat the band as a rough guide, not a guarantee.")
                    except ValueError as e:
                        st.warning(str(e))

            # --- Full pipeline: hypotheses + testing + report, on YOUR data ---
            st.markdown("---")
            st.subheader("🧠 Run Full Root-Cause Pipeline on a Detected Anomaly")
            anomaly_labels = [f"Week {a.week} · {a.dimension_value} · {a.deviation_pct:+.1%}" for a in detected]
            chosen_idx = st.selectbox(
                "Choose an anomaly to investigate", options=range(len(detected)),
                format_func=lambda i: anomaly_labels[i], key="custom_anomaly_picker",
            )
            if not st.session_state.custom_evidence_sources:
                st.info("No evidence sources uploaded above — the pipeline will still run, but will report 'Unexplained Statistical Deviation' since there's nothing to correlate against.")

            if st.button("🚀 Run Full Pipeline", type="primary", key="custom_full_pipeline_btn"):
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
                st.caption(
                    "⚠️ Candidates below are matched by **time window only** — not proven causes. "
                    "Confirm or reject after checking with your team."
                )
                for h in cpr["gen_result"].hypotheses:
                    test_result = next((t for t in cpr["evaluation"].test_results if t.hypothesis_id == h.hypothesis_id), None)
                    with st.expander(f"[{h.hypothesis_id}] {h.title}" + (f" — {verdict_pill(test_result.verdict)}" if test_result else ""), expanded=False):
                        st.write(h.description)
                        st.caption(f"🧮 Confidence: {h.confidence_score:.2f} — {h.evidence_basis}")
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
                            tag = "✅ Confirmed" if prior_fb["analyst_verdict"] == CONFIRMED else "❌ Rejected"
                            st.info(f"**Analyst feedback on record:** {tag}")

                        note_key = f"custom_note_{h.hypothesis_id}_{a.week}"
                        note = st.text_input("Optional note", key=note_key, label_visibility="collapsed", placeholder="Optional note")
                        fcol1, fcol2 = st.columns(2)
                        with fcol1:
                            if st.button("✅ Confirm", key=f"custom_confirm_{h.hypothesis_id}_{a.week}", use_container_width=True):
                                record_feedback(a, h,
                                                 system_verdict=test_result.verdict if test_result else "Unknown",
                                                 system_score=test_result.final_score if test_result else h.confidence_score,
                                                 analyst_verdict=CONFIRMED, note=note)
                                with st.spinner("🔄 Retraining confidence model on updated feedback..."):
                                    train_model()
                                st.rerun()
                        with fcol2:
                            if st.button("❌ Reject", key=f"custom_reject_{h.hypothesis_id}_{a.week}", use_container_width=True):
                                record_feedback(a, h,
                                                 system_verdict=test_result.verdict if test_result else "Unknown",
                                                 system_score=test_result.final_score if test_result else h.confidence_score,
                                                 analyst_verdict=REJECTED, note=note)
                                with st.spinner("🔄 Retraining confidence model on updated feedback..."):
                                    train_model()
                                st.rerun()

                st.markdown("##### Executive Summary")
                st.caption("Based on time-correlated candidates — verify before treating as confirmed cause.")
                st.write(cpr["report"].executive_summary)
                for rec in cpr["report"].key_recommendations:
                    st.markdown(f"- {rec}")

    st.caption(
        "Note: hypothesis generation and evidence citation currently require the curated "
        "delivery/feedback/events CSVs from the demo dataset. Uploaded-data mode runs the "
        "same statistical detection engine, but root-cause hypotheses aren't generated for it yet."
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
result = st.session_state.pipeline_result

if result:
    anomaly = result["anomaly"]
    gen_result = result["gen_result"]
    evaluation = result["evaluation"]
    report = result["report"]
    all_anomalies = result["all_anomalies"]
    sales_all = st.session_state.datasets["sales_weekly"] if st.session_state.datasets is not None else None

    st.caption("This is the pipeline's headline finding — the single highest-magnitude anomaly across all metrics and weeks. Use 'Look Up a Specific Week' below to inspect any other week.")

    is_opportunity = getattr(anomaly, "nature", "RISK") == "OPPORTUNITY"

    if is_opportunity:
        notif_banner(
            f"📈 Growth Opportunity Detected: <b>{anomaly.metric}</b> in <b>{anomaly.dimension_value}</b> "
            f"(Week {anomaly.week}) — {anomaly.deviation_pct:+.1%} vs. expected. Investigating what drove it, so it can be replicated.",
            danger=False,
        )
    else:
        notif_banner(
            f"🚨 Detected Anomaly: <b>{anomaly.metric}</b> in <b>{anomaly.dimension_value}</b> "
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
        glow_card("Modified Z-Score", f"{anomaly.modified_z_score:.2f}", "threshold ≥ 3.5", "info")
    with kc4:
        nature_label = "📈 Growth Driver" if is_opportunity else "⚠️ Risk"
        glow_card("Nature", nature_label, "opportunity vs. risk classification", "good" if is_opportunity else "medium")

    # --- Geometric visual: anomaly score gauge + weekly trend ---
    gcol1, gcol2 = st.columns([1, 2])
    with gcol1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=anomaly.anomaly_score * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Anomaly Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ec4899"},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(34,255,176,0.25)"},
                    {"range": [40, 75], "color": "rgba(255,157,61,0.25)"},
                    {"range": [75, 100], "color": "rgba(255,77,109,0.3)"},
                ],
            },
        ))
        st.plotly_chart(plotly_dark_layout(gauge, height=280), use_container_width=True)
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
            st.plotly_chart(plotly_dark_layout(trend), use_container_width=True)

            # --- Forecast panel ---
            with st.expander(f"🔮 Forecast future {anomaly.dimension_value} revenue"):
                horizon = st.slider("Weeks to forecast ahead", min_value=2, max_value=12, value=4, key="forecast_horizon_main")
                if st.button("Generate Forecast", key="forecast_btn_main"):
                    fc_series = region_series[["week", "revenue"]].rename(columns={"revenue": "value"})
                    try:
                        min_date = st.session_state.datasets.get("min_date") if st.session_state.datasets else None
                        forecast_df, backend = generate_forecast(fc_series, horizon, min_date=min_date)
                        fc_fig = build_forecast_chart(fc_series, forecast_df, title=f"{anomaly.dimension_value} Revenue — Historical + Forecast")
                        st.plotly_chart(plotly_dark_layout(fc_fig, height=380), use_container_width=True)
                        st.caption(
                            f"Backend: **{backend}**. This is a simple trend projection, not a "
                            "validated statistical model — treat the shaded band as a rough guide, "
                            "widening naturally the further out it projects."
                        )
                    except ValueError as e:
                        st.warning(str(e))
    if len(all_anomalies) > 1:
        st.caption(f"{len(all_anomalies) - 1} additional anomal{'y' if len(all_anomalies) == 2 else 'ies'} detected in the Sales/revenue metric alone.")

    all_metric_anomalies = st.session_state.get("all_detected_anomalies", [])
    if len(all_metric_anomalies) > 1:
        with st.expander(f"📋 View all {len(all_metric_anomalies)} anomalies detected across every metric (revenue, units sold, delivery)"):
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
            st.plotly_chart(plotly_dark_layout(bar, height=max(280, 40 * len(all_metric_anomalies))), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["💡 AI Hypotheses", "🔬 Validation & Evidence", "📋 Executive Briefing"])

    with tab1:
        header = "Growth-Driver Candidates (What Went Right)" if is_opportunity else "Contributing-Factor Candidates (Time-Correlated)"
        st.subheader(header)
        st.caption(
            "⚠️ These candidates are matched to the anomaly by **time window only** — "
            "they are NOT proven causes. Confirm or reject each one after checking with the "
            "actual team/region so the system's scoring can be validated against reality."
        )
        st.write(gen_result.anomaly_summary)

        for h in gen_result.hypotheses:
            # Find the matching test result for verdict/score used in feedback logging
            test_result = next((t for t in evaluation.test_results if t.hypothesis_id == h.hypothesis_id), None)

            with st.expander(f"[{h.hypothesis_id}] {h.title}"):
                st.write(h.description)
                st.caption(f"🧮 Confidence: {h.confidence_score:.2f} — {h.evidence_basis}")
                if h.confidence_adjustment != 0:
                    st.success(h.learning_note)
                elif h.learning_sample_count > 0:
                    st.caption(h.learning_note)
                st.markdown("**Cited Evidence (co-occurring in time) — click an ID to inspect the exact raw row:**")
                evidence_id_buttons(h.supporting_evidence_ids, key_prefix=h.hypothesis_id)

                st.markdown("---")
                prior_feedback = get_feedback(anomaly, h.hypothesis_id)
                if prior_feedback:
                    tag = "✅ Confirmed" if prior_feedback["analyst_verdict"] == CONFIRMED else "❌ Rejected"
                    st.info(f"**Analyst feedback on record:** {tag}" + (f" — _{prior_feedback['note']}_" if prior_feedback.get("note") else ""))

                st.markdown("**Analyst review:** did you verify this against what actually happened?")
                note_key = f"note_{h.hypothesis_id}_{anomaly.week}"
                note = st.text_input("Optional note (what you found)", key=note_key, label_visibility="collapsed", placeholder="Optional note (what you found)")
                fb_col1, fb_col2 = st.columns(2)
                with fb_col1:
                    if st.button("✅ Confirm this factor", key=f"confirm_{h.hypothesis_id}_{anomaly.week}", use_container_width=True):
                        record_feedback(
                            anomaly, h,
                            system_verdict=test_result.verdict if test_result else "Unknown",
                            system_score=test_result.final_score if test_result else h.confidence_score,
                            analyst_verdict=CONFIRMED, note=note,
                        )
                        with st.spinner("🔄 Retraining confidence model on updated feedback..."):
                            train_model()
                        st.rerun()
                with fb_col2:
                    if st.button("❌ Reject this factor", key=f"reject_{h.hypothesis_id}_{anomaly.week}", use_container_width=True):
                        record_feedback(
                            anomaly, h,
                            system_verdict=test_result.verdict if test_result else "Unknown",
                            system_score=test_result.final_score if test_result else h.confidence_score,
                            analyst_verdict=REJECTED, note=note,
                        )
                        with st.spinner("🔄 Retraining confidence model on updated feedback..."):
                            train_model()
                        st.rerun()

        if st.session_state.selected_evidence_id:
            st.markdown("---")
            render_evidence_inspector(st.session_state.selected_evidence_id)
            if st.button("✖ Close inspector"):
                st.session_state.selected_evidence_id = None
                st.rerun()

    with tab2:
        st.subheader("Hypothesis Testing & Scoring")
        for test in evaluation.test_results:
            st.markdown(
                f"**[{test.hypothesis_id}] {test.title}** — {verdict_pill(test.verdict)} &nbsp; Score: **{test.final_score}**",
                unsafe_allow_html=True,
            )
            st.caption(test.rationale)

        st.markdown("---")
        st.subheader("Raw Evidence Retrieved")
        ev = result["evidence"]
        ecol1, ecol2 = st.columns([1, 1])
        with ecol1:
            glow_card("Delivery Records", str(len(ev.delivery_records)), "delivery.csv", "info")
            glow_card("Feedback Records", str(len(ev.feedback_records)), "customer_feedback.csv", "purple")
            glow_card("Event Records", str(len(ev.event_records)), "events.csv", "medium")
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
                st.plotly_chart(plotly_dark_layout(donut, height=300), use_container_width=True)
        with st.expander("View raw evidence details"):
            for item in ev.delivery_records + ev.feedback_records + ev.event_records:
                st.markdown(f"**{item.evidence_id}** ({item.source_file}) — {item.summary}")

    with tab3:
        st.subheader("Executive Summary & Action Plan")
        st.caption("Based on time-correlated candidates — verify with the team before treating this as confirmed cause.")
        st.write(report.executive_summary)
        st.markdown("### Key Recommendations")
        for rec in report.key_recommendations:
            st.markdown(f"- {rec}")
else:
    st.info("Click **Run Full Intelligence Pipeline** in the sidebar to get started, or ask the local AI assistant a question below.")

# ---------------------------------------------------------------------------
# Week-specific data lookup — browse any week directly, independent of
# which single anomaly the pipeline happened to highlight.
# ---------------------------------------------------------------------------
if st.session_state.datasets is not None:
    st.markdown("---")
    st.subheader("🔍 Look Up a Specific Week")
    sales_all = st.session_state.datasets["sales_weekly"]
    min_week, max_week = int(sales_all["week"].min()), int(sales_all["week"].max())
    chosen_week = st.number_input(
        f"Week number ({min_week}-{max_week}):",
        min_value=min_week, max_value=max_week, value=min_week, step=1,
    )
    wcol1, wcol2 = st.columns(2)
    with wcol1:
        if st.button("Show week data"):
            st.code(get_week_context(int(chosen_week)), language=None)
    with wcol2:
        if st.button("🎯 Analyze this week (retarget dashboard)", type="primary"):
            candidates = [a for a in st.session_state.all_detected_anomalies if a.week == int(chosen_week)]
            if not candidates:
                st.warning(
                    f"No anomaly was flagged for week {int(chosen_week)} across any metric — "
                    "nothing for the pipeline to investigate. Use 'Show week data' above for the raw numbers."
                )
            else:
                # If multiple metrics flagged this week, retarget on the strongest one.
                def _z_key(a):
                    z = a.modified_z_score
                    return abs(z) if z not in (float("inf"), float("-inf")) else float("inf")

                target = max(candidates, key=_z_key)
                with st.spinner(f"Running root-cause analysis for week {int(chosen_week)}..."):
                    run_pipeline_for_anomaly(target, st.session_state.revenue_anomalies)
                st.success(f"Dashboard retargeted to week {int(chosen_week)} ({target.metric} in {target.dimension_value}). Scroll up ⬆️ to see the full analysis.")
                st.rerun()

# ---------------------------------------------------------------------------
# Interactive Local Ollama Chat Assistant — context-aware + suggested questions
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("💬 Ask the AI Assistant (Local Llama 3)")

if result:
    st.caption("The assistant has access to the current pipeline's anomaly, evidence, and hypotheses.")
    anomaly = result["anomaly"]
    suggested_questions = [
        f"Why did {anomaly.metric} anomaly happen in {anomaly.dimension_value}?",
        "Which hypothesis has the strongest evidence, and why?",
        "What should leadership do first based on this analysis?",
        "What evidence would change the current verdicts?",
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
    "Or type your own question (editable/copyable):",
    key="user_query_input",
)

ask_col, clear_col = st.columns([1, 1])
ask_clicked = ask_col.button("Ask Local AI", type="primary")
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