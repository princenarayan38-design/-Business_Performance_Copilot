# BusinessIntelligence.ai

**An autonomous decision intelligence engine** that goes beyond "what changed" dashboards to investigate *why* — combining statistical anomaly detection, evidence retrieval, hypothesis generation, a self-learning confidence engine, and a local LLM assistant, on either a built-in demo dataset or your own uploaded business data.

Built by Team Scubacats for the Accenture Innovation Challenge 2026.

---

## What this actually does

Most BI dashboards tell you a number moved. This system tries to go one step further:

1. **Detects** statistically meaningful anomalies in a metric (revenue, units, delivery times, or any custom metric you upload) using a robust median/MAD-based method — not just "did this cross an arbitrary line."
2. **Classifies** each anomaly as a growth **opportunity** or a **risk**, since not every large deviation is bad news (a revenue spike and a delivery-delay spike are both "anomalies" but mean opposite things).
3. **Retrieves evidence** — raw records (deliveries, feedback, support tickets, whatever you provide) that fall within the anomaly's time window.
4. **Generates candidate explanations** for the anomaly based on that evidence, with a confidence score derived from how much evidence actually supports each one.
5. **Lets an analyst confirm or reject** each candidate after real investigation — and this feedback doesn't just sit in a log. It retrains a real classifier (`ai/model_trainer.py`) after every single decision, so the system's confidence scoring **gets measurably more accurate every session**, without any manual retraining step. See [Self-Learning Confidence Engine](#self-learning-confidence-engine) below for exactly how.
6. **Forecasts** the metric forward with a simple trend + confidence band.
7. **Chats** with a local Llama 3 model (via Ollama) that's grounded in the actual pipeline output, not just guessing.

**Important honesty note, stated up front:** evidence-to-hypothesis matching is done by **time-window correlation, not causal inference**. If a support ticket and a revenue drop happened the same week, the system will surface that as a "contributing-factor candidate" — it has not proven that one caused the other. This is by design, and the UI labels it that way everywhere (see [Known Limitations](#known-limitations) below).

---

## Two ways to use it

### 1. Demo Mode (built-in dataset)
Ships with a synthetic dataset (`data/raw/*.csv`) simulating a multi-region business with a deliberate anomaly baked in (a North-region disruption). Good for exploring what the system does without needing your own data. Click **"▶️ Run Full Intelligence Pipeline"** in the sidebar.

### 2. Universal Mode (your own data)
Upload any CSV/Excel file, map your own columns (date, metric, category) via dropdowns, and the same statistical engine runs on your real numbers. Optionally upload supporting evidence files (support tickets, delivery logs, etc.) to unlock the full hypothesis-generation pipeline on your own data too. Find this under **"🗂️ Analyze Your Own Data"**.

---

## Project structure

```
businessintelligence-ai/
├── app.py                          # Streamlit dashboard — all UI lives here
├── data/
│   ├── raw/                        # Demo dataset (sales, delivery, feedback, events CSVs)
│   ├── feedback.json               # Analyst confirm/reject history + ML feature snapshots (auto-created)
│   └── confidence_model.joblib     # Trained self-learning classifier (auto-created once enough feedback exists)
├── .env                             # GEMINI_API_KEY (optional — see Setup)
└── ai/
    ├── data_processor.py            # Loads & weekly-aggregates the DEMO dataset
    ├── anomaly_detector.py          # Core stats engine: MAD/Z-score detection + opportunity/risk classification
    ├── evidence_retriever.py        # Pulls raw evidence rows for the DEMO dataset, by time window
    ├── hypothesis_generator.py      # Generates candidate explanations for the DEMO dataset (fixed templates)
    ├── hypothesis_tester.py         # Scores any hypothesis set against its cited evidence (generic — works for both modes)
    ├── reporter.py                  # Executive summary via Gemini, or a deterministic offline fallback
    ├── universal_pipeline.py        # Universal Mode: ingest ANY uploaded file into the detector's expected shape
    ├── generic_evidence.py          # Universal Mode: evidence retrieval from user-uploaded supporting files
    ├── generic_hypothesis.py        # Universal Mode: hypothesis generation from whatever evidence sources exist
    ├── feedback_store.py            # Analyst confirm/reject storage + calibration stats + heuristic confidence recalibration (JSON-backed)
    ├── model_trainer.py              # Trains/retrains a real classifier on accumulated feedback - the self-learning engine
    ├── event_context.py             # Real-world event matching (holidays, disruptions) fed into LLM prompts
    └── forecasting.py               # Trend forecasting (Prophet if installed, linear-trend fallback otherwise)
```

**Why two hypothesis paths?** `hypothesis_generator.py` uses three fixed templates ("Supply Chain Disruption," "Customer Sentiment Shift," "External Event Impact") that were written for the demo's specific business story. `generic_hypothesis.py` instead generates one candidate **per evidence source you actually upload**, named after whatever you called it — it makes no assumptions about your business. Both feed into the same `hypothesis_tester.py` and `reporter.py`, since those two only need generic `.all_records` / `.total_evidence_count` — they don't care which mode produced the hypotheses. Both paths also call the same `model_trainer.py` / `feedback_store.py` self-learning engine, so confidence scores improve from accumulated feedback regardless of which mode generated the hypothesis.

---

## Setup

### 1. Install dependencies
```bash
pip install streamlit pandas numpy plotly ollama python-dotenv pydantic
pip install scikit-learn joblib   # powers the self-learning confidence model (ai/model_trainer.py)
pip install prophet          # optional — enables better forecasting; falls back gracefully if skipped
pip install google-genai     # optional — enables Gemini-powered executive summaries
```

### 2. Set up the local LLM (optional but recommended)
```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3
```
If Ollama isn't running, the chat assistant will show a connection error but the rest of the app works fine.

### 3. Optional: Gemini API key for richer executive summaries
Create a `.env` file **at the project root** (same folder as `app.py`, not inside `ai/`):
```
GEMINI_API_KEY="your-key-here"
```
Without this, the reporter automatically falls back to a deterministic offline summary based on the actual hypothesis verdicts — it does not need Gemini to function.

⚠️ **Never commit `.env` to version control.** Add it to `.gitignore`. If a key is ever accidentally exposed (e.g., pasted into a screenshot or chat), rotate it immediately in the provider's console.

### 4. Run it
```bash
streamlit run app.py
```

If you edit any file under `ai/`, you must **fully restart** Streamlit (`Ctrl+C`, then re-run) — Python caches imported modules for the life of the process, so a browser refresh alone won't pick up backend changes.

---

## Feature guide

### Anomaly detection
Uses a **median + Median Absolute Deviation (MAD)** baseline instead of mean/std — robust to the anomaly itself contaminating its own baseline. Flags a week when `|modified z-score| ≥ 3.5` **and** the deviation is at least 5% in real terms (a purely statistical blip on tiny numbers isn't flagged). Baselines are built causally (never look at future weeks) and self-correcting (a flagged week is excluded from future baselines, so a multi-week event doesn't drag its own comparison point along with it).

### Opportunity vs. Risk
Each anomaly is tagged `OPPORTUNITY` or `RISK` based on metric direction. Revenue/units going up = opportunity; delivery delays/late rates going up = risk. You control this explicitly for uploaded metrics via a "higher is better / lower is better" toggle.

### Evidence & Hypotheses
Every cited evidence ID (e.g. `DEL-1792`, `FB-751`) is clickable and opens the **exact raw row** from the source file it came from, styled with a glowing highlight — proving the citation is real, not hallucinated. Confidence scores are derived from how much of the total retrieved evidence supports each candidate (more evidence, more of it concentrated on one candidate → higher score), capped below 1.0 since this method should never claim certainty.

### Self-Learning Confidence Engine

This is what makes confidence scores **improve every session** instead of staying fixed at design time. It's a two-tier system that upgrades itself automatically as feedback accumulates — there's no separate "training day," no manual retraining step, and no external database; everything lives in `data/feedback.json` and `data/confidence_model.joblib`.

**Tier 1 — Heuristic recalibration (`feedback_store.get_confidence_adjustment`)**
Active from the very first piece of feedback. Looks at every past analyst Confirm/Reject decision for the *exact same* (metric, hypothesis type) pair — e.g. "Supply Chain Disruption & Late Deliveries" raised specifically for "Avg Delivery Days" anomalies — and nudges that hypothesis type's confidence score by up to ±0.12 based on its historical confirm rate. Needs at least 3 prior decisions before it acts, so it never overreacts to one lucky or unlucky guess. Fully transparent: the exact math is in the function's docstring, and the UI shows precisely how many past decisions produced the adjustment.

**Tier 2 — Trained classifier (`ai/model_trainer.py`)**
Once at least 8 labeled decisions exist (with at least 2 Confirmed *and* 2 Rejected — a classifier needs both classes to learn anything), the system trains a `LogisticRegression` on six features snapshotted at feedback time: evidence count, evidence-derived base confidence, anomaly deviation %, modified z-score, severity, and opportunity/risk. The label is simply whether the analyst confirmed or rejected that hypothesis. **This model retrains from scratch after every single Confirm/Reject click** — so the classifier scoring the next hypothesis is never more than one decision stale. Its prediction is blended 50/50 with the evidence-based score, and the UI shows a 🤖 badge naming the sample count and training accuracy it was built on.

Once the trained model has enough data to activate, it takes over from the Tier-1 heuristic for that scoring step; below the data threshold, Tier 1 keeps things running so the "self-learning" behavior is visible even on session one.

**Why logistic regression and not something fancier:** the training set is one row per analyst decision — realistically dozens to a few hundred rows for a single analyst's usage, not a web-scale dataset. A higher-capacity model would overfit immediately and produce confidence numbers that *look* precise but mean nothing, which is exactly the anti-pattern this project's original `_derive_confidence()` was written to avoid. Logistic regression's coefficients are also directly inspectable, which matters for a system whose entire premise is explainable, auditable scoring rather than a black box.

**Honesty note on the reported accuracy:** the number shown in the sidebar ("Train accuracy: X%") is accuracy on the same data the model was trained on, not held-out validation accuracy. With only a handful to a few dozen labeled examples, a real train/test split would be too noisy to be meaningful — this is disclosed directly in the UI rather than presented as a validated benchmark. As real usage volume grows, a proper held-out split becomes worth adding.

**Graceful degradation:** if `scikit-learn`/`joblib` aren't installed, or there isn't yet enough labeled data, `predict_confidence()` returns "not available" and the system falls back to the Tier-1 heuristic automatically — the same optional-dependency pattern already used for Prophet in `forecasting.py`.

### Analyst Calibration (confirm/reject)
Every hypothesis has ✅ Confirm / ❌ Reject buttons. This is the single most important feature for real-world use — it's the only way to check whether the system's own scoring is actually any good, and it's also literally the training signal for the Self-Learning Confidence Engine above. Feedback persists to `data/feedback.json` and the sidebar shows a running **agreement rate** (how often "Validated" was later confirmed true, and "Refuted" confirmed false) alongside the current confidence model's status: whether it's trained yet, on how many examples, and its training accuracy.

### Real-world event context
Upload a CSV of known events (holidays, promotions, disruptions — needs at least a date and description column) and the local LLM will be told about any events matching the week it's discussing, so it can factor real-world context into its commentary instead of only seeing numbers.

### Forecasting
Projects a metric forward a chosen number of weeks with a shaded confidence band. Uses Prophet if installed (better seasonality handling), otherwise a linear-trend + residual-based band that needs no extra dependencies. Labeled clearly as a simple projection, not a validated forecasting model.

### Week Lookup & Dashboard Retargeting
Look up any week's raw numbers directly, or click "🎯 Analyze this week" to re-run the full pipeline against that week specifically — not just the single highest-magnitude anomaly the system finds by default.

---

## Known Limitations

Documented here deliberately, so nobody mistakes this for more rigorous than it is:

- **No seasonality modeling.** Weekly/monthly/holiday patterns in real data will likely get flagged as repeated "anomalies" until you account for this.
- **One global anomaly threshold** across all metrics — a naturally noisy metric will false-positive more than a stable one at the same cutoff.
- **Evidence matching is temporal correlation, not causation.** A record existing in the same week as an anomaly is not proof it caused the anomaly.
- **Confidence scores are heuristic, not statistically validated** — they react to evidence volume, but haven't been checked against real, known outcomes. Use the confirm/reject workflow to start building that validation.
- **The trained confidence classifier's reported accuracy is train-set accuracy, not held-out validation accuracy.** With a small number of labeled analyst decisions, a real train/test split would be too noisy to trust — the sidebar states this plainly. Treat the model as a lightweight, continuously-updating recalibration layer, not a validated predictive model, until real usage volume justifies a proper split.
- **The self-learning loop learns analyst agreement patterns, not ground truth about the business.** If analysts are systematically wrong about a hypothesis type, the model will confidently learn to agree with that mistake. The confirm/reject workflow is only as good as the legwork behind each decision.
- **No seasonality-aware regime-change detection** — a permanent business shift (e.g., a real price increase) will keep getting flagged as "anomalous" indefinitely rather than becoming the new accepted baseline.
- **Universal Mode's full pipeline requires you to supply your own evidence sources.** Without them, the system correctly reports "Unexplained Statistical Deviation" rather than inventing a cause — this is intentional, not a bug.
- **Not access-controlled or audited** beyond the JSON feedback file — fine for a single analyst's own use, not yet ready for multi-user production deployment.

**Recommendation before relying on this for real decisions:** backtest it against a few months of your own historical data where you already know what happened, and start using the confirm/reject workflow immediately so the calibration panel reflects reality rather than showing zero entries.

---

## Tech stack

Python · Streamlit · Plotly · Pandas · NumPy · scikit-learn (self-learning confidence model) · Ollama (local Llama 3) · Google Gemini (optional) · Prophet (optional)