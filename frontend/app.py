"""AutoRoute web portal (Streamlit). Talks ONLY to the gateway via REST (API_URL)."""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
MAX_CHARS = 5000
TOOL_LINKS = {"MLflow (experiments, models)": os.environ.get("MLFLOW_UI_URL", "http://localhost:5000"),
              "Grafana (live monitoring)": os.environ.get("GRAFANA_URL", "http://localhost:3000"),
              "Prometheus (metrics, alerts)": os.environ.get("PROMETHEUS_URL", "http://localhost:9090")}

st.set_page_config(page_title="AutoRoute", page_icon="📬", layout="wide")

SAMPLES = {
    "Debt collection": "A debt collector keeps calling me at work even after I asked them to stop. I do not recognise this debt.",
    "Credit reporting": "There is an account on my credit report that is not mine. I disputed it twice and nothing changed.",
    "Mortgage": "My mortgage servicer raised my escrow payment without explanation and lost my last payment.",
    "Bank account": "The bank charged me three overdraft fees in one day and froze my checking account.",
}


def call_api(method: str, path: str, **kwargs):
    """Call the gateway. Returns (json, error_message); never raises."""
    try:
        resp = requests.request(method, f"{API_URL}{path}", headers=HEADERS, timeout=15, **kwargs)
    except requests.RequestException:
        return None, "Cannot reach the AutoRoute service. Please try again in a moment."
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", "Request failed")
        except ValueError:
            detail = "Request failed"
        if isinstance(detail, list):
            detail = "Please check your input."
        return None, str(detail)
    return resp.json(), None


def page_classify():
    st.header("Classify a complaint")
    st.write("Paste a customer complaint below. AutoRoute will suggest which product team should handle it.")
    choice = st.selectbox("Or try an example", ["(write my own)"] + list(SAMPLES))
    default = SAMPLES.get(choice, "")
    text = st.text_area("Complaint text", value=default, height=180, max_chars=MAX_CHARS,
                        key=f"text_{choice}")
    st.caption(f"{len(text)} / {MAX_CHARS} characters")
    if st.button("Classify", type="primary"):
        if not text.strip():
            st.warning("Please type or paste a complaint first.")
            return
        with st.spinner("Analysing..."):
            result, error = call_api("POST", "/predict", json={"text": text})
        if error:
            st.error(error)
            return
        explain, _ = call_api("POST", "/explain", json={"text": text, "category": result["category"]})
        st.session_state["last"] = {"result": result, "explain": explain, "feedback_sent": False}
    show_result()


def show_result():
    """Show the last prediction (kept in session state so the feedback buttons work after a rerun)."""
    last = st.session_state.get("last")
    if not last:
        return
    result = last["result"]
    st.success(f"Suggested team: **{result['category']}**")
    st.progress(min(max(result["confidence"], 0.0), 1.0), text=f"Confidence {result['confidence']:.0%}")
    if result["confidence"] < 0.5:
        st.info("The system is not very sure about this one. A human should double-check it.")
    st.subheader("Top 3 guesses")
    st.dataframe(pd.DataFrame(result["top3"]).rename(columns={"category": "Category", "score": "Score"}),
                 hide_index=True, use_container_width=True)
    with st.expander("Why this category?"):
        explain = last.get("explain")
        if not explain:
            st.write("An explanation is not available right now.")
        else:
            matched = explain.get("matched_terms", [])
            st.write("Words and phrases in your text that are typical for this category: " +
                     (", ".join(f"**{t}**" for t in matched) if matched else "none of the strongest ones."))
            st.caption("Typical for this category overall: " + ", ".join(explain.get("top_terms", [])))
    st.write("Was this suggestion correct?")
    c1, c2, _ = st.columns([1, 1, 4])
    if not last["feedback_sent"]:
        for col, label, correct in ((c1, "Yes", True), (c2, "No", False)):
            if col.button(label, key=f"fb_{label}"):
                _, error = call_api("POST", "/feedback", json={"predicted": result["category"], "correct": correct})
                st.error(error) if error else st.session_state["last"].update(feedback_sent=True)
                st.rerun()
    else:
        st.caption("Thank you! Your feedback helps us measure real accuracy.")


DAG = """digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e8f1fb", color="#1f3a5f"];
 test [label="test\\n(flake8 + pytest)"]; ingest [label="ingest\\n(Ray)"]; validate [label="validate\\n(data quality)"];
 prepare; train [label="train\\n(MLflow)"]; evaluate [label="evaluate\\n(promote champion)"];
 test -> ingest -> validate -> prepare -> train -> evaluate; }"""


def page_pipeline():
    st.header("Pipeline console")
    status, err = call_api("GET", "/pipeline/status")
    if err:
        st.error(err)
    else:
        state = status.get("status", "idle")
        icon = {"running": "🟡", "success": "🟢", "failed": "🔴", "idle": "⚪"}.get(state, "⚪")
        st.subheader(f"{icon} Current state: {state}")

    c1, c2, c3 = st.columns(3)
    running = bool(status) and status.get("status") == "running"
    if c1.button("Retrain model", disabled=running, type="primary",
                 help="Re-runs training and evaluation on the current data."):
        _, e = call_api("POST", "/pipeline/trigger", json={"action": "retrain"})
        st.error(e) if e else st.success("Retraining started. Click Refresh to follow progress.")
    if c2.button("Run full pipeline", disabled=running, help="Re-runs every stage whose inputs changed."):
        _, e = call_api("POST", "/pipeline/trigger", json={"action": "full"})
        st.error(e) if e else st.success("Pipeline started. Click Refresh to follow progress.")
    c3.button("Refresh")

    st.subheader("Pipeline stages (DVC)")
    st.graphviz_chart(DAG)
    st.caption("Open the underlying tools:  " + "  |  ".join(f"[{n}]({u})" for n, u in TOOL_LINKS.items()))

    st.subheader("Speed and throughput")
    summary, e = call_api("GET", "/pipeline/summary")
    m = (summary or {}).get("metrics", {})
    if e:
        st.warning(e)
    elif not m:
        st.info("No training run yet.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Ingest (Ray), rows/s", f"{m.get('ingest_ray_full_rows_per_sec', 0):,.0f}")
        b.metric("Ingest time (s)", f"{m.get('ingest_ray_full_seconds', 0):,.1f}")
        c.metric("Training time (s)", f"{m.get('train_seconds', 0):,.1f}")
        d.metric("Rows after validation", f"{m.get('dq_rows', 0):,.0f}")
        bench = {"Single process": m.get("ingest_single_rows_per_sec"), "Multiprocessing": m.get("ingest_multiprocessing_rows_per_sec"),
                 "Ray": m.get("ingest_ray_rows_per_sec")}
        if all(v is not None for v in bench.values()):
            st.caption("Cleaning benchmark on the same sample (rows per second, higher is faster)")
            st.bar_chart(pd.DataFrame({"rows/s": bench}))
        if "dq_hard_failures" in m:
            ok = m["dq_hard_failures"] == 0
            st.write(("✅ Data-quality checks passed" if ok else "❌ Data-quality checks failed") +
                     f" | Tests: {m.get('ci_tests_passed', 0):.0f} passed, {m.get('ci_tests_failed', 0):.0f} failed")

    st.subheader("Pipeline jobs (successes and failures)")
    jobs, e = call_api("GET", "/pipeline/jobs")
    if e:
        st.error(e)
    elif jobs:
        df = pd.DataFrame(jobs)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption(f"{(df['status'] == 'success').sum()} succeeded, {(df['status'] == 'failed').sum()} failed")
    else:
        st.info("No pipeline jobs have been triggered from this console yet.")
    if status and status.get("log_tail"):
        with st.expander("Latest job log"):
            st.code(status["log_tail"])

    st.subheader("Training runs (MLflow)")
    runs, e = call_api("GET", "/pipeline/runs")
    if e:
        st.error(e)
    elif runs:
        st.dataframe(pd.DataFrame(runs), hide_index=True, use_container_width=True)

    st.subheader("Current model")
    info, e = call_api("GET", "/model/info")
    if e:
        st.warning(e)
    else:
        a, b, c = st.columns(3)
        a.metric("Version", info["version"])
        b.metric("Macro F1", f"{info['macro_f1']:.3f}" if info.get("macro_f1") is not None else "n/a")
        c.metric("Trained", info["trained_at"][:10])


def page_drift():
    st.header("Health and drift")
    health, e = call_api("GET", "/health")
    if e:
        st.error(e)
    else:
        cols = st.columns(len(health["components"]))
        for col, (name, up) in zip(cols, health["components"].items()):
            col.metric(name.replace("_", " ").title(), "Up" if up else "Down")
    drift, e = call_api("GET", "/drift/status")
    if e:
        st.error(e)
    elif not drift.get("enabled"):
        st.info("Drift monitoring is not enabled (no baseline file found).")
    elif not drift["buffer_ready"]:
        st.info(f"Collecting data: {drift['buffer_size']} requests seen. Drift scores appear after enough requests.")
    else:
        a, b, c = st.columns(3)
        psi_val = drift["psi_prediction"]
        a.metric("Prediction drift (PSI)", f"{psi_val:.3f}",
                 "stable" if psi_val < 0.1 else ("moderate" if psi_val < 0.25 else "significant"),
                 delta_color="off")
        b.metric("Unknown-word rate", f"{drift['oov_rate']:.1%}")
        c.metric("Length KS p-value", f"{drift['length_ks_pvalue']:.3g}" if drift["length_ks_pvalue"] is not None else "n/a")
    fb, e = call_api("GET", "/feedback/stats")
    if not e and fb and fb["accuracy"] is not None:
        st.metric("Accuracy confirmed by users (recent feedback)", f"{fb['accuracy']:.0%}", f"{fb['window']} answers", delta_color="off")
    st.caption("Live dashboards are in Grafana (port 3000).")
    st.button("Refresh", key="refresh_drift")


def page_help():
    st.header("How to use AutoRoute")
    st.markdown("""
**1. Classify** - choose *Classify*, paste a complaint (or pick an example) and press **Classify**.
You will see the suggested team and how confident the system is. If confidence is low, a person should check it.

**2. Pipeline console** - shows whether the model is being retrained, the history of successful and failed jobs,
and the current model's quality. Press **Retrain model** to train again.

**3. Was it right?** - under each result press Yes or No. This tells us how accurate the system really is.

**4. Health and drift** - shows whether all services are running and whether incoming complaints look different
from the ones the model learned from.

**Troubleshooting:** if you see *Cannot reach the AutoRoute service*, wait a minute and press the button again.
If it keeps happening, ask the administrator to check the services.
""")


PAGES = {"Classify": page_classify, "Pipeline console": page_pipeline,
         "Health and drift": page_drift, "Help": page_help}
choice = st.sidebar.radio("Go to", list(PAGES))
st.sidebar.caption(f"Backend: {API_URL}")
PAGES[choice]()
