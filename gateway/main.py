"""AutoRoute gateway (FastAPI).

Validates requests, calls the MLflow model server, records Prometheus metrics and drift,
and exposes the pipeline-console endpoints used by the Streamlit frontend.
"""
import logging
import threading
import time

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from gateway import metrics, mlflow_client
from gateway.config import settings
from gateway.drift import DriftMonitor
from gateway.feedback import FeedbackTracker
from gateway.schemas import (ClassScore, ExplainRequest, FeedbackRequest, PredictRequest, PredictResponse,
                             TriggerRequest)
from text_utils import tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gateway")

app = FastAPI(title="AutoRoute Gateway", version="1.0")
drift = DriftMonitor.from_file(settings.baseline_path, settings.drift_window, settings.drift_min_buffer)
feedback = FeedbackTracker(settings.feedback_window)
_last_auto_retrain = [0.0]


def verify_key(x_api_key: str = Header(default="")):
    """Optional API-key check (enabled when API_KEY env var is set)."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Count every request and time it (skips /metrics itself)."""
    path = request.url.path
    if path == "/metrics":
        return await call_next(request)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        log.exception("Unhandled error on %s", path)
        status, response = 500, Response('{"detail":"Internal server error"}', status_code=500,
                                         media_type="application/json")
    metrics.REQUESTS.labels(path, str(status)).inc()
    metrics.LATENCY.labels(path).observe(time.perf_counter() - start)
    return response


def call_model_server(text: str) -> dict:
    """POST to the MLflow model server and return {class: probability}."""
    payload = {"dataframe_split": {"columns": ["narrative"], "data": [[text]]}}
    try:
        resp = requests.post(f"{settings.model_server_url}/invocations", json=payload, timeout=10)
    except requests.RequestException as exc:
        log.error("Model server unreachable: %s", exc)
        raise HTTPException(status_code=503, detail="Model server unavailable")
    if resp.status_code != 200:
        log.error("Model server returned %s: %s", resp.status_code, resp.text[:200])
        raise HTTPException(status_code=502, detail="Model server returned an error")
    try:
        return resp.json()["predictions"][0]
    except (KeyError, IndexError, ValueError):
        raise HTTPException(status_code=502, detail="Unexpected model server response")


def is_up(url: str) -> bool:
    try:
        return requests.get(url, timeout=1.5).status_code == 200
    except requests.RequestException:
        return False


def component_status() -> dict:
    status = {"model_server": is_up(f"{settings.model_server_url}/ping"),
              "mlflow": is_up(f"{settings.mlflow_uri}/health"),
              "trainer": is_up(f"{settings.trainer_url}/health"),
              "frontend": is_up(f"{settings.frontend_url}/_stcore/health")}
    for name, up in status.items():
        metrics.COMPONENT_UP.labels(name).set(1 if up else 0)
    return status


@app.get("/health")
def health():
    status = component_status()
    return {"status": "ok" if status["model_server"] else "degraded", "components": status}


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_key)])
def predict(req: PredictRequest):
    probs = call_model_server(req.text)
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    category, confidence = ranked[0]
    metrics.PREDICTIONS.labels(category).inc()
    metrics.CONFIDENCE.observe(confidence)
    if confidence < settings.low_confidence:
        metrics.LOW_CONFIDENCE.inc()
    maybe_auto_retrain(drift.record(req.text, category))
    return PredictResponse(category=category, confidence=round(confidence, 4),
                           top3=[ClassScore(category=c, score=round(s, 4)) for c, s in ranked[:3]])


def maybe_auto_retrain(drift_result: dict):
    """Optional (AUTO_RETRAIN_ON_DRIFT=true): start the full pipeline when PSI is high, at most once per cooldown.

    Retraining only helps if new data was added to data/raw/train, hence the 'full' action
    (DVC re-runs ingest only when the raw data changed).
    """
    if not settings.auto_retrain or not drift_result.get("buffer_ready"):
        return
    if (drift_result.get("psi_prediction") or 0) <= settings.auto_retrain_psi:
        return
    if time.time() - _last_auto_retrain[0] < settings.auto_retrain_cooldown_s:
        return
    _last_auto_retrain[0] = time.time()
    metrics.AUTO_RETRAIN.inc()
    log.warning("Drift above threshold: starting automatic retraining")

    def trigger():
        try:
            requests.post(f"{settings.trainer_url}/run", json={"action": "full"}, timeout=10)
        except requests.RequestException as exc:
            log.error("Automatic retraining could not start: %s", exc)
    threading.Thread(target=trigger, daemon=True).start()


@app.post("/feedback", dependencies=[Depends(verify_key)])
def post_feedback(req: FeedbackRequest):
    """User says whether a prediction was right; feeds the live-accuracy gauge."""
    return feedback.add(req.correct)


@app.get("/feedback/stats", dependencies=[Depends(verify_key)])
def feedback_stats():
    return feedback.stats()


@app.post("/explain", dependencies=[Depends(verify_key)])
def explain(req: ExplainRequest):
    """Global explanation: which words/word-pairs of this text are typical for the predicted class."""
    try:
        features = mlflow_client.top_features()
    except Exception as exc:
        log.error("Could not load top features: %s", exc)
        raise HTTPException(status_code=503, detail="Explanation unavailable")
    top = features.get(req.category, [])
    words = tokens(req.text)
    candidates = set(words) | {f"{a} {b}" for a, b in zip(words, words[1:])}
    return {"category": req.category, "matched_terms": [t for t in top if t in candidates], "top_terms": top[:15]}


@app.get("/pipeline/summary", dependencies=[Depends(verify_key)])
def pipeline_summary():
    try:
        return mlflow_client.pipeline_summary()
    except Exception as exc:
        log.error("MLflow query failed: %s", exc)
        raise HTTPException(status_code=503, detail="MLflow unavailable")


@app.get("/model/info", dependencies=[Depends(verify_key)])
def model_info():
    try:
        info = mlflow_client.champion_info()
    except Exception as exc:
        log.error("Could not read champion model: %s", exc)
        raise HTTPException(status_code=503, detail="No champion model available")
    metrics.MODEL_VERSION.set(info["version"])
    return info


@app.get("/drift/status", dependencies=[Depends(verify_key)])
def drift_status():
    return drift.compute()


@app.get("/pipeline/runs", dependencies=[Depends(verify_key)])
def pipeline_runs(limit: int = 20):
    try:
        return mlflow_client.recent_runs(limit)
    except Exception as exc:
        log.error("MLflow query failed: %s", exc)
        raise HTTPException(status_code=503, detail="MLflow unavailable")


def trainer_call(method: str, path: str, **kwargs):
    try:
        resp = requests.request(method, f"{settings.trainer_url}{path}", timeout=10, **kwargs)
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="Trainer service unavailable")
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.json().get("detail", "Trainer error"))
    return resp.json()


@app.get("/pipeline/status", dependencies=[Depends(verify_key)])
def pipeline_status():
    return trainer_call("GET", "/status")


@app.get("/pipeline/jobs", dependencies=[Depends(verify_key)])
def pipeline_jobs():
    return trainer_call("GET", "/jobs")


@app.post("/pipeline/trigger", status_code=202, dependencies=[Depends(verify_key)])
def pipeline_trigger(req: TriggerRequest):
    log.info("Pipeline trigger requested: %s", req.action)
    return trainer_call("POST", "/run", json={"action": req.action})


@app.get("/metrics")
def prometheus_metrics():
    component_status()
    drift.compute()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
