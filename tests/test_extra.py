"""Extra tests written for the test plan: API edge cases, drift statistics and helpers."""
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from common import map_label
from gateway import main, metrics
from gateway.drift import DriftMonitor, psi
from text_utils import clean_text

client = TestClient(main.app)


class FakeResponse:
    """Stand-in for a requests.Response coming from the model server."""

    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code = status
        self._payload = payload
        self._bad_json = bad_json
        self.text = "fake"

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


def fake_model_factory(probs):
    return lambda text: dict(probs)


# ---------- /predict behaviour ----------
def test_predict_returns_highest_probability_first(monkeypatch):
    monkeypatch.setattr(main, "call_model_server", fake_model_factory({"A": 0.1, "B": 0.6, "C": 0.3}))
    body = client.post("/predict", json={"text": "hello"}).json()
    assert body["category"] == "B"
    scores = [item["score"] for item in body["top3"]]
    assert scores == sorted(scores, reverse=True)


def test_predict_top3_has_three_items_when_many_classes(monkeypatch):
    monkeypatch.setattr(main, "call_model_server",
                        fake_model_factory({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}))
    assert len(client.post("/predict", json={"text": "hello"}).json()["top3"]) == 3


def test_predict_top3_has_two_items_when_two_classes(monkeypatch):
    monkeypatch.setattr(main, "call_model_server", fake_model_factory({"A": 0.7, "B": 0.3}))
    assert len(client.post("/predict", json={"text": "hello"}).json()["top3"]) == 2


def test_predict_accepts_text_of_exactly_5000_characters(monkeypatch):
    monkeypatch.setattr(main, "call_model_server", fake_model_factory({"A": 0.9, "B": 0.1}))
    assert client.post("/predict", json={"text": "a" * 5000}).status_code == 200


def test_predict_accepts_unicode_text(monkeypatch):
    monkeypatch.setattr(main, "call_model_server", fake_model_factory({"A": 0.9, "B": 0.1}))
    assert client.post("/predict", json={"text": "Caf\u00e9 \u20b9500 charged twice \u2014 please help"}).status_code == 200


def test_low_confidence_counter_increments(monkeypatch):
    monkeypatch.setattr(main, "call_model_server", fake_model_factory({"A": 0.3, "B": 0.3, "C": 0.4}))
    before = metrics.LOW_CONFIDENCE._value.get()
    client.post("/predict", json={"text": "unclear complaint"})
    assert metrics.LOW_CONFIDENCE._value.get() == before + 1


def test_model_server_error_status_becomes_502(monkeypatch):
    monkeypatch.setattr(main.requests, "post", lambda *a, **k: FakeResponse(status=500))
    assert client.post("/predict", json={"text": "hello"}).status_code == 502


def test_model_server_bad_json_becomes_502(monkeypatch):
    monkeypatch.setattr(main.requests, "post", lambda *a, **k: FakeResponse(bad_json=True))
    assert client.post("/predict", json={"text": "hello"}).status_code == 502


def test_model_server_empty_predictions_becomes_502(monkeypatch):
    monkeypatch.setattr(main.requests, "post", lambda *a, **k: FakeResponse(payload={"predictions": []}))
    assert client.post("/predict", json={"text": "hello"}).status_code == 502


# ---------- /model/info and /pipeline/* with mocked MLflow / trainer ----------
def test_model_info_returns_champion_details(monkeypatch):
    info = {"model_name": "AutoRouteModel", "alias": "champion", "version": 3, "run_id": "abc",
            "macro_f1": 0.82, "accuracy": 0.86, "trained_at": "2026-11-01T00:00:00+00:00"}
    monkeypatch.setattr(main.mlflow_client, "champion_info", lambda: info)
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["version"] == 3


def test_model_info_503_when_no_champion(monkeypatch):
    def boom():
        raise RuntimeError("no champion")
    monkeypatch.setattr(main.mlflow_client, "champion_info", boom)
    assert client.get("/model/info").status_code == 503


def test_pipeline_runs_returns_list_and_passes_limit(monkeypatch):
    seen = {}

    def fake_runs(limit):
        seen["limit"] = limit
        return [{"run_id": "abc12345", "status": "FINISHED", "macro_f1": 0.8}]
    monkeypatch.setattr(main.mlflow_client, "recent_runs", fake_runs)
    response = client.get("/pipeline/runs?limit=5")
    assert response.status_code == 200 and response.json()[0]["status"] == "FINISHED"
    assert seen["limit"] == 5


def test_pipeline_runs_503_when_mlflow_down(monkeypatch):
    def boom(limit):
        raise RuntimeError("mlflow down")
    monkeypatch.setattr(main.mlflow_client, "recent_runs", boom)
    assert client.get("/pipeline/runs").status_code == 503


def test_pipeline_trigger_forwards_to_trainer(monkeypatch):
    calls = []

    def fake_trainer_call(method, path, **kwargs):
        calls.append((method, path, kwargs.get("json")))
        return {"job_id": "x1", "status": "running"}
    monkeypatch.setattr(main, "trainer_call", fake_trainer_call)
    response = client.post("/pipeline/trigger", json={"action": "retrain"})
    assert response.status_code == 202
    assert calls == [("POST", "/run", {"action": "retrain"})]


def test_pipeline_trigger_passes_through_409_when_busy(monkeypatch):
    def busy(method, path, **kwargs):
        raise HTTPException(status_code=409, detail="A job is already running")
    monkeypatch.setattr(main, "trainer_call", busy)
    assert client.post("/pipeline/trigger", json={"action": "full"}).status_code == 409


# ---------- drift statistics ----------
BASE = {"classes": ["a", "b"], "class_prior": {"a": 0.5, "b": 0.5}, "vocab": ["debt", "call", "loan"],
        "length_sample": [20] * 200}


def test_ks_pvalue_high_when_lengths_match_training():
    monitor = DriftMonitor(BASE, window=100, min_buffer=50)
    text = " ".join(["debt"] * 20)
    for _ in range(60):
        result = monitor.record(text, "a")
    assert result["length_ks_pvalue"] > 0.05


def test_ks_pvalue_low_when_lengths_shift():
    monitor = DriftMonitor(BASE, window=100, min_buffer=50)
    text = " ".join(["debt"] * 80)
    for _ in range(60):
        result = monitor.record(text, "a")
    assert result["length_ks_pvalue"] < 0.001


def test_oov_rate_zero_for_known_words_and_one_for_unknown_words():
    monitor = DriftMonitor(BASE, window=10, min_buffer=1)
    assert monitor.record("debt call loan", "a")["oov_rate"] == pytest.approx(0.0)
    monitor = DriftMonitor(BASE, window=10, min_buffer=1)
    assert monitor.record("zzz qqq xxx", "a")["oov_rate"] == pytest.approx(1.0)


def test_drift_window_keeps_only_most_recent_requests():
    monitor = DriftMonitor(BASE, window=100, min_buffer=50)
    for _ in range(150):
        result = monitor.record("debt call", "a")
    assert result["buffer_size"] == 100


def test_psi_grows_as_distribution_skews_more():
    prior = {"a": 0.5, "b": 0.5}
    mild = psi(prior, {"a": 60, "b": 40}, ["a", "b"])
    medium = psi(prior, {"a": 80, "b": 20}, ["a", "b"])
    severe = psi(prior, {"a": 95, "b": 5}, ["a", "b"])
    assert mild < medium < severe


# ---------- helpers ----------
def test_clean_text_handles_accents_and_numbers():
    assert clean_text("Caf\u00e9 123!!") == "caf 123"


def test_map_label_strips_whitespace():
    assert map_label("  Debt collection  ", {"Debt collection": "Debt collection"}) == "Debt collection"
