from fastapi.testclient import TestClient

from gateway import main

client = TestClient(main.app)
FAKE = {"Debt collection": 0.8, "Mortgage": 0.15, "Credit card": 0.05}


def fake_model(text):
    return dict(FAKE)


def test_predict_ok(monkeypatch):
    monkeypatch.setattr(main, "call_model_server", fake_model)
    r = client.post("/predict", json={"text": "collector keeps calling"})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "Debt collection"
    assert body["confidence"] == 0.8
    assert len(body["top3"]) == 3


def test_predict_empty_text_rejected():
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_too_long_rejected():
    assert client.post("/predict", json={"text": "a" * 5001}).status_code == 422


def test_predict_missing_field_rejected():
    assert client.post("/predict", json={}).status_code == 422


def test_model_server_down_returns_503(monkeypatch):
    monkeypatch.setattr(main.settings, "model_server_url", "http://127.0.0.1:9")
    r = client.post("/predict", json={"text": "hello"})
    assert r.status_code == 503


def test_api_key_enforced(monkeypatch):
    monkeypatch.setattr(main, "call_model_server", fake_model)
    monkeypatch.setattr(main.settings, "api_key", "secret")
    assert client.post("/predict", json={"text": "hi"}).status_code == 401
    assert client.post("/predict", json={"text": "hi"}, headers={"X-API-Key": "secret"}).status_code == 200


def test_metrics_endpoint_exposes_custom_metrics(monkeypatch):
    monkeypatch.setattr(main, "call_model_server", fake_model)
    client.post("/predict", json={"text": "hello there"})
    body = client.get("/metrics").text
    for name in ("requests_total", "request_latency_seconds", "predictions_total",
                 "prediction_confidence", "drift_buffer_ready", "component_up"):
        assert name in body


def test_health_degraded_when_services_down(monkeypatch):
    for attr in ("model_server_url", "mlflow_uri", "trainer_url"):
        monkeypatch.setattr(main.settings, attr, "http://127.0.0.1:9")
    body = client.get("/health").json()
    assert body["status"] == "degraded"


def test_pipeline_trigger_rejects_bad_action():
    assert client.post("/pipeline/trigger", json={"action": "explode"}).status_code == 422


def test_pipeline_status_503_when_trainer_down(monkeypatch):
    monkeypatch.setattr(main.settings, "trainer_url", "http://127.0.0.1:9")
    assert client.get("/pipeline/status").status_code == 503
