import joblib
import pandas as pd
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from gateway import main
from gateway.feedback import FeedbackTracker
from model_wrapper import AutoRouteModel

client = TestClient(main.app)


def test_feedback_tracker_rolling_accuracy():
    tracker = FeedbackTracker(window=4)
    for value in (True, True, False, True):
        stats = tracker.add(value)
    assert stats["window"] == 4 and stats["accuracy"] == pytest.approx(0.75)
    assert tracker.add(False)["accuracy"] == pytest.approx(0.5)     # oldest item dropped


def test_feedback_endpoint():
    assert client.post("/feedback", json={"predicted": "Mortgage", "correct": True}).status_code == 200
    assert client.get("/feedback/stats").json()["window"] >= 1


def test_feedback_endpoint_validation():
    assert client.post("/feedback", json={"predicted": "Mortgage"}).status_code == 422


def test_explain_returns_matched_terms(monkeypatch):
    monkeypatch.setattr(main.mlflow_client, "top_features",
                        lambda: {"Debt collection": ["debt collector", "collection", "calls"]})
    body = client.post("/explain", json={"text": "The debt collector calls daily", "category": "Debt collection"}).json()
    assert set(body["matched_terms"]) == {"debt collector", "calls"}


def test_explain_unavailable_returns_503(monkeypatch):
    def boom():
        raise RuntimeError("no mlflow")
    monkeypatch.setattr(main.mlflow_client, "top_features", boom)
    assert client.post("/explain", json={"text": "x", "category": "y"}).status_code == 503


def test_auto_retrain_respects_flag_and_cooldown(monkeypatch):
    calls = []
    monkeypatch.setattr(main.requests, "post", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(main.settings, "auto_retrain", False)
    main.maybe_auto_retrain({"buffer_ready": True, "psi_prediction": 5})
    assert main._last_auto_retrain[0] == 0.0 or not calls
    monkeypatch.setattr(main.settings, "auto_retrain", True)
    main._last_auto_retrain[0] = 0.0
    main.maybe_auto_retrain({"buffer_ready": False, "psi_prediction": 5})
    assert main._last_auto_retrain[0] == 0.0                         # buffer not ready: nothing
    main.maybe_auto_retrain({"buffer_ready": True, "psi_prediction": 5})
    first = main._last_auto_retrain[0]
    assert first > 0
    main.maybe_auto_retrain({"buffer_ready": True, "psi_prediction": 5})
    assert main._last_auto_retrain[0] == first                       # inside cooldown: no second trigger


def test_wrapper_decrypts_encrypted_model(tmp_path, monkeypatch):
    pipe = Pipeline([("t", TfidfVectorizer()), ("c", LogisticRegression())])
    pipe.fit(["debt call", "mortgage escrow", "debt collector", "mortgage rate"], ["d", "m", "d", "m"])
    plain = tmp_path / "p.joblib"
    joblib.dump(pipe, plain)
    key = Fernet.generate_key().decode()
    enc = tmp_path / "p.joblib.enc"
    enc.write_bytes(Fernet(key.encode()).encrypt(plain.read_bytes()))

    class Ctx:
        artifacts = {"sklearn_pipeline": str(enc)}

    model = AutoRouteModel()
    monkeypatch.delenv("MODEL_ENCRYPTION_KEY", raising=False)
    with pytest.raises(RuntimeError):
        model.load_context(Ctx())                                    # no key: refuse to load
    monkeypatch.setenv("MODEL_ENCRYPTION_KEY", key)
    model.load_context(Ctx())
    assert model.predict(None, pd.DataFrame({"narrative": ["debt call"]})).shape == (1, 2)
