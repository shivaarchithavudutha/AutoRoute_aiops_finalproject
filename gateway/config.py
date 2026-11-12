"""Gateway settings, all read from environment variables (set in docker-compose.yml)."""
import os


class Settings:
    model_server_url = os.environ.get("MODEL_SERVER_URL", "http://localhost:5001")
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    trainer_url = os.environ.get("TRAINER_URL", "http://localhost:8100")
    baseline_path = os.environ.get("BASELINE_PATH", "data/processed/baseline.json")
    model_name = os.environ.get("MODEL_NAME", "AutoRouteModel")
    experiment = os.environ.get("MLFLOW_EXPERIMENT", "autoroute")
    alias = os.environ.get("MODEL_ALIAS", "champion")
    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:8501")
    auto_retrain = os.environ.get("AUTO_RETRAIN_ON_DRIFT", "false").lower() == "true"
    auto_retrain_psi = float(os.environ.get("AUTO_RETRAIN_PSI", "0.25"))
    auto_retrain_cooldown_s = int(os.environ.get("AUTO_RETRAIN_COOLDOWN_S", "3600"))
    feedback_window = int(os.environ.get("FEEDBACK_WINDOW", "50"))
    api_key = os.environ.get("API_KEY", "")            # empty = auth disabled
    drift_window = int(os.environ.get("DRIFT_WINDOW", "100"))
    drift_min_buffer = int(os.environ.get("DRIFT_MIN_BUFFER", "50"))
    low_confidence = float(os.environ.get("LOW_CONFIDENCE", "0.5"))


settings = Settings()
