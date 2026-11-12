"""Prometheus metrics exposed at /metrics."""
from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5))
PREDICTIONS = Counter("predictions_total", "Predictions by class", ["category"])
CONFIDENCE = Histogram("prediction_confidence", "Top-class probability",
                       buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
LOW_CONFIDENCE = Counter("low_confidence_total", "Predictions below the confidence threshold")
OOV_RATE = Gauge("oov_rate", "Rolling out-of-vocabulary word rate (data drift)")
LENGTH_KS_P = Gauge("length_ks_pvalue", "KS-test p-value of text length vs training (data drift)")
PSI_PREDICTION = Gauge("psi_prediction", "PSI of predicted classes vs training prior (prediction drift)")
BUFFER_READY = Gauge("drift_buffer_ready", "1 once enough requests are buffered for drift stats")
MODEL_VERSION = Gauge("model_version", "Champion model version served")
COMPONENT_UP = Gauge("component_up", "1 if a component answers its health check", ["component"])
FEEDBACK = Counter("feedback_total", "User feedback on predictions", ["result"])
FEEDBACK_ACCURACY = Gauge("feedback_accuracy", "Share of recent predictions users confirmed as correct")
FEEDBACK_WINDOW = Gauge("feedback_window", "Number of feedback items in the rolling window")
AUTO_RETRAIN = Counter("auto_retrain_triggers_total", "Retraining jobs started automatically after drift")
