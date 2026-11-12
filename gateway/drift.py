"""Drift detection on live traffic.

DATA drift      : out-of-vocabulary (OOV) rate + KS test on text length vs training.
PREDICTION drift: Population Stability Index (PSI) of predicted classes vs training prior.
(Concept drift would need true labels, which we do not have in production.)
"""
import json
import logging
import math
import threading
from collections import Counter, deque

from scipy.stats import ks_2samp

from gateway import metrics
from text_utils import tokens

log = logging.getLogger("drift")


def psi(expected: dict, actual_counts: dict, classes, eps: float = 1e-4) -> float:
    """PSI = sum((a - e) * ln(a / e)). eps avoids log(0) for empty classes."""
    total = sum(actual_counts.values())
    if total == 0:
        return 0.0
    score = 0.0
    for c in classes:
        e = max(expected.get(c, 0.0), eps)
        a = max(actual_counts.get(c, 0) / total, eps)
        score += (a - e) * math.log(a / e)
    return score


class DriftMonitor:
    def __init__(self, baseline: dict, window: int = 100, min_buffer: int = 50):
        self.vocab = set(baseline.get("vocab", []))
        self.prior = baseline.get("class_prior", {})
        self.classes = baseline.get("classes", list(self.prior))
        self.base_lengths = baseline.get("length_sample", [])
        self.min_buffer = min_buffer
        self.labels = deque(maxlen=window)
        self.lengths = deque(maxlen=window)
        self.oov = deque(maxlen=window)
        self.lock = threading.Lock()
        self.enabled = bool(self.prior and self.vocab)

    @classmethod
    def from_file(cls, path: str, window: int, min_buffer: int):
        try:
            with open(path) as f:
                return cls(json.load(f), window, min_buffer)
        except Exception as exc:   # gateway must still run without a baseline
            log.warning("Drift baseline not loaded (%s). Drift monitoring disabled.", exc)
            return cls({}, window, min_buffer)

    def record(self, text: str, predicted: str) -> dict:
        """Add one request to the rolling window and refresh the Prometheus gauges."""
        if not self.enabled:
            return {"enabled": False}
        words = tokens(text)
        oov = (sum(1 for w in words if w not in self.vocab) / len(words)) if words else 0.0
        with self.lock:
            self.labels.append(predicted)
            self.lengths.append(len(words))
            self.oov.append(oov)
        return self.compute()

    def compute(self) -> dict:
        """Current drift statistics. Gauges are only updated once the buffer is ready."""
        if not self.enabled:
            return {"enabled": False, "buffer_ready": False}
        with self.lock:
            n = len(self.labels)
            ready = n >= self.min_buffer
            result = {"enabled": True, "buffer_size": n, "buffer_ready": ready,
                      "psi_prediction": None, "oov_rate": None, "length_ks_pvalue": None}
            if ready:
                result["psi_prediction"] = psi(self.prior, Counter(self.labels), self.classes)
                result["oov_rate"] = sum(self.oov) / n
                if self.base_lengths:
                    result["length_ks_pvalue"] = float(ks_2samp(list(self.lengths), self.base_lengths).pvalue)
        metrics.BUFFER_READY.set(1 if ready else 0)
        if ready:
            metrics.PSI_PREDICTION.set(result["psi_prediction"])
            metrics.OOV_RATE.set(result["oov_rate"])
            if result["length_ks_pvalue"] is not None:
                metrics.LENGTH_KS_P.set(result["length_ks_pvalue"])
        return result
