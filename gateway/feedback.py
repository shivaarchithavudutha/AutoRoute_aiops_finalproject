"""Rolling accuracy from user feedback ("was this prediction correct?").

This is the only production signal that uses true labels, so it is the closest thing
we have to measuring real performance (and concept drift).
"""
import threading
from collections import deque

from gateway import metrics


class FeedbackTracker:
    def __init__(self, window: int = 50):
        self.items = deque(maxlen=window)
        self.lock = threading.Lock()

    def add(self, correct: bool) -> dict:
        with self.lock:
            self.items.append(bool(correct))
        metrics.FEEDBACK.labels("correct" if correct else "incorrect").inc()
        return self.stats()

    def stats(self) -> dict:
        with self.lock:
            n = len(self.items)
            acc = (sum(self.items) / n) if n else None
        metrics.FEEDBACK_WINDOW.set(n)
        if acc is not None:
            metrics.FEEDBACK_ACCURACY.set(acc)
        return {"window": n, "accuracy": acc}
