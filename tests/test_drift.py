import pytest

from gateway.drift import DriftMonitor, psi

BASE = {"classes": ["a", "b"], "class_prior": {"a": 0.5, "b": 0.5},
        "vocab": ["debt", "call", "loan"], "length_sample": [3] * 200}


def test_psi_zero_for_identical_distribution():
    assert psi({"a": 0.5, "b": 0.5}, {"a": 50, "b": 50}, ["a", "b"]) == pytest.approx(0, abs=1e-9)


def test_psi_large_for_shifted_distribution():
    assert psi({"a": 0.5, "b": 0.5}, {"a": 95, "b": 5}, ["a", "b"]) > 0.25


def test_psi_handles_empty_class_without_error():
    assert psi({"a": 0.5, "b": 0.5}, {"a": 100}, ["a", "b"]) > 0


def test_buffer_not_ready_until_min_buffer():
    mon = DriftMonitor(BASE, window=100, min_buffer=5)
    for i in range(4):
        result = mon.record("debt call loan", "a")
    assert result["buffer_ready"] is False and result["psi_prediction"] is None
    result = mon.record("debt call loan", "b")
    assert result["buffer_ready"] is True and result["psi_prediction"] is not None


def test_oov_rate_counts_unknown_words():
    mon = DriftMonitor(BASE, window=10, min_buffer=1)
    result = mon.record("debt zzz qqq call", "a")      # 2 of 4 words unknown
    assert result["oov_rate"] == pytest.approx(0.5)


def test_disabled_without_baseline():
    mon = DriftMonitor({}, window=10, min_buffer=1)
    assert mon.record("hello", "a") == {"enabled": False}
