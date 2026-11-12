"""Generates monitoring/grafana/dashboards/autoroute.json (run once: python monitoring/make_dashboard.py)."""
import json

DS = {"type": "prometheus", "uid": "prometheus"}


def panel(pid, title, exprs, x, y, w=8, h=7, kind="timeseries", unit=None, thresholds=None):
    p = {"id": pid, "title": title, "type": kind, "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
         "targets": [{"refId": chr(65 + i), "expr": e, "legendFormat": l, "datasource": DS}
                     for i, (e, l) in enumerate(exprs)],
         "fieldConfig": {"defaults": {}, "overrides": []}}
    if unit:
        p["fieldConfig"]["defaults"]["unit"] = unit
    if thresholds:
        p["fieldConfig"]["defaults"]["thresholds"] = {"mode": "absolute", "steps": thresholds}
        p["fieldConfig"]["defaults"]["custom"] = {"thresholdsStyle": {"mode": "line"}}
    return p


GREEN_RED = [{"color": "green", "value": None}, {"color": "red", "value": 0.25}]
panels = [
    panel(1, "Predict requests / sec", [('sum(rate(requests_total{endpoint="/predict"}[1m]))', "req/s")], 0, 0),
    panel(2, "p95 latency (s)", [('histogram_quantile(0.95, sum(rate(request_latency_seconds_bucket{endpoint="/predict"}[1m])) by (le))', "p95")], 8, 0, unit="s"),
    panel(3, "Error responses / sec (5xx)", [('sum(rate(requests_total{status=~"5.."}[1m]))', "5xx")], 16, 0),
    panel(4, "Predictions per class (last 5m)", [('sum by (category) (increase(predictions_total[5m]))', "{{category}}")], 0, 7),
    panel(5, "Median confidence", [('histogram_quantile(0.5, sum(rate(prediction_confidence_bucket[2m])) by (le))', "median")], 8, 7),
    panel(6, "Low-confidence predictions (5m)", [('increase(low_confidence_total[5m])', "low conf")], 16, 7),
    panel(7, "Prediction drift (PSI)", [('psi_prediction', "PSI")], 0, 14, thresholds=GREEN_RED),
    panel(8, "Unknown-word rate (data drift)", [('oov_rate', "OOV")], 8, 14, unit="percentunit"),
    panel(9, "Text length KS p-value (data drift)", [('length_ks_pvalue', "p-value")], 16, 14),
    panel(10, "Drift buffer ready (1 = yes)", [('drift_buffer_ready', "ready")], 0, 21, w=6, h=5, kind="stat"),
    panel(11, "Model version", [('model_version', "version")], 6, 21, w=6, h=5, kind="stat"),
    panel(12, "Component up (1 = up)", [('component_up', "{{component}}")], 12, 21, w=6, h=5, kind="stat"),
    panel(13, "Pipeline jobs", [('trainer_jobs_total', "{{status}}")], 18, 21, w=6, h=5, kind="stat"),
    panel(14, "User-confirmed accuracy (recent feedback)", [('feedback_accuracy', "accuracy")], 0, 26, unit="percentunit"),
    panel(15, "Feedback answers", [('sum by (result) (feedback_total)', "{{result}}")], 8, 26),
    panel(16, "Memory (RSS) per service", [('process_resident_memory_bytes{job=~"gateway|trainer"}', "{{job}}")], 16, 26, unit="bytes"),
    panel(17, "CPU use per service (cores)", [('rate(process_cpu_seconds_total{job=~"gateway|trainer"}[1m])', "{{job}}")], 0, 33),
    panel(18, "Automatic retraining triggers", [('auto_retrain_triggers_total', "triggers")], 8, 33, w=8, h=7, kind="stat"),
]
dash = {"uid": "autoroute", "title": "AutoRoute Monitoring", "schemaVersion": 39, "version": 1,
        "refresh": "5s", "time": {"from": "now-15m", "to": "now"}, "panels": panels}
with open("monitoring/grafana/dashboards/autoroute.json", "w") as f:
    json.dump(dash, f, indent=1)
print("dashboard written")
