"""Read-only helpers that query MLflow for the Pipeline Console."""
import json
from datetime import datetime, timezone

from mlflow.tracking import MlflowClient

from gateway.config import settings


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=settings.mlflow_uri)


def champion_info() -> dict:
    """Which model version is currently the champion, and how good is it."""
    client = _client()
    mv = client.get_model_version_by_alias(settings.model_name, settings.alias)
    run = client.get_run(mv.run_id)
    return {"model_name": settings.model_name, "alias": settings.alias, "version": int(mv.version),
            "run_id": mv.run_id, "macro_f1": run.data.metrics.get("macro_f1"),
            "accuracy": run.data.metrics.get("accuracy"),
            "trained_at": datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc).isoformat()}


def recent_runs(limit: int = 20) -> list:
    """Most recent training runs with status, commit, F1 and duration."""
    client = _client()
    exp = client.get_experiment_by_name(settings.experiment)
    if exp is None:
        return []
    rows = []
    for r in client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=limit):
        end = r.info.end_time or r.info.start_time
        rows.append({"run_id": r.info.run_id[:8], "model": r.info.run_name, "status": r.info.status,
                     "commit": r.data.tags.get("git_commit", "unknown")[:8],
                     "macro_f1": r.data.metrics.get("macro_f1"),
                     "duration_s": round((end - r.info.start_time) / 1000, 1),
                     "started": datetime.fromtimestamp(r.info.start_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")})
    return rows


_features_cache = {}


def top_features() -> dict:
    """Most indicative words per class for the champion model (global explanation)."""
    client = _client()
    mv = client.get_model_version_by_alias(settings.model_name, settings.alias)
    if mv.version not in _features_cache:
        path = client.download_artifacts(mv.run_id, "top_features.json")
        with open(path) as f:
            _features_cache[mv.version] = json.load(f)
    return _features_cache[mv.version]


def pipeline_summary() -> dict:
    """Speed/throughput and data-quality numbers from the latest training run."""
    client = _client()
    exp = client.get_experiment_by_name(settings.experiment)
    if exp is None:
        return {}
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        return {}
    m = runs[0].data.metrics
    keep = {k: v for k, v in m.items() if k.startswith(("ingest_", "dq_", "ci_")) or k in ("train_seconds", "macro_f1")}
    return {"run_id": runs[0].info.run_id[:8], "metrics": keep}
