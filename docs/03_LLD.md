# Low-Level Design (LLD): API Specification

Base URL: `http://<host>:8000`. If `API_KEY` is set, send header `X-API-Key`.

| Endpoint | Method | Request | Success response | Errors |
|---|---|---|---|---|
| `/health` | GET | none | `{"status":"ok","components":{"model_server":true,"mlflow":true,"trainer":true}}` | none (status says `degraded`) |
| `/predict` | POST | `{"text": "<1-5000 chars>"}` | `{"category":"Debt collection","confidence":0.93,"top3":[{"category":"...","score":0.93}, ...]}` | 422 invalid text; 401 bad key; 502 bad model-server reply; 503 model server down |
| `/model/info` | GET | none | `{"model_name","alias","version","run_id","macro_f1","accuracy","trained_at"}` | 503 no champion |
| `/drift/status` | GET | none | `{"enabled","buffer_size","buffer_ready","psi_prediction","oov_rate","length_ks_pvalue"}` | none |
| `/pipeline/runs` | GET | `?limit=20` | list of `{run_id, model, status, commit, macro_f1, duration_s, started}` | 503 MLflow down |
| `/pipeline/jobs` | GET | none | list of trainer jobs `{job_id, action, status, started, finished}` | 503 trainer down |
| `/pipeline/status` | GET | none | latest job incl. `log_tail` | 503 trainer down |
| `/pipeline/trigger` | POST | `{"action":"retrain"|"full"}` | 202 `{"job_id","status":"running"}` | 409 job running; 422 bad action; 503 trainer down |
| `/feedback` | POST | `{"predicted":"Mortgage","correct":true}` | `{"window":12,"accuracy":0.83}` | 422 invalid body; 401 bad key |
| `/feedback/stats` | GET | none | `{"window":12,"accuracy":0.83}` | none |
| `/explain` | POST | `{"text":"...","category":"Debt collection"}` | `{"category","matched_terms":[...],"top_terms":[...]}` | 503 explanation unavailable |
| `/pipeline/summary` | GET | none | `{"run_id","metrics":{"ingest_ray_full_rows_per_sec":...,"train_seconds":...,"dq_rows":...}}` | 503 MLflow down |
| `/metrics` | GET | none | Prometheus text format | none |

**Model server (MLflow):** `POST :5001/invocations` with
`{"dataframe_split":{"columns":["narrative"],"data":[["text"]]}}` returns `{"predictions":[{"<class>":prob, ...}]}`.

**Trainer:** `POST :8100/run {"action": ...}`, `GET /status`, `GET /jobs`, `GET /metrics`, `GET /health`.

**Prometheus metrics:** requests_total, request_latency_seconds, predictions_total, prediction_confidence,
low_confidence_total, feedback_total, feedback_accuracy, feedback_window, auto_retrain_triggers_total, oov_rate, length_ks_pvalue, psi_prediction, drift_buffer_ready, model_version,
component_up (model_server, mlflow, trainer, frontend), trainer_jobs_total, trainer_running, plus default process_* CPU/memory metrics.

TODO: add class diagram / module list, sequence diagram for `/predict`, and error-handling table.

**Pipeline stages (dvc.yaml):** test (flake8 + pytest, writes reports/ci_summary.json) -> ingest (Ray) -> validate (data-quality gate,
writes data/interim/data_quality.json; exits 1 on a hard failure) -> prepare -> train -> evaluate (promotion gate F1 >= 0.75).
