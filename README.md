# AutoRoute - AIOps Complaint Classifier (DA3408)

Reads a customer complaint and routes it to a product category. Built with Ray, DVC, MLflow,
FastAPI, Streamlit, Prometheus, Grafana and Docker Compose.

**Follow the PDF guide (AutoRoute_Step_by_Step_Guide.pdf) - this README is only a cheat sheet.**

## Quick start (after the one-time setup in the guide)
```bash
cp .env.example .env                 # then put your `id -u` / `id -g` numbers inside
docker compose up -d --build mlflow  # experiment tracker on http://localhost:5000
source venv/bin/activate
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/make_sample_data.py       # fake data for a dry run (skip when you have real data)
dvc repro                            # test -> ingest -> validate -> prepare -> train -> evaluate
docker compose up -d --build         # everything else
```
| Service | URL |
|---|---|
| Web portal | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

## Layout
`src/` pipeline stages | `gateway/` FastAPI | `trainer/` job runner | `frontend/` Streamlit |
`serving/` MLflow model-server entrypoint | `monitoring/` Prometheus + Grafana | `tests/` pytest |
`docs/` design documents (fill these in; 07_DATA_CARD.md too) | `scripts/` replay + rollback helpers
