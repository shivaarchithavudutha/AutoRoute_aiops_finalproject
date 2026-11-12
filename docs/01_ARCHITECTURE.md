# Architecture Diagram and Block Explanation

(Draw this as a picture with draw.io / diagrams.net and export PNG into docs/. Use the blocks below.)

```
CFPB CSV -> [Ray ingest] -> [DVC: prepare -> train -> evaluate] -> MLflow (tracking + registry)
                                                                        |
Streamlit UI --REST (API_URL)--> FastAPI gateway --REST--> MLflow model server
                                      |  /metrics                 ^ (auto-switches to 'champion')
                                 Prometheus --> Grafana
Gateway --REST--> Trainer service --> dvc repro
```

| Block | Technology | Responsibility |
|---|---|---|
| Data pipeline | Ray | parallel text cleaning, throughput benchmark |
| Pipeline orchestration | DVC | stages, DAG, data/model versioning |
| Experiment tracking + registry | MLflow | params, metrics, artifacts, model versions, `champion` alias |
| Model server | `mlflow models serve` | exposes the registered model at /invocations |
| Gateway | FastAPI | validation, auth, telemetry, drift, pipeline endpoints |
| Trainer | FastAPI + DVC | runs the pipeline on request |
| Frontend | Streamlit | classifier page, pipeline console, health/drift page |
| Monitoring | Prometheus + Grafana | metrics, dashboards, alert rules |
| Packaging | Docker Compose | separate services, memory limits |

TODO: add one paragraph per block in your own words.
