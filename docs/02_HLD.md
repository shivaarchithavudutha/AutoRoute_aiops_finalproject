# High-Level Design (HLD)
1. **Problem and goals** - copy your approved project statement.
2. **Users and use cases** - support agent classifies a complaint; ML engineer retrains and monitors.
3. **Architecture overview** - reference the diagram; explain data flow for (a) a prediction, (b) a retraining run, (c) a drift alert.
4. **Technology choices and WHY** - for each tool: what it does, why chosen, one alternative you rejected.
   (Examples: TF-IDF + linear model: fast, explainable, light containers. Ray: required parallel engine, scales out.
   DVC: versions data + defines the pipeline DAG. MLflow serve + FastAPI gateway: separation of model deployment and business logic.)
5. **Loose coupling** - frontend only knows `API_URL`; backend never imports frontend code.
6. **Data design** - dataset, label mapping, split strategy (stratified, capped classes), drift baseline contents.
7. **Model lifecycle** - train -> evaluate gate (macro F1 >= 0.75) -> promote alias -> auto-serve; rollback by moving the alias.
8. **Monitoring design** - metrics list; data drift (OOV, KS) vs prediction drift (PSI); thresholds and alerts.
9. **Security** - optional API key header, no secrets in Git, input length limits.
10. **Deployment** - Docker Compose services, memory limits, shared volumes.
11. **Deployment strategy** - online (real-time REST) because a support agent needs an answer immediately; batch scoring is possible through the same API. State why edge deployment was not chosen.
12. **Data quality** - `validate` stage with hard checks (rows, classes, nulls, short/duplicate text) and warnings (imbalance, dates); bad data stops the pipeline.
13. **CI inside DVC** - first stage `test` runs flake8 + pytest; downstream stages depend on its summary, so failing code never trains a model.
14. **Explainability** - top words per class stored as an MLflow artifact; the portal shows which words of the user's text match them. Limitation: global, not per-prediction (LIME/SHAP are future work).
15. **Security** - API key, input validation, optional encryption of the model artifact (Fernet, key from environment). Not implemented: HTTPS, encryption of data at rest.
16. **Scalability** - Ray scales ingest across cores/machines; services are separate containers; limits: single gateway worker, single node. Describe what you would change (more workers, load balancer, Kubernetes).
17. **Production performance monitoring** - confidence statistics plus user feedback (Yes/No) turned into a rolling accuracy gauge; drift metrics; alert rules.
18. **Automation of retraining** - Retrain button, `/pipeline/trigger`, optional automatic trigger when PSI is high (AUTO_RETRAIN_ON_DRIFT), cooldown to avoid loops.
19. **Risks and limitations** - imbalance, opt-in narratives, PSI noise on small windows, no concept-drift measurement.
