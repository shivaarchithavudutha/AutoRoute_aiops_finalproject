"""STAGE 3 - train: fit TF-IDF + linear model, log everything to MLflow, register the model.

Run:  python src/train.py
Needs MLFLOW_TRACKING_URI (default http://localhost:5000).
"""
import sys
import json
import os
import tempfile
import time

import joblib
import matplotlib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common import file_md5, get_logger, git_commit, load_params
from model_wrapper import AutoRouteModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = get_logger("train")


def build_pipeline(params: dict) -> Pipeline:
    f, t = params["features"], params["train"]
    vec = TfidfVectorizer(max_features=f["max_features"], ngram_range=(f["ngram_min"], f["ngram_max"]),
                          sublinear_tf=True)
    kind = t["model_type"]
    if kind == "logistic_regression":
        clf = LogisticRegression(C=t["C"], class_weight=t["class_weight"], max_iter=300)
    elif kind == "linear_svm":   # calibrate so we get probabilities
        clf = CalibratedClassifierCV(LinearSVC(C=t["C"], class_weight=t["class_weight"]), cv=3)
    elif kind == "naive_bayes":
        clf = MultinomialNB(alpha=t["C"])
    else:
        raise ValueError(f"Unknown model_type: {kind}")
    return Pipeline([("tfidf", vec), ("clf", clf)])


def top_features(pipe: Pipeline, k: int = 30) -> dict:
    """Most indicative words per class (simple explainability)."""
    try:
        names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "coef_"):
            weights = clf.coef_
        elif hasattr(clf, "feature_log_prob_"):
            weights = clf.feature_log_prob_
        else:
            weights = np.mean([c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0)
        return {str(c): names[np.argsort(weights[i])[-k:][::-1]].tolist()
                for i, c in enumerate(pipe.classes_)}
    except Exception as exc:   # explainability must never break training
        log.warning("Could not compute top features: %s", exc)
        return {}


def save_confusion_matrix(y_true, y_pred, classes, path):
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_title("Confusion matrix (row-normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    params = load_params()
    paths, mcfg = params["paths"], params["mlflow"]
    train, test = pd.read_parquet(paths["train"]), pd.read_parquet(paths["test"])

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(mcfg["experiment"])
    client = MlflowClient()

    with mlflow.start_run(run_name=f"{params['train']['model_type']}") as run:
        mlflow.set_tags({"git_commit": git_commit(),
                         "data_md5_train": file_md5(paths["train"]),
                         "stage": "train"})
        mlflow.log_params({**{f"features.{k}": v for k, v in params["features"].items()},
                           **{f"train.{k}": v for k, v in params["train"].items()},
                           "n_train": len(train), "n_test": len(test)})

        pipe = build_pipeline(params)
        start = time.perf_counter()
        pipe.fit(train["text"], train["label"])
        train_seconds = time.perf_counter() - start
        pred = pipe.predict(test["text"])

        metrics = {"accuracy": accuracy_score(test["label"], pred),
                   "macro_f1": f1_score(test["label"], pred, average="macro"),
                   "weighted_f1": f1_score(test["label"], pred, average="weighted"),
                   "train_seconds": train_seconds}
        report = classification_report(test["label"], pred, output_dict=True, zero_division=0)
        for cls in pipe.classes_:
            metrics[f"f1_{cls}"] = report[cls]["f1-score"]
        mlflow.log_metrics({k.replace(" ", "_"): float(v) for k, v in metrics.items()})
        log.info("macro_f1=%.4f accuracy=%.4f", metrics["macro_f1"], metrics["accuracy"])

        # Extra information beyond autolog (rubric item).
        ing = "data/interim/ingest_metrics.json"
        if os.path.exists(ing):
            with open(ing) as f:
                mlflow.log_metrics({f"ingest_{k}": float(v) for k, v in json.load(f).items()})
        for path, prefix in (("data/interim/data_quality.json", "dq"), ("reports/ci_summary.json", "ci")):
            if os.path.exists(path):
                with open(path) as f:
                    info = json.load(f)
                mlflow.log_metrics({f"{prefix}_{k}": float(v) for k, v in info.items()
                                    if isinstance(v, (int, float, bool))})
                mlflow.log_artifact(path, artifact_path="quality")
        mlflow.log_dict({"class_distribution": train["label"].value_counts().to_dict()}, "class_distribution.json")
        mlflow.log_dict(top_features(pipe), "top_features.json")
        mlflow.log_dict(classification_report(test["label"], pred, output_dict=True, zero_division=0),
                        "classification_report.json")
        mlflow.log_dict({"classes": [str(c) for c in pipe.classes_]}, "classes.json")
        mlflow.log_artifact("params.yaml")
        if os.path.exists(paths["baseline"]):
            mlflow.log_artifact(paths["baseline"], artifact_path="drift_baseline")

        with tempfile.TemporaryDirectory() as tmp:
            png = os.path.join(tmp, "confusion_matrix.png")
            save_confusion_matrix(test["label"], pred, [str(c) for c in pipe.classes_], png)
            mlflow.log_artifact(png)

            # Package model: pyfunc wrapper + the fitted sklearn pipeline.
            pkl = os.path.join(tmp, "pipeline.joblib")
            joblib.dump(pipe, pkl)
            key = os.environ.get("MODEL_ENCRYPTION_KEY", "")
            if key:   # optional security: encrypt the model file before it is stored in MLflow
                from cryptography.fernet import Fernet
                with open(pkl, "rb") as f:
                    token = Fernet(key.encode()).encrypt(f.read())
                pkl = pkl + ".enc"
                with open(pkl, "wb") as f:
                    f.write(token)
                mlflow.set_tag("model_encrypted", "true")
                log.info("Model artifact encrypted with MODEL_ENCRYPTION_KEY")
            sample_in = pd.DataFrame({"narrative": ["sample complaint text"]})
            sample_out = pd.DataFrame(pipe.predict_proba(["sample complaint text"]),
                                      columns=[str(c) for c in pipe.classes_])
            mlflow.pyfunc.log_model(
                artifact_path="model", python_model=AutoRouteModel(),
                artifacts={"sklearn_pipeline": pkl},
                code_paths=["src/model_wrapper.py", "src/text_utils.py"],
                signature=infer_signature(sample_in, sample_out),
                registered_model_name=mcfg["model_name"],
                pip_requirements=["mlflow==2.22.0", "scikit-learn==1.5.2", "pandas==2.2.3",
                                  "numpy==1.26.4", "joblib==1.4.2", "cryptography"])

        versions = client.search_model_versions(f"run_id='{run.info.run_id}'")
        version = max(int(v.version) for v in versions) if versions else None
        os.makedirs("models", exist_ok=True)
        with open("models/run_info.json", "w") as f:
            json.dump({"run_id": run.info.run_id, "model_version": version}, f, indent=2)
        with open("metrics.json", "w") as f:
            json.dump({"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
                       "weighted_f1": metrics["weighted_f1"], "train_seconds": train_seconds}, f, indent=2)
        log.info("Registered %s version %s (run %s)", mcfg["model_name"], version, run.info.run_id)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("train stage crashed")
        sys.exit(1)
