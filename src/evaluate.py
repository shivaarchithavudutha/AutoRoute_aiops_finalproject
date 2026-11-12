"""STAGE 4 - evaluate: apply the acceptance criterion and promote the model.

A new model becomes 'champion' only if macro F1 >= evaluate.min_macro_f1 AND it beats
the current champion. Otherwise nothing changes (safe: the old model keeps serving).
Run:  python src/evaluate.py
"""
import sys
import json
import os

import mlflow
from mlflow.tracking import MlflowClient

from common import get_logger, load_params

log = get_logger("evaluate")


def main():
    params = load_params()
    mcfg, min_f1 = params["mlflow"], params["evaluate"]["min_macro_f1"]
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()

    with open("models/run_info.json") as f:
        info = json.load(f)
    with open("metrics.json") as f:
        new_f1 = json.load(f)["macro_f1"]

    old_f1, old_version = None, None
    try:
        champ = client.get_model_version_by_alias(mcfg["model_name"], mcfg["alias"])
        old_version = int(champ.version)
        old_f1 = client.get_run(champ.run_id).data.metrics.get("macro_f1")
    except Exception:
        log.info("No champion yet.")

    if new_f1 < min_f1:
        decision = f"REJECTED: macro_f1 {new_f1:.4f} < required {min_f1}"
    elif old_f1 is not None and new_f1 < old_f1 and old_version != info["model_version"]:
        decision = f"KEPT old champion v{old_version}: new {new_f1:.4f} < old {old_f1:.4f}"
    else:
        client.set_registered_model_alias(mcfg["model_name"], mcfg["alias"], str(info["model_version"]))
        decision = f"PROMOTED version {info['model_version']} to '{mcfg['alias']}' (macro_f1 {new_f1:.4f})"
    log.info(decision)
    with open("models/evaluation.json", "w") as f:
        json.dump({"decision": decision, "new_macro_f1": new_f1, "old_macro_f1": old_f1,
                   "model_version": info["model_version"]}, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("evaluate stage crashed")
        sys.exit(1)
