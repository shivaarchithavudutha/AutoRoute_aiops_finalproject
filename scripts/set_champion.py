"""Roll back (or roll forward) by pointing the 'champion' alias at a version.
Usage: python scripts/set_champion.py <version>"""
import os
import sys

from mlflow.tracking import MlflowClient

name = os.environ.get("MODEL_NAME", "AutoRouteModel")
client = MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
if len(sys.argv) != 2:
    for mv in client.search_model_versions(f"name='{name}'"):
        print("version", mv.version, "run", mv.run_id[:8])
    sys.exit("Usage: python scripts/set_champion.py <version>")
client.set_registered_model_alias(name, "champion", sys.argv[1])
print(f"champion -> version {sys.argv[1]} (the model server switches within ~30 seconds)")
