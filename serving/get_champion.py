"""Print the version number the 'champion' alias points to (or nothing if none)."""
import os

from mlflow.tracking import MlflowClient

try:
    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    mv = client.get_model_version_by_alias(os.environ.get("MODEL_NAME", "AutoRouteModel"),
                                           os.environ.get("MODEL_ALIAS", "champion"))
    print(mv.version)
except Exception:
    pass
