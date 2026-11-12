"""MLflow pyfunc wrapper: takes a DataFrame with a 'narrative' column and returns a
DataFrame of class probabilities (one column per class).

Why a wrapper? A bare scikit-learn Pipeline would receive the DataFrame itself and
TF-IDF would iterate over column names. The wrapper picks the column and applies the
SAME text cleaning used in training.
"""
import io
import os

import joblib
import mlflow.pyfunc
import pandas as pd

from text_utils import clean_text


class AutoRouteModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        path = context.artifacts["sklearn_pipeline"]
        if path.endswith(".enc"):          # encrypted model (optional security feature)
            key = os.environ.get("MODEL_ENCRYPTION_KEY", "")
            if not key:
                raise RuntimeError("This model is encrypted: set MODEL_ENCRYPTION_KEY for the model server")
            from cryptography.fernet import Fernet
            with open(path, "rb") as f:
                self.pipe = joblib.load(io.BytesIO(Fernet(key.encode()).decrypt(f.read())))
        else:
            self.pipe = joblib.load(path)
        self.classes = [str(c) for c in self.pipe.classes_]

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["narrative"].astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]
        proba = self.pipe.predict_proba([clean_text(t) for t in texts])
        return pd.DataFrame(proba, columns=self.classes)
