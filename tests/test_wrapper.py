import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from model_wrapper import AutoRouteModel


class Ctx:
    def __init__(self, path):
        self.artifacts = {"sklearn_pipeline": path}


def test_wrapper_returns_probability_dataframe(tmp_path):
    pipe = Pipeline([("t", TfidfVectorizer()), ("c", LogisticRegression())])
    pipe.fit(["debt call", "mortgage escrow", "debt collector", "mortgage rate"], ["d", "m", "d", "m"])
    path = str(tmp_path / "p.joblib")
    joblib.dump(pipe, path)
    model = AutoRouteModel()
    model.load_context(Ctx(path))
    out = model.predict(None, pd.DataFrame({"narrative": ["DEBT call!!", "mortgage"]}))
    assert list(out.columns) == ["d", "m"]
    assert out.shape == (2, 2)
    assert abs(out.iloc[0].sum() - 1) < 1e-6
