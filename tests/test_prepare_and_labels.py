import pandas as pd

from common import map_label
from prepare import build_baseline

LABEL_MAP = {"Credit card or prepaid card": "Credit card", "Credit card": "Credit card"}


def test_map_label_known_and_unknown():
    assert map_label("Credit card or prepaid card", LABEL_MAP) == "Credit card"
    assert map_label("Something new", LABEL_MAP) == "Something new"


def test_baseline_contents():
    df = pd.DataFrame({"text": ["a b c", "a b d", "a e"], "label": ["x", "x", "y"]})
    base = build_baseline(df, {"vocab_min_count": 2, "length_sample": 10, "seed": 0})
    assert base["classes"] == ["x", "y"]
    assert abs(sum(base["class_prior"].values()) - 1) < 1e-9
    assert set(base["vocab"]) == {"a", "b"}
    assert base["n_train"] == 3
