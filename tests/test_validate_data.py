import pandas as pd

from validate_data import run_checks

CFG = {"min_rows": 4, "min_classes": 2, "min_words": 3, "max_short_share": 0.5,
       "max_duplicate_share": 0.5, "max_imbalance_ratio": 100}


def make(texts, labels):
    return pd.DataFrame({"date": ["2020-01-01"] * len(texts), "label": labels, "text": texts})


def test_good_data_passes():
    df = make(["one two three four", "five six seven eight", "nine ten eleven twelve", "a b c d e"], ["x", "x", "y", "y"])
    assert run_checks(df, CFG)["passed"] is True


def test_too_few_rows_fails():
    assert run_checks(make(["a b c d"], ["x"]), CFG)["passed"] is False


def test_single_class_fails():
    df = make(["a b c d", "e f g h", "i j k l", "m n o p"], ["x"] * 4)
    report = run_checks(df, CFG)
    assert report["passed"] is False
    assert any(c["name"] == "min_classes" and not c["passed"] for c in report["checks"])


def test_many_duplicates_fail():
    df = make(["same text here"] * 4, ["x", "x", "y", "y"])
    assert run_checks(df, CFG)["passed"] is False


def test_missing_columns_reported():
    assert run_checks(pd.DataFrame({"text": ["a"]}), CFG)["passed"] is False
