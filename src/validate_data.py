"""STAGE 2 - validate: data-quality checks on the cleaned data.

HARD checks stop the pipeline (exit code 1), so bad data never reaches training.
WARNING checks are only reported. Output: data/interim/data_quality.json
Run:  python src/validate_data.py
"""
import json
import sys

import pandas as pd

from common import get_logger, load_params

log = get_logger("validate")
REQUIRED = ["date", "label", "text"]


def run_checks(df: pd.DataFrame, cfg: dict) -> dict:
    """Return a report dict with flat numeric metrics plus a list of individual checks."""
    checks = []

    def add(name, passed, value, threshold, severity="hard"):
        checks.append({"name": name, "passed": bool(passed), "value": value,
                       "threshold": threshold, "severity": severity})

    missing = [c for c in REQUIRED if c not in df.columns]
    add("required_columns_present", not missing, missing, REQUIRED)
    if missing:
        return {"passed": False, "checks": checks, "hard_failures": 1}

    counts = df["label"].value_counts()
    words = df["text"].str.split().str.len()
    short_share = float((words < cfg["min_words"]).mean()) if len(df) else 1.0
    dup_share = float(df["text"].duplicated().mean()) if len(df) else 1.0
    null_share = float(df[REQUIRED].isna().any(axis=1).mean()) if len(df) else 1.0
    imbalance = float(counts.max() / counts.min()) if len(counts) else 0.0
    bad_dates = float(pd.to_datetime(df["date"], errors="coerce").isna().mean()) if len(df) else 1.0

    add("min_rows", len(df) >= cfg["min_rows"], int(len(df)), cfg["min_rows"])
    add("min_classes", len(counts) >= cfg["min_classes"], int(len(counts)), cfg["min_classes"])
    add("no_null_values", null_share == 0, round(null_share, 4), 0)
    add("short_text_share", short_share <= cfg["max_short_share"], round(short_share, 4), cfg["max_short_share"])
    add("duplicate_text_share", dup_share <= cfg["max_duplicate_share"], round(dup_share, 4), cfg["max_duplicate_share"])
    add("class_imbalance_ratio", imbalance <= cfg["max_imbalance_ratio"], round(imbalance, 1),
        cfg["max_imbalance_ratio"], severity="warning")
    add("unparseable_dates_share", bad_dates <= 0.01, round(bad_dates, 4), 0.01, severity="warning")

    hard_failures = sum(1 for c in checks if c["severity"] == "hard" and not c["passed"])
    return {"passed": hard_failures == 0, "hard_failures": hard_failures, "rows": int(len(df)),
            "n_classes": int(len(counts)), "short_text_share": round(short_share, 4),
            "duplicate_text_share": round(dup_share, 4), "class_imbalance_ratio": round(imbalance, 1),
            "median_words": float(words.median()) if len(df) else 0.0, "checks": checks}


def main():
    params = load_params()
    df = pd.read_parquet(params["paths"]["interim"])
    report = run_checks(df, params["validate"])
    with open("data/interim/data_quality.json", "w") as f:
        json.dump(report, f, indent=2)
    for c in report["checks"]:
        log.info("%s %-26s value=%s threshold=%s (%s)", "PASS" if c["passed"] else "FAIL",
                 c["name"], c["value"], c["threshold"], c["severity"])
    if not report["passed"]:
        log.error("Data quality gate FAILED: %d hard check(s). Fix the data before training.", report["hard_failures"])
        sys.exit(1)
    log.info("Data quality gate passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("validate stage crashed")
        sys.exit(1)
