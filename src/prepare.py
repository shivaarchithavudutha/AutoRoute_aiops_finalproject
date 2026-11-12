"""STAGE 2 - prepare: drop tiny classes, cap big classes, split train/test,
and build the drift baseline (vocabulary, length sample, class prior).

Run:  python src/prepare.py
"""
import sys
import json
import os
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

from common import get_logger, load_params

log = get_logger("prepare")


def build_baseline(train: pd.DataFrame, cfg: dict) -> dict:
    """Reference statistics that the gateway compares live traffic against."""
    counts = Counter(w for t in train["text"] for w in t.split())
    vocab = sorted(w for w, c in counts.items() if c >= cfg["vocab_min_count"])
    lengths = train["text"].str.split().str.len()
    sample = lengths.sample(min(cfg["length_sample"], len(lengths)), random_state=cfg["seed"])
    prior = train["label"].value_counts(normalize=True)
    return {"classes": sorted(prior.index.tolist()),
            "class_prior": {k: float(v) for k, v in prior.items()},
            "vocab": vocab,
            "length_sample": [int(x) for x in sample],
            "n_train": int(len(train))}


def main():
    params = load_params()
    cfg, paths = params["prepare"], params["paths"]
    df = pd.read_parquet(paths["interim"])

    counts = df["label"].value_counts()
    keep = counts[counts >= cfg["min_class_rows"]].index
    log.info("Class counts:\n%s", counts.to_string())
    log.info("Keeping classes: %s", list(keep))
    df = df[df["label"].isin(keep)]
    parts = [g.sample(min(len(g), cfg["max_rows_per_class"]), random_state=cfg["seed"])
             for _, g in df.groupby("label")]
    df = pd.concat(parts, ignore_index=True)

    train, test = train_test_split(df, test_size=cfg["test_size"], stratify=df["label"],
                                   random_state=cfg["seed"])
    os.makedirs(os.path.dirname(paths["train"]), exist_ok=True)
    train.to_parquet(paths["train"], index=False)
    test.to_parquet(paths["test"], index=False)
    with open(paths["baseline"], "w") as f:
        json.dump(build_baseline(train, cfg), f)
    log.info("train=%d test=%d classes=%d", len(train), len(test), df["label"].nunique())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("prepare stage crashed")
        sys.exit(1)
