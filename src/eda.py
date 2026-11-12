"""Exploratory data analysis: class balance, text length, complaints per year.
Writes charts and a summary into docs/eda/. Run after ingest:  python src/eda.py"""
import os

import matplotlib
import pandas as pd

from common import get_logger, load_params

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = get_logger("eda")


def main():
    params = load_params()
    df = pd.read_parquet(params["paths"]["interim"])
    out = "docs/eda"
    os.makedirs(out, exist_ok=True)

    counts = df["label"].value_counts()
    ax = counts.sort_values().plot.barh(figsize=(8, 4), color="#1f3a5f")
    ax.set_title("Complaints per class")
    plt.tight_layout()
    plt.savefig(f"{out}/class_distribution.png", dpi=120)
    plt.close()

    words = df["text"].str.split().str.len()
    words.clip(upper=words.quantile(0.99)).plot.hist(bins=50, figsize=(8, 4), color="#1f3a5f")
    plt.title("Words per complaint (clipped at 99th percentile)")
    plt.tight_layout()
    plt.savefig(f"{out}/text_length_hist.png", dpi=120)
    plt.close()

    years = pd.to_datetime(df["date"], errors="coerce").dt.year.value_counts().sort_index()
    years.plot.bar(figsize=(8, 4), color="#1f3a5f")
    plt.title("Complaints per year")
    plt.tight_layout()
    plt.savefig(f"{out}/complaints_per_year.png", dpi=120)
    plt.close()

    with open(f"{out}/eda_summary.md", "w") as f:
        f.write("# EDA summary\n\n")
        f.write(f"- rows: {len(df)}\n- classes: {counts.shape[0]}\n- median words per complaint: {words.median():.0f}\n")
        f.write(f"- imbalance (largest/smallest class): {counts.max() / counts.min():.1f}\n\n")
        f.write("## Class counts\n\n" + counts.to_markdown() + "\n")
    log.info("EDA written to %s (class counts:\n%s)", out, counts.to_string())


if __name__ == "__main__":
    main()
