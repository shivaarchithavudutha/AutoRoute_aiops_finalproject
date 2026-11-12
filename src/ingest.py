"""STAGE 1 - ingest: read raw CFPB CSVs, keep rows that have a narrative, clean the
text in parallel with Ray, and benchmark Ray against plain Python and multiprocessing.

Run:  python src/ingest.py
Out:  data/interim/clean.parquet, data/interim/ingest_metrics.json
"""
import sys
import glob
import json
import multiprocessing as mp
import os
import time

import pandas as pd

from common import get_logger, load_params, map_label
from text_utils import clean_text

log = get_logger("ingest")
COLS = ["Date received", "Product", "Consumer complaint narrative"]


def clean_chunk(texts):
    """Clean a list of texts (the unit of work handed to each worker)."""
    return [clean_text(t) for t in texts]


def load_raw(raw_dir: str, label_map: dict) -> pd.DataFrame:
    """Read every CSV in raw_dir, keep only rows with a narrative, map labels."""
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No .csv files found in {raw_dir}. Unzip the CFPB files there first.")
    frames = []
    for path in files:
        log.info("Reading %s", path)
        df = pd.read_csv(path, usecols=COLS, dtype=str)
        before = len(df)
        df = df.dropna(subset=["Consumer complaint narrative"])
        log.info("  %d rows, %d with a narrative", before, len(df))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"Date received": "date", "Product": "product",
                            "Consumer complaint narrative": "raw_text"})
    df["label"] = df["product"].map(lambda p: map_label(p, label_map))
    return df


def split_chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def run_single(texts, size):
    return [c for chunk in split_chunks(texts, size) for c in clean_chunk(chunk)]


def run_multiprocessing(texts, size, workers):
    with mp.Pool(workers) as pool:
        parts = pool.map(clean_chunk, split_chunks(texts, size))
    return [c for part in parts for c in part]


def run_ray(texts, size, workers):
    import ray
    if not ray.is_initialized():
        ray.init(num_cpus=workers, include_dashboard=False, ignore_reinit_error=True,
                 logging_level="WARNING")
    remote_clean = ray.remote(clean_chunk)
    futures = [remote_clean.remote(chunk) for chunk in split_chunks(texts, size)]
    return [c for part in ray.get(futures) for c in part]


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def main():
    params = load_params()
    ing, paths = params["ingest"], params["paths"]
    workers, size = ing["workers"], ing["chunk_size"]
    bench_rows = ing.get("bench_rows", 50000)

    df = load_raw(paths["raw_train_dir"], params["label_map"])
    texts = df["raw_text"].tolist()
    log.info("Total rows to clean: %d", len(texts))

    # Benchmark on the same sample so the comparison is fair.
    sample = texts[:bench_rows]
    _, t_single = timed(run_single, sample, size)
    _, t_mp = timed(run_multiprocessing, sample, size, workers)
    run_ray(sample[:100], 50, workers)                      # warm-up (Ray start-up cost)
    _, t_ray = timed(run_ray, sample, size, workers)
    n = len(sample)

    # Real run with Ray on all rows.
    cleaned, t_full = timed(run_ray, texts, size, workers)
    df["text"] = cleaned
    df = df[df["text"].str.len() > 0][["date", "label", "text"]]

    metrics = {
        "rows_total": len(texts), "rows_kept": len(df), "workers": workers, "bench_rows": n,
        "single_rows_per_sec": round(n / t_single, 1),
        "multiprocessing_rows_per_sec": round(n / t_mp, 1),
        "ray_rows_per_sec": round(n / t_ray, 1),
        "ray_speedup_vs_single": round(t_single / t_ray, 2),
        "ray_full_seconds": round(t_full, 2),
        "ray_full_rows_per_sec": round(len(texts) / t_full, 1),
    }
    os.makedirs(os.path.dirname(paths["interim"]), exist_ok=True)
    df.to_parquet(paths["interim"], index=False)
    with open("data/interim/ingest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Ingest metrics: %s", metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("ingest stage crashed")
        sys.exit(1)
