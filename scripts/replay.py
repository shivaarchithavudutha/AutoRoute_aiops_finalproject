"""Send real complaints to the API to generate traffic (drift demo).

Examples:
  python scripts/replay.py --dir data/raw/production --n 300
  python scripts/replay.py --dir data/raw/train --n 100          # in-distribution traffic
"""
import argparse
import glob
import os
import random
import sys
import time

import pandas as pd
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True, help="folder with unzipped CFPB csv files")
parser.add_argument("--n", type=int, default=200, help="number of complaints to send")
parser.add_argument("--api", default="http://localhost:8000")
parser.add_argument("--product", default="", help="only send complaints whose Product contains this text (simulates a shift in traffic)")
parser.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
args = parser.parse_args()

files = sorted(glob.glob(os.path.join(args.dir, "*.csv")))
if not files:
    sys.exit(f"No csv files in {args.dir}")
df = pd.read_csv(random.choice(files), usecols=["Product", "Consumer complaint narrative"], dtype=str).dropna()
if args.product:
    df = df[df["Product"].str.contains(args.product, case=False, regex=False)]
    if df.empty:
        sys.exit(f"No complaints with Product containing {args.product!r}")
texts = df["Consumer complaint narrative"].sample(min(args.n, len(df)), random_state=1).tolist()
headers = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}
ok = 0
for i, text in enumerate(texts, 1):
    try:
        r = requests.post(f"{args.api}/predict", json={"text": text[:5000]}, headers=headers, timeout=15)
        ok += r.status_code == 200
    except requests.RequestException as exc:
        print("request failed:", exc)
    if i % 50 == 0:
        print(f"sent {i}/{len(texts)} (ok={ok})")
    time.sleep(args.delay)
print(f"Done: {ok}/{len(texts)} successful")
