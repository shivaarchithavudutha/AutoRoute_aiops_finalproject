"""Measure /predict latency (acceptance criterion: p95 < 300 ms).  Usage: python scripts/latency_check.py [n]"""
import os
import statistics
import sys
import time

import requests

API = os.environ.get("API_URL", "http://localhost:8000")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
HEADERS = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}
TEXT = "A debt collector keeps calling me about a loan I already paid. I want this to stop and the account corrected."

times = []
for _ in range(N):
    start = time.perf_counter()
    r = requests.post(f"{API}/predict", json={"text": TEXT}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    times.append((time.perf_counter() - start) * 1000)
times.sort()
p95 = times[int(0.95 * len(times)) - 1]
print(f"requests={N} mean={statistics.mean(times):.1f} ms median={statistics.median(times):.1f} ms p95={p95:.1f} ms")
print("PASS (p95 < 300 ms)" if p95 < 300 else "FAIL (p95 >= 300 ms)")
