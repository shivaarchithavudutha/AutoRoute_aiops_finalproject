"""Trainer service: a tiny REST job runner that executes the DVC pipeline.

The gateway calls this service, so training never competes with inference for CPU.
POST /run {"action": "retrain"}  -> force train + evaluate (skips slow ingest)
POST /run {"action": "full"}     -> dvc repro (re-runs only stages whose inputs changed)
"""
import os
import subprocess
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from fastapi.responses import Response

REPO = os.environ.get("REPO_DIR", "/repo")
ACTIONS = {"retrain": [["dvc", "repro", "-s", "-f", "train"], ["dvc", "repro", "-s", "-f", "evaluate"]],
           "full": [["dvc", "repro"]]}

app = FastAPI(title="AutoRoute Trainer")
JOBS_TOTAL = Counter("trainer_jobs_total", "Pipeline jobs", ["status"])
RUNNING = Gauge("trainer_running", "1 while a job is running")
lock = threading.Lock()
jobs = []   # newest last


class RunRequest(BaseModel):
    action: str = "retrain"


def execute(job: dict):
    RUNNING.set(1)
    began = time.perf_counter()
    log_parts = []
    try:
        for cmd in ACTIONS[job["action"]]:
            log_parts.append("$ " + " ".join(cmd))
            proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            log_parts.append((proc.stdout + proc.stderr)[-3000:])
            if proc.returncode != 0:
                job["status"] = "failed"
                break
        else:
            job["status"] = "success"
    except Exception as exc:
        log_parts.append(f"ERROR: {exc}")
        job["status"] = "failed"
    job["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    job["duration_s"] = round(time.perf_counter() - began, 1)
    job["log_tail"] = "\n".join(log_parts)[-4000:]
    JOBS_TOTAL.labels(job["status"]).inc()
    RUNNING.set(0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run", status_code=202)
def run(req: RunRequest):
    if req.action not in ACTIONS:
        raise HTTPException(status_code=422, detail=f"action must be one of {list(ACTIONS)}")
    with lock:
        if jobs and jobs[-1]["status"] == "running":
            raise HTTPException(status_code=409, detail="A job is already running")
        job = {"job_id": uuid.uuid4().hex[:8], "action": req.action, "status": "running",
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None, "duration_s": None,
               "log_tail": ""}
        jobs.append(job)
        del jobs[:-20]
    threading.Thread(target=execute, args=(job,), daemon=True).start()
    return {"job_id": job["job_id"], "status": "running"}


@app.get("/status")
def status():
    return jobs[-1] if jobs else {"status": "idle", "log_tail": ""}


@app.get("/jobs")
def list_jobs():
    return [{k: v for k, v in j.items() if k != "log_tail"} for j in reversed(jobs)]


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
