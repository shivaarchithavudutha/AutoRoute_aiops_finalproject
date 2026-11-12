"""Small helpers used by several pipeline scripts."""
import hashlib
import logging
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints timestamped lines (also visible in Docker logs)."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


def load_params(path: str = "params.yaml") -> dict:
    """Load params.yaml and merge in label_map.yaml under the key 'label_map'."""
    with open(path) as f:
        params = yaml.safe_load(f)
    with open(os.path.join(os.path.dirname(path) or ".", "label_map.yaml")) as f:
        params["label_map"] = yaml.safe_load(f)
    return params


def map_label(product: str, label_map: dict) -> str:
    """Map a raw CFPB Product name to our consolidated class name."""
    return label_map.get(str(product).strip(), str(product).strip())


def git_commit() -> str:
    """Current Git commit hash, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def file_md5(path: str) -> str:
    """MD5 of a file, used to record which data version trained a model."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
