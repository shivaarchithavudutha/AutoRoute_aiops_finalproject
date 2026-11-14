#!/usr/bin/env python3
"""Check that every link, image and file path mentioned in README.md and docs/*.md really exists.

Usage (from the repository root):  python scripts/check_docs_links.py
Exit code 1 if a markdown link/image is broken; paths mentioned in `backticks` that do not exist
are listed as warnings.
"""
import glob
import os
import re
import sys
import urllib.parse

LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+(?:\s+\"[^\"]*\")?)\)")
HTML = re.compile(r"(?:src|href)=[\"']([^\"']+)[\"']")
TICK = re.compile(r"`([^`\s]+)`")
EXT = (".md", ".pdf", ".png", ".jpg", ".py", ".yml", ".yaml", ".json", ".txt", ".sh", ".toml", ".cfg")


def is_external(target):
    return target.startswith(("http://", "https://", "mailto:", "#", "data:"))


def main():
    files = ["README.md"] + sorted(glob.glob("docs/**/*.md", recursive=True))
    broken, warnings, checked = [], [], 0
    for path in files:
        if not os.path.isfile(path):
            continue
        base = os.path.dirname(path) or "."
        with open(path, encoding="utf-8") as f:
            text = f.read()
        targets = [m.split(' "')[0] for m in LINK.findall(text)] + HTML.findall(text)
        for t in targets:
            if is_external(t):
                continue
            checked += 1
            rel = urllib.parse.unquote(t.split("#")[0].split("?")[0])
            if rel and not os.path.exists(os.path.normpath(os.path.join(base, rel))):
                broken.append(f"{path}: broken link -> {t}")
        for t in TICK.findall(text):
            if "/" not in t or not t.endswith(EXT) or any(c in t for c in "*<>{}$"):
                continue
            checked += 1
            if not (os.path.exists(t) or os.path.exists(os.path.join(base, t))):
                warnings.append(f"{path}: path in backticks not found -> {t}")
    print(f"Checked {checked} references in {len(files)} file(s).")
    for w in warnings:
        print("WARNING ", w)
    for b in broken:
        print("BROKEN  ", b)
    if broken:
        print(f"\n{len(broken)} broken link(s). Fix them (or run scripts/reorganize_docs.py --apply) and run this again.")
        sys.exit(1)
    print("All markdown links and images resolve." + (" (Some mentioned paths were not found, see warnings.)" if warnings else ""))


if __name__ == "__main__":
    main()
