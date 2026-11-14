#!/usr/bin/env python3
"""Give the documentation assets clear names and fix every link that points to them.

What it does (dry run by default; add --apply to really do it):
  1. docs/screenshots/Screenshot From <date>.png  ->  docs/screenshots/<meaningful_name>.png
  2. docs/screenshots/{architecture,retrain_flow,sequence_predict}.png  ->  docs/diagrams/
  3. docs/Reports/  ->  docs/pdf/   (so it is not confused with the root reports/ folder that holds
     the CI outputs written by DVC).  Use --keep-reports-folder to skip this step.
  4. Rewrites the old names/paths inside README.md and docs/*.md.

Usage (from the repository root):
    python scripts/reorganize_docs.py            # show what would change
    python scripts/reorganize_docs.py --apply    # do it
Then run:  python scripts/check_docs_links.py
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import urllib.parse

SHOTS = {
    "Screenshot From 2026-11-12 17-04-10.png": "mlflow_first_run.png",
    "Screenshot From 2026-11-12 17-07-45.png": "dvc_dag_and_metrics.png",
    "Screenshot From 2026-11-12 17-07-57.png": "dvc_metrics_and_first_commit.png",
    "Screenshot From 2026-11-12 18-24-45.png": "mlflow_compare_runs_params_metrics.png",
    "Screenshot From 2026-11-12 18-24-54.png": "mlflow_compare_runs_metrics_artifacts.png",
    "Screenshot From 2026-11-12 18-25-06.png": "mlflow_compare_parallel_coordinates.png",
    "Screenshot From 2026-11-12 18-25-21.png": "mlflow_compare_runs_details.png",
    "Screenshot From 2026-11-12 18-25-27.png": "mlflow_compare_runs_per_class_f1.png",
    "Screenshot From 2026-11-12 18-25-40.png": "mlflow_experiment_runs_list.png",
    "Screenshot From 2026-11-14 16-49-43.png": "grafana_drift_panels_start.png",
    "Screenshot From 2026-11-14 16-51-26.png": "prometheus_alert_prediction_drift_firing.png",
    "Screenshot From 2026-11-14 16-52-22.png": "grafana_dashboard_drift_demo.png",
    "Screenshot From 2026-11-14 16-53-32.png": "grafana_psi_peak.png",
    "Screenshot From 2026-11-14 16-54-12.png": "portal_health_and_drift.png",
    "Screenshot From 2026-11-14 16-55-20.png": "prometheus_targets_up.png",
    "Screenshot From 2026-11-14 16-56-14.png": "mlflow_model_registry_champion.png",
    "Screenshot From 2026-11-14 16-56-40.png": "portal_classify_input.png",
    "Screenshot From 2026-11-14 16-56-46.png": "portal_classify_result_feedback.png",
    "Screenshot From 2026-11-14 17-03-35.png": "portal_pipeline_console_failed_jobs.png",
    "Screenshot From 2026-11-14 17-07-25.png": "portal_pipeline_console_success_job.png",
    "Screenshot From 2026-11-14 17-14-06.png": "terminal_latency_check.png",
    "Screenshot From 2026-11-14 17-19-25.png": "mlflow_macro_f1_metric.png",
}
DIAGRAMS = ["architecture.png", "retrain_flow.png", "sequence_predict.png"]


def in_git():
    return subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True).returncode == 0


def move(src, dst, apply, git):
    if not os.path.exists(src):
        return False
    print(f"  move  {src}  ->  {dst}")
    if apply:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if git and subprocess.run(["git", "mv", src, dst], capture_output=True).returncode == 0:
            return True
        os.rename(src, dst)
    return True


def variants(name):
    """Ways an old file name may be written inside a markdown file."""
    return {name, urllib.parse.quote(name), name.replace(" ", "%20"), name.replace(" ", "+")}


def leftover_scan(keep_reports):
    """After rewriting, list lines in markdown files that still mention an old name."""
    needles = ["Screenshot From", "screenshots/architecture.png", "screenshots/retrain_flow.png", "screenshots/sequence_predict.png"]
    if not keep_reports:
        needles.append("Reports/")
    found = []
    for path in ["README.md"] + sorted(set(glob.glob("docs/*.md")) | set(glob.glob("docs/**/*.md", recursive=True))):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for no, line in enumerate(f, 1):
                if any(n in line for n in needles) or re.search(r"Screenshot\s+From", line):
                    found.append(f"{path}:{no}: {line.strip()[:100]}")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="really rename and rewrite (default: dry run)")
    ap.add_argument("--keep-reports-folder", action="store_true", help="do not rename docs/Reports to docs/pdf")
    args = ap.parse_args()
    if not os.path.isdir("docs"):
        sys.exit("Run this from the repository root (the folder that contains docs/).")
    git = in_git()
    print("DRY RUN (nothing changed). Add --apply to do it.\n" if not args.apply else "APPLYING changes.\n")

    replacements = []          # (old text, new text) used inside markdown files
    moved = 0
    print("Screenshots:")
    for old, new in SHOTS.items():
        if move(os.path.join("docs", "screenshots", old), os.path.join("docs", "screenshots", new), args.apply, git):
            moved += 1
        for v in variants(old):
            replacements.append((v, new))
    print("Diagrams:")
    for name in DIAGRAMS:
        if move(os.path.join("docs", "screenshots", name), os.path.join("docs", "diagrams", name), args.apply, git):
            moved += 1
        replacements.append((f"screenshots/{name}", f"diagrams/{name}"))
    if not args.keep_reports_folder and os.path.isdir(os.path.join("docs", "Reports")):
        print("PDF folder:")
        print("  move  docs/Reports  ->  docs/pdf")
        moved += 1
        if args.apply:
            if not (git and subprocess.run(["git", "mv", "docs/Reports", "docs/pdf"], capture_output=True).returncode == 0):
                os.rename("docs/Reports", "docs/pdf")
        replacements.append(("docs/Reports/", "docs/pdf/"))
        replacements.append(("docs/Reports", "docs/pdf"))
        replacements.append(("Reports/AutoRoute_", "pdf/AutoRoute_"))

    print("\nLinks rewritten in text files:")
    changed_files = 0
    for path in ["README.md"] + sorted(set(glob.glob("docs/*.md")) | set(glob.glob("docs/**/*.md", recursive=True))):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        new_text, n = text, 0
        for old, new in replacements:
            if old in new_text:
                n += new_text.count(old)
                new_text = new_text.replace(old, new)
        if n:
            changed_files += 1
            print(f"  {path}: {n} reference(s)")
            if args.apply:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_text)
    if not changed_files:
        print("  (no references to old names found in README.md / docs/*.md)")
    if args.apply:
        left = leftover_scan(args.keep_reports_folder)
        if left:
            print("\nWARNING: old names still appear in these places; edit them by hand:")
            for item in left:
                print("  " + item)
    print(f"\n{moved} item(s) moved, {changed_files} file(s) with updated links.")
    if not args.apply:
        print("Run again with --apply to make the changes.")
    else:
        print("Next: python scripts/check_docs_links.py")


if __name__ == "__main__":
    main()
