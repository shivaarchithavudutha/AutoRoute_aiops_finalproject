"""CI checks used as the first DVC stage: flake8 (PEP8) + pytest.

Writes reports/ci_summary.json (deterministic counts, used as a DVC metric/dependency)
and reports/test_output.txt (full output = your attached test report).
Exits with code 1 if anything fails, so DVC stops before training bad code.
"""
import json
import os
import re
import subprocess
import sys

ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    return proc.returncode, proc.stdout + proc.stderr


def count(pattern, text):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else 0


def main():
    os.makedirs("reports", exist_ok=True)
    lint_rc, lint_out = run([sys.executable, "-m", "flake8", "."])
    test_rc, test_out = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
    summary = {"tests_passed": count(r"(\d+) passed", test_out), "tests_failed": count(r"(\d+) failed", test_out),
               "tests_errors": count(r"(\d+) error", test_out),
               "lint_issues": len(lint_out.strip().splitlines()) if lint_rc else 0}
    summary["all_passed"] = lint_rc == 0 and test_rc == 0
    with open("reports/test_output.txt", "w") as f:
        f.write("=== flake8 ===\n" + (lint_out or "no issues\n") + "\n=== pytest ===\n" + test_out)
    with open("reports/ci_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))
    sys.exit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
