# Test Plan, Test Cases and Report
**Scope:** text cleaning, label mapping, drift maths, model wrapper, gateway API, end-to-end stack.
**Approach:** pytest unit tests (`python -m pytest -v`), manual integration checks on the running stack, load check for latency.
**Acceptance criteria:** macro F1 >= 0.75 on held-out test; p95 `/predict` latency < 300 ms; all endpoints match the LLD; all critical tests pass.

## Test cases (extend this table; the automated ones already exist in tests/)
| ID | Description | Input | Expected | Actual | Pass/Fail |
|---|---|---|---|---|---|
| T01 | clean_text lowercases and strips punctuation | "Hello, World!!" | "hello world" | | |
| T02 | clean_text removes XXXX | "paid XXXX" | "paid" | | |
| T03 | /predict valid text | text | 200, category, confidence | | |
| T04 | /predict empty text | "   " | 422 | | |
| T05 | /predict text over 5000 chars | 5001 chars | 422 | | |
| T06 | model server down | stop model-server | 503 | | |
| T07 | API key enforced | key missing | 401 | | |
| T08 | PSI identical distribution | same | ~0 | | |
| T09 | drift buffer not ready | <50 requests | buffer_ready false | | |
| T10 | /metrics exposes custom metrics | GET | names present | | |
| T11 | retrain trigger | POST trigger | job success, new version | | |
| T12 | rollback | set_champion older version | served version changes | | |
| T13 | acceptance: macro F1 | test set | >= 0.75 | | |
| T14 | acceptance: p95 latency | 200 requests | < 300 ms | | |

## Report (fill after the final run)
Total tests: __  Passed: __  Failed: __  Date: __  Commit: __
Attach the pytest output and screenshots.
