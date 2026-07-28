#!/usr/bin/env bash
# Pre-push gate: formatting and lint must be clean.
# Run before every push (skip only for docs-only changes). CI runs the same.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

if [ -x .venv/bin/ruff ]; then
    RUFF=.venv/bin/ruff
else
    RUFF=ruff
fi

echo "==> ruff format --check"
"$RUFF" format --check .

echo "==> ruff check"
"$RUFF" check .

echo "==> mypy"
if [ -x .venv/bin/mypy ]; then MYPY=.venv/bin/mypy; else MYPY=mypy; fi
"$MYPY" -p descan

echo "OK: gate passed"
