#!/usr/bin/env bash
# Lint + format check + tests. Run from anywhere.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PATHS=("Code/src/nnunet_isles" "Code/tests" "Code/scripts" "Code/paths")

echo "[1/3] ruff check"
ruff check "${PATHS[@]}"

echo "[2/3] ruff format --check"
ruff format --check "${PATHS[@]}"

echo "[3/3] pytest -x"
pytest -x Code/tests/

echo "OK"
