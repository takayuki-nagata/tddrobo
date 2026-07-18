#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Takayuki Nagata


set -e

# Determine project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# Find python
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python"
fi

echo "=== 🧹 Running Ruff Formatter ==="
$PYTHON -m ruff format .

echo "=== 🔍 Running Ruff Linter ==="
$PYTHON -m ruff check . --fix

echo "=== 🏷️ Running Mypy Type Checker ==="
$PYTHON -m mypy .

echo "=== 🧪 Running Pytest ==="
$PYTHON -m pytest --cov=. --cov-report=term-missing --cov-fail-under=100

echo "=== 📊 Generating Workflow Graph ==="
$PYTHON cli.py --draw-graph

echo ""
echo "=== ✅ All checks passed successfully! 🎉 ==="