#!/bin/bash

set -e

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

echo ""
echo "=== ✅ All checks passed successfully! 🎉 ==="