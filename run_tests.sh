#!/bin/bash

set -e

# Find python/ruff/mypy/pytest
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
    RUFF=".venv/bin/ruff"
    MYPY=".venv/bin/mypy"
    PYTEST=".venv/bin/pytest"
else
    PYTHON="python"
    RUFF="ruff"
    MYPY="mypy"
    PYTEST="pytest"
fi

echo "=== 🧹 Running Ruff Formatter ==="
$RUFF format .

echo "=== 🔍 Running Ruff Linter ==="
$RUFF check . --fix

echo "=== 🏷️ Running Mypy Type Checker ==="
$MYPY .

echo "=== 🧪 Running Pytest ==="
$PYTEST --cov=. --cov-report=term-missing --cov-fail-under=100

echo ""
echo "=== ✅ All checks passed successfully! 🎉 ==="