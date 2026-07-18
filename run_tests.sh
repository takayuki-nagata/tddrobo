#!/bin/bash

set -e

echo "=== 🧹 Running Ruff Formatter ==="
uv run ruff format .

echo "=== 🔍 Running Ruff Linter ==="
uv run ruff check . --fix

echo "=== 🏷️ Running Mypy Type Checker ==="
uv run mypy .

echo "=== 🧪 Running Pytest ==="
uv run pytest --cov=. --cov-report=term-missing --cov-fail-under=100

echo ""
echo "=== ✅ All checks passed successfully! 🎉 ==="