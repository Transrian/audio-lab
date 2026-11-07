#!/usr/bin/env bash
# Run every check that CI runs. Must pass before every push.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run ruff check .
uv run ruff format --check .
uv run pytest -q

# Check the Owner-Approved trailer on commits not pushed yet.
if upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null); then
  range="$upstream..HEAD"
else
  range=all
fi
uv run python scripts/check_repo.py --commits "$range"
