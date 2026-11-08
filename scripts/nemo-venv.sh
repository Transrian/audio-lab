#!/usr/bin/env bash
# Create the separate virtual environment that can run a local model.
#
# The heavy dependencies (NeMo, PyTorch, CUDA) live in requirements/nemo.txt
# rather than in pyproject.toml, so the default environment, `uv sync` and CI
# stay light: nothing here is needed to run the evaluation. This script builds
# an environment that has both the heavy stack and audiolab itself, and leaves
# the light one alone.
#
#   bash scripts/nemo-venv.sh                        # ./.venv-nemo
#   AUDIOLAB_NEMO_VENV=.venv-cuda bash scripts/nemo-venv.sh
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${AUDIOLAB_NEMO_VENV:-.venv-nemo}"
PYTHON="${AUDIOLAB_NEMO_PYTHON_VERSION:-3.12}"

# --clear so that running it twice gives the same environment instead of an
# error, and so that a changed pin list is applied.
uv venv --clear --python "$PYTHON" "$VENV"
# One command for both indexes; requirements/nemo.txt explains the index-strategy
# flag and why PyTorch comes from the CUDA index rather than from PyPI.
uv pip install --index-strategy unsafe-best-match \
  --python "$VENV/bin/python" -e . -r requirements/nemo.txt

cat <<EOF

Local model environment ready: $VENV

  Serve or train from it (it has NeMo, so it needs no help):

    $VENV/bin/audiolab serve --model configs/models/parakeet-tdt-0.6b-v3-nemo.yaml

  Let the light environment start that server by itself:

    export AUDIOLAB_NEMO_PYTHON=$PWD/$VENV/bin/python
    uv run audiolab eval --model configs/models/parakeet-tdt-0.6b-v3-nemo.yaml \\
        --suite configs/suites/fast.yaml
EOF
