#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e ".[dev,tokenizers]"

if command -v npm >/dev/null 2>&1; then
  (
    cd web
    npm install
    npm run build
  )
else
  echo "npm was not found; skipped frontend dependency install and rebuild."
  echo "The checked-in web/static assets can still be served by the Python app."
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Edit .env before calling a model."
fi

echo "Setup complete."
