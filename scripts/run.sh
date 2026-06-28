#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

if [ ! -d "$VENV_DIR" ]; then
  echo "Virtual environment not found. Run ./scripts/setup.sh first."
  exit 1
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Fill in API_KEY, BASE_URL, and MODEL, then rerun this script."
  exit 1
fi

if grep -Eq '^(API_KEY|BASE_URL|MODEL)=$' .env; then
  echo "Warning: .env still has empty API_KEY, BASE_URL, or MODEL values."
  echo "The web UI can start, but model calls will fail until provider settings are filled."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

exec python -m web.main --host "$HOST" --port "$PORT"
