#!/bin/bash
set -euo pipefail

export MEDICAL_MI_BASE_DIR="${MEDICAL_MI_BASE_DIR:-/home/eagle0914/medical_mi}"
export MEDICAL_MI_DATA_ROOT="${MEDICAL_MI_DATA_ROOT:-/data/heejae}"
export UV_CACHE_DIR="$MEDICAL_MI_DATA_ROOT/.cache/uv"
export HF_HOME="$MEDICAL_MI_DATA_ROOT/.cache/huggingface"
export TMPDIR="$MEDICAL_MI_DATA_ROOT/tmp"
export PYTHON_VENV="$MEDICAL_MI_DATA_ROOT/.venv"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TMPDIR"

if [ ! -d "$PYTHON_VENV" ]; then
  echo "Missing venv: $PYTHON_VENV"
  echo "Run: cd $MEDICAL_MI_BASE_DIR && bash prepare_server.sh"
  exit 1
fi

source "$PYTHON_VENV/bin/activate"
cd "$MEDICAL_MI_BASE_DIR"

exec "$@"
