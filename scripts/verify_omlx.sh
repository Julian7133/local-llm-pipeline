#!/usr/bin/env bash
# Verify oMLX is installed and the models directory is present.
set -euo pipefail

MODELS_DIR="${HOME}/.omlx/models"

if ! command -v omlx >/dev/null 2>&1; then
  echo "error: omlx not found in PATH (install oMLX separately)" >&2
  exit 1
fi

if [[ ! -d "${MODELS_DIR}" ]]; then
  echo "error: ${MODELS_DIR} does not exist (run: mkdir -p ${MODELS_DIR})" >&2
  exit 1
fi

if omlx --version >/dev/null 2>&1; then
  echo "ok: omlx $(omlx --version)"
else
  echo "ok: omlx installed"
fi
echo "ok: models dir ${MODELS_DIR}"
