#!/usr/bin/env bash
# Install the convert/serve toolchain for the UD 3-bit pipeline.
#
# mlx-lm from git main is required for Qwen3.5/3.6 MoE quant predicates;
# mlx-vlm for the VLM convert path. oMLX is the serving runtime (install
# separately — see README).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found (requires >= 3.11)" >&2
  exit 1
fi

PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="${PY_VER%%.*}"
PY_MINOR="${PY_VER#*.}"
if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 11) )); then
  echo "error: python ${PY_VER} is too old; need >= 3.11" >&2
  exit 1
fi

echo "==> Installing pipeline package (editable) + convert/dev extras"
python3 -m pip install -e '.[convert,dev]'

echo "==> Installing mlx-lm from git main (Qwen3.5/3.6 MoE arch fixes)"
python3 -m pip install --upgrade 'git+https://github.com/ml-explore/mlx-lm.git'

echo "==> Installing mlx-vlm and huggingface_hub"
python3 -m pip install --upgrade mlx-vlm huggingface_hub

echo "==> Ensuring oMLX models directory exists"
mkdir -p "${HOME}/.omlx/models"

if [[ -x "${ROOT}/scripts/verify_omlx.sh" ]]; then
  echo "==> Verifying oMLX (optional — skip if not installed yet)"
  bash "${ROOT}/scripts/verify_omlx.sh" || true
fi

echo "Setup complete. Run: make test"
