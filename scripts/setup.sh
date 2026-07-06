#!/usr/bin/env bash
# Install the convert/serve toolchain. mlx-lm from git main is required for the
# Qwen3.5/3.6 MoE quant predicates; mlx-vlm for the VLM path.
# TODO(T4): pin versions once a working combo is confirmed on the 24GB Mac.
set -euo pipefail

python3 -m pip install -e '.[convert,dev]'
python3 -m pip install --upgrade 'git+https://github.com/ml-explore/mlx-lm.git'
python3 -m pip install --upgrade mlx-vlm huggingface_hub

echo "Verify omlx is installed and models dir exists:"
echo "  ls ~/.omlx/models || mkdir -p ~/.omlx/models"
