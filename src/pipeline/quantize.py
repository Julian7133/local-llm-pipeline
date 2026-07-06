"""Stage 3: quantize with the UD recipe via mlx-vlm (VLM) or mlx-lm (text).

Owner: T2. See PLAN.md §2.3.

The recipe -> predicate + config overrides is already provided in recipe.py.
This module's job is purely dispatch + wiring, NOT re-deriving bit allocations.

TODO(T2):
  - predicate = build_predicate(cfg.recipe); overrides = config_overrides(...)
  - if cfg.model_type is VLM: call mlx_vlm.convert(hf_path, mlx_path,
        quant_predicate=predicate, q_group_size=recipe.group_size, ...)
    else: mlx_lm.convert(...) with the same predicate.
  - After convert, merge `overrides` into the written config.json's
    `quantization` block so the mixed-bit layout is self-describing.
  - Verify no DeltaNet state tensor (A_log/conv1d/norms) was quantized.
  - Return StageResult with artifact_path = mlx output dir.

Note: mlx-lm from git main is required for the predicate API on this arch.
"""

from __future__ import annotations

from pathlib import Path

from .recipe import build_predicate, config_overrides
from .types import ModelConfig, StageResult


def quantize(cfg: ModelConfig, source_dir: Path) -> StageResult:  # noqa: ARG001
    raise NotImplementedError("T2: implement mlx-vlm/mlx-lm dispatch + recipe apply")
