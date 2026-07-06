"""Stage 4: architecture fixups after quantization.

Owner: T2. See PLAN.md §2.4 and §0 (arch quirks).

The reference model declares MTP heads in config but ships no mtp.* weights;
oMLX patches this at runtime. Naive converters can also mis-handle the hybrid
Gated DeltaNet layers and the shared_expert_gate.

TODO(T2):
  - MTP: if cfg.strip_mtp_config, remove mtp_* keys from config.json; else leave
    them (oMLX's runtime patch skips MTPModule attachment). Validate BOTH once.
  - Assert A_log / dt_bias / conv1d / *norm tensors remain unquantized.
  - Assert router gate + shared_expert_gate are 8-bit per the recipe overrides.
  - Ensure tokenizer + preprocessor_config (VLM) copied into the output dir.
  - Return StageResult noting which fixups were applied.
"""

from __future__ import annotations

from pathlib import Path

from .types import ModelConfig, StageResult


def apply_fixups(cfg: ModelConfig, mlx_dir: Path) -> StageResult:  # noqa: ARG001
    raise NotImplementedError("T2: implement MTP/DeltaNet/gate fixups")
