"""Stage 1: fetch BF16 base from HF, with a disk-space guard.

Owner: T2. See PLAN.md §2.1.

TODO(T2):
  - Resolve exact hf_repo (some ids in configs are marked "confirm at fetch").
  - Use huggingface_hub.snapshot_download into cfg.work_dir.
  - Guard: 35B BF16 ~= 70GB; raise if free disk < needed * 1.1 before download.
  - Skip download if a complete snapshot already exists (idempotent).
  - Return StageResult with artifact_path = local snapshot dir.
"""

from __future__ import annotations

from .types import ModelConfig, StageResult


def fetch(cfg: ModelConfig) -> StageResult:  # noqa: ARG001
    raise NotImplementedError("T2: implement HF snapshot download + disk guard")
