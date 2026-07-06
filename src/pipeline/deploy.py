"""Stage 6: deploy to ~/.omlx/models and smoke-test under oMLX.

Owner: T3. See PLAN.md §2.6.

TODO(T3):
  - Copy mlx_dir -> ~/.omlx/models/<cfg.org>/<cfg.output_name>/ (atomic: temp
    then rename). Do not overwrite an existing dir without --force.
  - Start/queue an omlx load of the model; wait for "Loaded model" in the log.
  - Send one chat completion via the OpenAI-compatible endpoint; assert the
    response is coherent (non-empty, no repetition loop) and tok/s >= min_tok_s.
  - For VLMs, include one image in the smoke prompt to exercise the vision path.
  - Return StageResult with metrics {resident_gb, tok_s}.
"""

from __future__ import annotations

from pathlib import Path

from .types import ModelConfig, StageResult


def deploy(cfg: ModelConfig, mlx_dir: Path) -> StageResult:  # noqa: ARG001
    raise NotImplementedError("T3: implement deploy + omlx smoke test")
