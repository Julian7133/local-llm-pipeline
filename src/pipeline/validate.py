"""Stage 5: validate quality + memory fit.

Owner: T3. See PLAN.md §2.5 and types.MemoryBudget.

Two gates, BOTH must pass (see ValidationReport.passed):
  1. Perplexity regression vs BF16 base <= recipe.max_perplexity_regression.
  2. Memory fit: quantized weights + KV cache at cfg.memory.min_context_tokens
     (at cfg.memory.kv_cache_bits) must fit within cfg.memory.usable_gb.
     This gate is why we stay at 3-bit: 4-bit weights + real KV won't fit 24GB.

TODO(T3):
  - Perplexity: run a small fixed calibration set through base (BF16) and the
    quantized model; compute mean NLL; regression = (q_ppl - base_ppl)/base_ppl.
    Base ppl may be measured once and cached (BF16 is large — allow SSD stream
    or a proxy layer-truncated estimate; document the method chosen).
  - Quick eval: one text task; for VLMs one image->text task. Record notes.
  - Memory: est_weights_gb from file sizes; est_kv_gb from layers * kv_heads *
    head_dim * 2 * min_context_tokens * (kv_cache_bits/8). fits_budget =
    (weights + kv) <= usable_gb.
  - Return ValidationReport.
"""

from __future__ import annotations

from pathlib import Path

from .types import ModelConfig, ValidationReport


def validate(cfg: ModelConfig, mlx_dir: Path) -> ValidationReport:  # noqa: ARG001
    raise NotImplementedError("T3: implement perplexity + memory-fit validation")
