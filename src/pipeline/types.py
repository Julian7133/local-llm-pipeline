"""Shared contracts. T1 owns this file; T2-T4 code against it.

Keep this dependency-light (stdlib + recipe). Everything crossing a stage
boundary is one of these dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from .recipe import QuantRecipe, UNSLOTH_UD_3BIT


class ModelType(str, Enum):
    VLM = "vlm"      # -> mlx_vlm.convert / serve
    TEXT = "text"    # -> mlx_lm.convert / serve


@dataclass
class MemoryBudget:
    """The 24GB constraint, made explicit.

    3-bit is the ceiling precisely because weights + KV cache must fit. A run
    that quantizes fine but can't hold a usable context is a failed run.
    """

    total_ram_gb: float = 24.0
    reserve_gb: float = 3.0          # OS + app headroom
    kv_cache_bits: int = 4           # oMLX TurboQuant KV
    min_context_tokens: int = 8192   # must fit this much KV alongside weights

    @property
    def usable_gb(self) -> float:
        return self.total_ram_gb - self.reserve_gb


@dataclass
class ModelConfig:
    """Parsed from configs/<name>.yaml."""

    name: str                         # short id, == config filename stem
    hf_repo: str                      # source HF repo (BF16 base)
    model_type: ModelType
    org: str                          # deploy under ~/.omlx/models/<org>/<name>
    output_name: str                  # deployed dir name
    recipe: QuantRecipe = field(default_factory=lambda: UNSLOTH_UD_3BIT)
    memory: MemoryBudget = field(default_factory=MemoryBudget)
    strip_mtp_config: bool = False    # default: leave MTP for oMLX runtime patch
    min_tok_s: float = 5.0            # smoke-test throughput floor
    work_dir: Path = Path("~/.cache/llm-quant-pipeline").expanduser()


@dataclass
class StageResult:
    stage: str
    ok: bool
    message: str = ""
    artifact_path: Optional[Path] = None
    metrics: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    base_perplexity: float
    quant_perplexity: float
    regression: float                 # fraction vs base
    est_weights_gb: float
    est_kv_gb_at_min_ctx: float
    fits_budget: bool
    eval_notes: str = ""

    def passed(self, cfg: ModelConfig) -> bool:
        return (
            self.regression <= cfg.recipe.max_perplexity_regression
            and self.fits_budget
        )
