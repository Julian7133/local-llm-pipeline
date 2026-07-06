"""UD-style per-layer mixed-bit quantization recipe.

This is the deterministic core of the pipeline. The bit map below is ported
verbatim from the reference model's config.json:

    unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit

and reproduces the quality/size tradeoff that makes a 35B-A3B MoE VLM fit in
~17 GB and serve on a 24 GB Mac. See PLAN.md §0 for the source table.

The output is consumed two ways:
  1. As an mlx ``quant_predicate`` callable passed to ``mlx_lm.convert`` /
     ``mlx_vlm.convert`` (returns per-module bit/group settings, or False to
     skip quantization entirely for a module).
  2. As the ``quantization`` block written into the produced config.json so the
     mixed-bit layout is self-describing and reloadable.

Child threads: DO NOT rewrite this map without a source config to back it up.
Wire it in via ``build_predicate`` and ``config_overrides``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

# Default quantization applied to any quantizable module not matched below.
DEFAULT_BITS = 3
DEFAULT_GROUP_SIZE = 64
DEFAULT_MODE = "affine"

# Substrings that identify a module by its (dot-joined) parameter path.
# Order matters: the FIRST matching rule wins, so most-specific rules come first.
#
# Each rule: (match_substrings, bits) or (match_substrings, None) to SKIP.
# ``None`` bits => leave the module in its original (unquantized) precision.
Rule = tuple[tuple[str, ...], Optional[int]]

UD_RULES: tuple[Rule, ...] = (
    # --- never quantize: DeltaNet / linear-attention state & small norms ---
    (("A_log",), None),
    (("dt_bias",), None),
    (("conv1d",), None),
    (("norm",), None),          # rmsnorm / layernorm weights
    (("bias",), None),

    # --- routing / gating: keep high precision (8-bit) ---
    # NB: the MoE router is handled specially in bits_for() (by the module being
    # *named* `gate`, not by substring) so that expert `gate_proj` projections —
    # including the MLX `switch_mlp.gate_proj` stacked layout — are NOT mistaken
    # for the router. Only shared_expert_gate is safe as a plain substring here.
    (("shared_expert_gate",), 8),

    # --- attention projections: 8-bit ---
    (("q_proj", "k_proj", "v_proj", "o_proj",
      "qkv_proj", "in_proj", "out_proj"), 8),

    # --- shared expert MLP projections: 4-bit ---
    (("shared_expert.gate_proj", "shared_expert.up_proj",
      "shared_expert.down_proj", "shared_experts."), 4),

    # --- token embeddings & output head: 6-bit ---
    (("embed_tokens",), 6),
    (("lm_head",), 6),

    # --- vision tower (VLM): keep high; validate downward later ---
    (("vision_model", "visual.", "vision_tower"), 6),

    # switchable/routed experts fall through to DEFAULT (3-bit), with a
    # last-layer bump handled specially in build_predicate().
)

# Layers whose routed experts get bumped above DEFAULT_BITS (from ref config:
# last decoder layer's experts are 4-bit rather than 3-bit).
LAST_LAYER_EXPERT_BITS = 4


@dataclass(frozen=True)
class QuantRecipe:
    """A named per-layer mixed-bit recipe."""

    name: str
    default_bits: int = DEFAULT_BITS
    group_size: int = DEFAULT_GROUP_SIZE
    mode: str = DEFAULT_MODE
    rules: tuple[Rule, ...] = UD_RULES
    last_layer_expert_bits: int = LAST_LAYER_EXPERT_BITS
    num_hidden_layers: int = 40
    # perplexity regression ceiling vs BF16 base, as a fraction (e.g. 0.08 = 8%)
    max_perplexity_regression: float = 0.10

    def bits_for(self, param_path: str) -> Optional[int]:
        """Resolve the bit-width for a module path.

        Returns an int (bits) to quantize, or ``None`` to skip (leave in
        original precision). Applies the last-layer expert bump.
        """
        # The MoE router is the module *named* `gate` (e.g. ...mlp.gate). Match
        # it by suffix, not substring, so expert `gate_proj` (HF `experts.N.*`
        # or MLX `switch_mlp.*`) is never misrouted to 8-bit. Normalize a
        # trailing tensor suffix first, since callers pass both module paths
        # (".../mlp.gate") and parameter paths (".../mlp.gate.weight").
        stem = param_path
        for suffix in (".weight", ".bias", ".scales", ".biases"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if stem.endswith(".gate") or stem.endswith(".block_sparse_moe.gate"):
            return 8

        for substrings, bits in self.rules:
            if any(s in param_path for s in substrings):
                return bits
        # Fell through: this is a routed expert (or generic linear). Apply the
        # last-layer bump if this path belongs to the final decoder layer.
        last = self.num_hidden_layers - 1
        if f".layers.{last}." in param_path and _looks_like_expert(param_path):
            return self.last_layer_expert_bits
        return self.default_bits


def _looks_like_expert(param_path: str) -> bool:
    # HF layout uses per-expert `experts.N.*`; MLX convert stacks them into
    # `switch_mlp.*`. Both are routed experts and get the last-layer bump.
    return any(
        s in param_path
        for s in (
            "experts.",
            "mlp.experts",
            "block_sparse_moe.experts",
            "switch_mlp.",
        )
    )


UNSLOTH_UD_3BIT = QuantRecipe(name="unsloth-ud-3bit")


def build_predicate(recipe: QuantRecipe) -> Callable[[str, object, dict], object]:
    """Return an mlx ``quant_predicate(path, module, config)``.

    mlx contract: return ``False`` to skip a module, or a dict of quantization
    kwargs (``bits``, ``group_size``, ``mode``) to override the defaults for it.
    """

    def predicate(path: str, module: object, config: dict):  # noqa: ARG001
        bits = recipe.bits_for(path)
        if bits is None:
            return False
        return {
            "bits": bits,
            "group_size": recipe.group_size,
            "mode": recipe.mode,
        }

    return predicate


def config_overrides(recipe: QuantRecipe) -> dict:
    """Top-level ``quantization`` block to merge into the produced config.json.

    Mirrors the reference model's layout: a default bits/group_size plus explicit
    per-scope overrides, so the mixed-bit model is self-describing on reload.
    """
    return {
        "group_size": recipe.group_size,
        "bits": recipe.default_bits,
        "mode": recipe.mode,
        "overrides": {
            "embed_tokens": {"bits": 6},
            "lm_head": {"bits": 6},
            "attention": {"bits": 8},
            "router_gate": {"bits": 8},
            "shared_expert_gate": {"bits": 8},
            "shared_expert": {"bits": 4},
            "experts_last_layer": {"bits": recipe.last_layer_expert_bits},
            "vision_tower": {"bits": 6},
            "deltanet_state": {"skip": True},
        },
    }
