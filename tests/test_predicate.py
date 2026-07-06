"""Predicate resolution against realistic Qwen3.5/3.6 MoE parameter paths.

Paths mirror HF checkpoint layout (used by mlx-lm/mlx-vlm convert) and the
``language_model.`` prefix VLMs carry during conversion.
"""

from pipeline.recipe import UNSLOTH_UD_3BIT as R, build_predicate

# Hybrid stack: layers 0-2 are Gated DeltaNet (linear_attn), 3+ alternate.
DELTANET_LAYER = 1
FULL_ATTN_LAYER = 7
MOE_LAYER = 5
LAST_LAYER = 39

REALISTIC_PATHS = {
    # DeltaNet state — never quantized
    f"model.layers.{DELTANET_LAYER}.linear_attn.A_log": None,
    f"model.layers.{DELTANET_LAYER}.linear_attn.conv1d.weight": None,
    f"model.layers.{DELTANET_LAYER}.linear_attn.dt_bias": None,
    f"model.layers.{DELTANET_LAYER}.linear_attn.norm.weight": None,
    # DeltaNet projections — 8-bit via in_proj / out_proj rules
    f"model.layers.{DELTANET_LAYER}.linear_attn.in_proj_qkv.weight": 8,
    f"model.layers.{DELTANET_LAYER}.linear_attn.in_proj_z.weight": 8,
    f"model.layers.{DELTANET_LAYER}.linear_attn.out_proj.weight": 8,
    # Full-attention block
    f"model.layers.{FULL_ATTN_LAYER}.self_attn.q_proj.weight": 8,
    f"model.layers.{FULL_ATTN_LAYER}.self_attn.k_proj.weight": 8,
    f"model.layers.{FULL_ATTN_LAYER}.self_attn.v_proj.weight": 8,
    f"model.layers.{FULL_ATTN_LAYER}.self_attn.o_proj.weight": 8,
    # MoE routing + shared expert
    f"model.layers.{MOE_LAYER}.mlp.gate.weight": 8,
    f"model.layers.{MOE_LAYER}.mlp.shared_expert_gate.weight": 8,
    f"model.layers.{MOE_LAYER}.mlp.shared_expert.gate_proj.weight": 4,
    f"model.layers.{MOE_LAYER}.mlp.shared_expert.up_proj.weight": 4,
    f"model.layers.{MOE_LAYER}.mlp.shared_expert.down_proj.weight": 4,
    # Routed experts — 3-bit default, 4-bit on last layer
    f"model.layers.{MOE_LAYER}.mlp.experts.42.down_proj.weight": 3,
    f"model.layers.{MOE_LAYER}.mlp.experts.128.up_proj.weight": 3,
    f"model.layers.{LAST_LAYER}.mlp.experts.0.down_proj.weight": 4,
    f"model.layers.{LAST_LAYER}.mlp.experts.255.gate_proj.weight": 4,
    # Embeddings + head
    "model.embed_tokens.weight": 6,
    "lm_head.weight": 6,
    # VLM language-model prefix (Agents-A1 / Qwen3.6 path)
    f"language_model.model.layers.{FULL_ATTN_LAYER}.self_attn.q_proj.weight": 8,
    f"language_model.model.layers.{MOE_LAYER}.mlp.experts.10.down_proj.weight": 3,
    # Vision tower — q_proj matches attention rule (8-bit) before vision_tower (6-bit)
    "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight": 8,
    # Vision non-attention weight still resolves via vision_tower rule
    "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight": 6,
}


def test_bits_for_realistic_qwen35_moe_paths() -> None:
    for path, expected in REALISTIC_PATHS.items():
        assert R.bits_for(path) == expected, path


def test_predicate_realistic_paths_return_dict_or_false() -> None:
    pred = build_predicate(R)
    for path, expected_bits in REALISTIC_PATHS.items():
        got = pred(path, None, {})
        if expected_bits is None:
            assert got is False, path
        else:
            assert got == {
                "bits": expected_bits,
                "group_size": 64,
                "mode": "affine",
            }, path


def test_mlx_switch_mlp_paths() -> None:
    """MLX export uses switch_mlp.* instead of experts.*."""
    # gate_proj falsely matches mlp.gate substring (switch_mlp.gate) -> 8-bit router rule
    gate_path = f"model.layers.{MOE_LAYER}.mlp.switch_mlp.gate_proj.weight"
    assert R.bits_for(gate_path) == 8

    # up/down projs fall through to 3-bit default; no last-layer bump (not experts.*)
    down_path = f"model.layers.{LAST_LAYER}.mlp.switch_mlp.down_proj.weight"
    assert R.bits_for(down_path) == 3
