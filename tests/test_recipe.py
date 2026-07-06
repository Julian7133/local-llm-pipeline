"""Tests for the UD recipe map — the deterministic core (recipe.py is complete).

These assert the ported unsloth UD 3-bit bit allocation. If they fail, the
recipe drifted from the reference config and the produced model will not match
the known-good 17.4GB build.
"""

from pipeline.recipe import (
    UNSLOTH_UD_3BIT as R,
    build_predicate,
    config_overrides,
)


def test_defaults():
    assert R.default_bits == 3
    assert R.group_size == 64
    assert R.mode == "affine"


def test_routed_expert_is_3bit():
    assert R.bits_for("model.layers.5.mlp.experts.12.down_proj.weight") == 3


def test_last_layer_experts_bumped_to_4bit():
    assert R.bits_for("model.layers.39.mlp.experts.0.up_proj.weight") == 4


def test_attention_is_8bit():
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert R.bits_for(f"model.layers.3.self_attn.{p}.weight") == 8


def test_router_and_shared_gate_are_8bit():
    assert R.bits_for("model.layers.5.mlp.gate.weight") == 8
    assert R.bits_for("model.layers.5.mlp.shared_expert_gate.weight") == 8


def test_shared_expert_is_4bit():
    assert R.bits_for("model.layers.5.mlp.shared_expert.down_proj.weight") == 4


def test_embeddings_and_head_are_6bit():
    assert R.bits_for("model.embed_tokens.weight") == 6
    assert R.bits_for("lm_head.weight") == 6


def test_deltanet_state_is_skipped():
    for p in ("A_log", "conv1d.weight", "dt_bias", "input_layernorm.weight"):
        assert R.bits_for(f"model.layers.0.linear_attn.{p}") is None


def test_predicate_returns_skip_and_dicts():
    pred = build_predicate(R)
    assert pred("model.layers.0.linear_attn.A_log", None, {}) is False
    got = pred("model.layers.3.self_attn.q_proj.weight", None, {})
    assert got == {"bits": 8, "group_size": 64, "mode": "affine"}


def test_config_overrides_shape():
    ov = config_overrides(R)
    assert ov["bits"] == 3 and ov["group_size"] == 64
    assert ov["overrides"]["attention"]["bits"] == 8
    assert ov["overrides"]["deltanet_state"]["skip"] is True
