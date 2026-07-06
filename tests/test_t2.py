"""Unit tests for T2 fetch, quantize, and fixups stages."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from pipeline.fetch import (
    DISK_HEADROOM_RATIO,
    base_dir,
    estimate_download_gb,
    fetch,
    resolve_hf_repo,
)
from pipeline.fixups import apply_fixups, expected_gate_bits
from pipeline.quantize import merge_quantization_config, mlx_dir, quantize
from pipeline.recipe import UNSLOTH_UD_3BIT, config_overrides
from pipeline.types import ModelConfig, ModelType


def _cfg(
    *,
    name: str = "test-model",
    model_type: ModelType = ModelType.TEXT,
    strip_mtp: bool = False,
    work_dir: Path | None = None,
) -> ModelConfig:
    return ModelConfig(
        name=name,
        hf_repo="org/model-Base",
        model_type=model_type,
        org="org",
        output_name="out",
        recipe=UNSLOTH_UD_3BIT,
        strip_mtp_config=strip_mtp,
        work_dir=work_dir or Path("/tmp/llm-quant-test"),
    )


def _write_minimal_hf_snapshot(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type": "qwen3_moe"}', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"\x00" * 16)
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def _write_minimal_mlx_tree(path: Path, *, quantize_gates: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "qwen3_moe",
        "text_config": {
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
        },
        "quantization": {
            "group_size": 64,
            "bits": 3,
            "mode": "affine",
            "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8},
            "language_model.model.layers.0.mlp.shared_expert_gate": {
                "group_size": 64,
                "bits": 8,
            },
        },
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    weights = {
        "language_model.model.layers.0.linear_attn.A_log": np.zeros(1, dtype=np.float16),
        "language_model.model.layers.0.linear_attn.conv1d.weight": np.zeros(1, dtype=np.float16),
        "language_model.model.layers.0.linear_attn.dt_bias": np.zeros(1, dtype=np.float16),
        "language_model.model.layers.0.linear_attn.norm.weight": np.zeros(1, dtype=np.float16),
        "language_model.model.layers.0.mlp.gate.weight": np.zeros(1, dtype=np.float16),
        "language_model.model.layers.0.mlp.shared_expert_gate.weight": np.zeros(
            1, dtype=np.float16
        ),
    }
    if quantize_gates:
        weights.update(
            {
                "language_model.model.layers.0.mlp.gate.scales": np.zeros(1, dtype=np.float16),
                "language_model.model.layers.0.mlp.gate.biases": np.zeros(1, dtype=np.float16),
                "language_model.model.layers.0.mlp.shared_expert_gate.scales": np.zeros(
                    1, dtype=np.float16
                ),
                "language_model.model.layers.0.mlp.shared_expert_gate.biases": np.zeros(
                    1, dtype=np.float16
                ),
            }
        )

    from safetensors.numpy import save_file

    save_file(weights, path / "model.safetensors")


def test_resolve_hf_repo_strips_base_suffix() -> None:
    api = MagicMock()
    not_found = RepositoryNotFoundError("missing", response=MagicMock(status_code=404))
    api.model_info.side_effect = [not_found, None]
    with patch("pipeline.fetch._repo_candidates", return_value=["org/model-Base", "org/model"]):
        assert resolve_hf_repo("org/model-Base", api=api) == "org/model"


def test_fetch_idempotent_when_snapshot_complete(tmp_path: Path) -> None:
    cfg = _cfg(work_dir=tmp_path)
    dest = base_dir(cfg)
    _write_minimal_hf_snapshot(dest)

    result = fetch(cfg)
    assert result.ok
    assert result.artifact_path == dest
    assert "cached" in result.message


def test_partial_snapshot_is_incomplete(tmp_path: Path) -> None:
    """An interrupted download (index present, a shard missing) must NOT read
    as complete — otherwise the pipeline quantizes on missing weights."""
    from pipeline.fetch import _snapshot_complete

    dest = tmp_path / "base"
    dest.mkdir()
    (dest / "config.json").write_text("{}", encoding="utf-8")
    (dest / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"a.weight": "model-00001-of-00002.safetensors",
                            "b.weight": "model-00002-of-00002.safetensors"}}
        ),
        encoding="utf-8",
    )
    (dest / "model-00001-of-00002.safetensors").write_bytes(b"\x00")  # only 1 of 2
    assert _snapshot_complete(dest) is False

    (dest / "model-00002-of-00002.safetensors").write_bytes(b"\x00")  # now complete
    assert _snapshot_complete(dest) is True


def test_fetch_checks_disk_space(tmp_path: Path) -> None:
    cfg = _cfg(work_dir=tmp_path)
    with patch("pipeline.fetch._snapshot_complete", return_value=False), patch(
        "pipeline.fetch._check_disk_space",
        side_effect=OSError("insufficient disk space"),
    ):
        result = fetch(cfg)
    assert not result.ok
    assert "insufficient disk space" in result.message


def test_fetch_downloads_when_missing(tmp_path: Path) -> None:
    cfg = _cfg(work_dir=tmp_path)
    dest = base_dir(cfg)

    def fake_download(*, repo_id: str, local_dir: str) -> str:
        _ = repo_id
        _write_minimal_hf_snapshot(Path(local_dir))
        return local_dir

    with patch("pipeline.fetch.resolve_hf_repo", return_value="org/model"), patch(
        "pipeline.fetch.snapshot_download", side_effect=fake_download
    ):
        result = fetch(cfg)

    assert result.ok
    assert result.artifact_path == dest
    assert dest.exists()


def test_estimate_download_gb_defaults_to_70() -> None:
    assert estimate_download_gb(_cfg()) == pytest.approx(70.0)


def test_merge_quantization_config_preserves_per_layer_entries() -> None:
    config = {
        "quantization": {
            "bits": 4,
            "language_model.model.layers.0.mlp.gate": {"bits": 8, "group_size": 64},
        }
    }
    merge_quantization_config(config, UNSLOTH_UD_3BIT)
    quant = config["quantization"]
    assert quant["bits"] == 3
    assert quant["group_size"] == 64
    assert quant["overrides"]["attention"]["bits"] == 8
    assert quant["language_model.model.layers.0.mlp.gate"]["bits"] == 8


def test_merge_matches_config_overrides_defaults() -> None:
    config: dict = {}
    merge_quantization_config(config, UNSLOTH_UD_3BIT)
    expected = config_overrides(UNSLOTH_UD_3BIT)
    assert config["quantization"]["bits"] == expected["bits"]
    assert config["quantization"]["overrides"] == expected["overrides"]


def test_quantize_skips_cached_output(tmp_path: Path) -> None:
    cfg = _cfg(work_dir=tmp_path)
    out = mlx_dir(cfg)
    _write_minimal_mlx_tree(out)

    result = quantize(cfg, base_dir(cfg))
    assert result.ok
    assert "cached" in result.message


def test_quantize_dispatches_mlx_lm(tmp_path: Path) -> None:
    cfg = _cfg(work_dir=tmp_path)
    source = base_dir(cfg)
    _write_minimal_hf_snapshot(source)
    out = mlx_dir(cfg)

    def fake_run_convert(model_cfg: ModelConfig, source_dir: Path, dest: Path) -> None:
        assert model_cfg.model_type is ModelType.TEXT
        assert source_dir == source
        _write_minimal_mlx_tree(dest)

    with patch("pipeline.quantize._run_convert", side_effect=fake_run_convert):
        result = quantize(cfg, source)

    assert result.ok
    assert result.artifact_path == out
    merged = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert merged["quantization"]["bits"] == 3
    assert merged["quantization"]["overrides"]["router_gate"]["bits"] == 8


def test_quantize_dispatches_mlx_vlm_for_vlm(tmp_path: Path) -> None:
    cfg = _cfg(work_dir=tmp_path, model_type=ModelType.VLM)
    source = base_dir(cfg)
    _write_minimal_hf_snapshot(source)

    seen: dict = {}

    def fake_run_convert(model_cfg: ModelConfig, source_dir: Path, dest: Path) -> None:
        seen["model_type"] = model_cfg.model_type
        _write_minimal_mlx_tree(dest)

    with patch("pipeline.quantize._run_convert", side_effect=fake_run_convert):
        result = quantize(cfg, source)

    assert result.ok
    assert seen["model_type"] is ModelType.VLM


def test_apply_fixups_leaves_mtp_by_default(tmp_path: Path) -> None:
    cfg = _cfg(work_dir=tmp_path, strip_mtp=False)
    source = base_dir(cfg)
    _write_minimal_hf_snapshot(source)
    out = mlx_dir(cfg)
    _write_minimal_mlx_tree(out)

    result = apply_fixups(cfg, out)
    assert result.ok
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["text_config"]["mtp_num_hidden_layers"] == 1


def test_apply_fixups_strips_mtp_when_configured(tmp_path: Path) -> None:
    cfg = _cfg(work_dir=tmp_path, strip_mtp=True)
    source = base_dir(cfg)
    _write_minimal_hf_snapshot(source)
    out = mlx_dir(cfg)
    _write_minimal_mlx_tree(out)

    result = apply_fixups(cfg, out)
    assert result.ok
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert "mtp_num_hidden_layers" not in config.get("text_config", {})


def test_apply_fixups_rejects_quantized_deltanet(tmp_path: Path) -> None:
    cfg = _cfg(work_dir=tmp_path)
    source = base_dir(cfg)
    _write_minimal_hf_snapshot(source)
    out = mlx_dir(cfg)
    _write_minimal_mlx_tree(out)

    weights_path = out / "model.safetensors"
    from safetensors.numpy import load_file, save_file

    weights = load_file(weights_path)
    weights["language_model.model.layers.0.linear_attn.A_log.scales"] = np.zeros(
        1, dtype=np.float16
    )
    save_file(weights, weights_path)

    result = apply_fixups(cfg, out)
    assert not result.ok
    assert "unquantized" in result.message


def test_apply_fixups_requires_gate_quantization(tmp_path: Path) -> None:
    cfg = _cfg(work_dir=tmp_path)
    source = base_dir(cfg)
    _write_minimal_hf_snapshot(source)
    out = mlx_dir(cfg)
    _write_minimal_mlx_tree(out, quantize_gates=False)

    result = apply_fixups(cfg, out)
    assert not result.ok
    assert "8-bit" in result.message


def test_apply_fixups_copies_tokenizer(tmp_path: Path) -> None:
    cfg = _cfg(work_dir=tmp_path, model_type=ModelType.VLM)
    source = base_dir(cfg)
    _write_minimal_hf_snapshot(source)
    (source / "preprocessor_config.json").write_text('{"do_resize": true}', encoding="utf-8")
    out = mlx_dir(cfg)
    _write_minimal_mlx_tree(out)

    result = apply_fixups(cfg, out)
    assert result.ok
    assert (out / "tokenizer.json").is_file()
    assert (out / "preprocessor_config.json").is_file()


def test_expected_gate_bits_is_8() -> None:
    assert expected_gate_bits() == 8


def test_disk_headroom_ratio() -> None:
    assert DISK_HEADROOM_RATIO == pytest.approx(1.1)
