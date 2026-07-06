"""Unit tests for validate helpers (T3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.recipe import UNSLOTH_UD_3BIT
from pipeline.types import MemoryBudget, ModelConfig, ModelType
from pipeline.validate import (
    BASE_PPL_METHOD,
    calibration_hash,
    estimate_kv_gb,
    estimate_weights_gb,
    is_coherent_output,
    memory_fits,
    perplexity_regression,
    validate,
)


def _cfg(tmp_path: Path, **overrides) -> ModelConfig:
    defaults = dict(
        name="test-model",
        hf_repo="org/model",
        model_type=ModelType.TEXT,
        org="org",
        output_name="out",
        recipe=UNSLOTH_UD_3BIT,
        memory=MemoryBudget(),
        work_dir=tmp_path / "work",
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_estimate_weights_gb_counts_safetensors(tmp_path: Path) -> None:
    mlx = tmp_path / "mlx"
    mlx.mkdir()
    (mlx / "a.safetensors").write_bytes(b"x" * 1024)
    (mlx / "b.safetensors").write_bytes(b"y" * 2048)
    (mlx / "readme.txt").write_text("ignore", encoding="utf-8")
    assert estimate_weights_gb(mlx) == pytest.approx(3072 / (1024**3))


def test_estimate_kv_gb_uses_text_config(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    model_config = {
        "text_config": {
            "num_hidden_layers": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
    }
    kv = estimate_kv_gb(cfg, model_config)
    expected_bytes = 40 * 8 * 128 * 2 * (4 / 8) * 8192
    assert kv == pytest.approx(expected_bytes / (1024**3))


def test_memory_fits_passes_within_budget(tmp_path: Path) -> None:
    mlx = tmp_path / "mlx"
    mlx.mkdir()
    (mlx / "config.json").write_text(
        json.dumps({"num_hidden_layers": 2, "num_key_value_heads": 2, "head_dim": 8}),
        encoding="utf-8",
    )
    (mlx / "tiny.safetensors").write_bytes(b"z" * 1024)
    cfg = _cfg(
        tmp_path,
        memory=MemoryBudget(total_ram_gb=24.0, reserve_gb=3.0, min_context_tokens=128),
    )
    weights, kv, fits = memory_fits(cfg, mlx)
    assert weights > 0
    assert kv > 0
    assert fits is True


def test_memory_fits_fails_when_kv_exceeds_budget(tmp_path: Path) -> None:
    mlx = tmp_path / "mlx"
    mlx.mkdir()
    (mlx / "config.json").write_text(
        json.dumps({"num_hidden_layers": 80, "num_key_value_heads": 32, "head_dim": 256}),
        encoding="utf-8",
    )
    (mlx / "weights.safetensors").write_bytes(b"w" * 4096)
    cfg = _cfg(
        tmp_path,
        memory=MemoryBudget(
            total_ram_gb=24.0,
            reserve_gb=3.0,
            min_context_tokens=1_000_000,
        ),
    )
    _, _, fits = memory_fits(cfg, mlx)
    assert fits is False


def test_perplexity_regression_fraction() -> None:
    assert perplexity_regression(100.0, 108.0) == pytest.approx(0.08)


def test_is_coherent_output_rejects_repetition() -> None:
    loop = "abcd " * 40
    assert is_coherent_output(loop) is False
    assert is_coherent_output("Hello from the smoke test.") is True


def test_calibration_hash_stable() -> None:
    assert calibration_hash() == calibration_hash()
    assert len(calibration_hash()) == 16


def test_validate_requires_mlx_dir(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(FileNotFoundError, match="quantized model dir"):
        validate(cfg, tmp_path / "missing")


def test_base_ppl_method_documents_cache() -> None:
    assert "eval_ppl" in BASE_PPL_METHOD
    assert "cached" in BASE_PPL_METHOD
