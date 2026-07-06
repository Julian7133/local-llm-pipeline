"""Unit tests for deploy helpers (T3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.deploy import (
    atomic_copytree,
    deploy_target,
    find_free_port,
    parse_loaded_model_line,
)
from pipeline.recipe import UNSLOTH_UD_3BIT
from pipeline.types import MemoryBudget, ModelConfig, ModelType


def _cfg(tmp_path: Path) -> ModelConfig:
    return ModelConfig(
        name="test-model",
        hf_repo="org/model",
        model_type=ModelType.VLM,
        org="unsloth",
        output_name="Qwen3.6-35B-A3B-UD-MLX-3bit",
        recipe=UNSLOTH_UD_3BIT,
        memory=MemoryBudget(),
        work_dir=tmp_path / "work",
    )


def test_deploy_target_under_omlx_models(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    target = deploy_target(cfg)
    assert target.name == cfg.output_name
    assert target.parent.name == cfg.org
    assert target.parent.parent.name == "models"


def test_atomic_copytree_replaces_existing(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text('{"v": 1}', encoding="utf-8")
    (dst / "old.txt").write_text("old", encoding="utf-8")

    atomic_copytree(src, dst)

    assert (dst / "config.json").read_text(encoding="utf-8") == '{"v": 1}'
    assert not (dst / "old.txt").exists()


def test_atomic_copytree_uses_temp_then_rename(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "org" / "model"
    src.mkdir(parents=True)
    (src / "weights.safetensors").write_bytes(b"abc")

    atomic_copytree(src, dst)

    assert (dst / "weights.safetensors").read_bytes() == b"abc"
    leftovers = list(dst.parent.glob(".*.tmp-*"))
    assert leftovers == []


def test_parse_loaded_model_line_extracts_resident_gb() -> None:
    line = (
        "2026-06-03 12:49:02,299 - omlx.engine_pool - INFO - "
        "Loaded model: Qwen3.6-35B-A3B-UD-MLX-3bit (actual: 15.47GB, estimated: 16.99GB)"
    )
    parsed = parse_loaded_model_line(line)
    assert parsed == ("Qwen3.6-35B-A3B-UD-MLX-3bit", 15.47)


def test_find_free_port_returns_positive_integer() -> None:
    port = find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535
