"""Stage 3: quantize with the UD recipe via mlx-vlm (VLM) or mlx-lm (text).

Owner: T2. See PLAN.md §2.3.

The recipe -> predicate + config overrides is already provided in recipe.py.
This module's job is purely dispatch + wiring, NOT re-deriving bit allocations.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .recipe import QuantRecipe, build_predicate, config_overrides
from .types import ModelConfig, ModelType, StageResult


def mlx_dir(cfg: ModelConfig) -> Path:
    """Quantized MLX output directory (matches cli.mlx_dir)."""
    return cfg.work_dir / cfg.name / "mlx"


def _mlx_complete(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any(path.glob("*.safetensors")) or (
        path / "model.safetensors.index.json"
    ).is_file()


def merge_quantization_config(config: dict[str, Any], recipe: QuantRecipe) -> dict[str, Any]:
    """Merge recipe ``config_overrides`` into the mlx ``quantization`` block."""
    recipe_block = config_overrides(recipe)
    quant = config.get("quantization")
    if not isinstance(quant, dict):
        quant = {}
    merged = dict(quant)

    for key in ("group_size", "bits", "mode"):
        merged[key] = recipe_block[key]

    recipe_overrides = recipe_block.get("overrides", {})
    existing_overrides = merged.get("overrides", {})
    if not isinstance(existing_overrides, dict):
        existing_overrides = {}
    merged["overrides"] = {**recipe_overrides, **existing_overrides}
    config["quantization"] = merged
    if "quantization_config" in config:
        config["quantization_config"] = merged
    return config


def _write_quantization_config(mlx_path: Path, recipe: QuantRecipe) -> None:
    config_path = mlx_path / "config.json"
    with config_path.open(encoding="utf-8") as fh:
        config = json.load(fh)
    merge_quantization_config(config, recipe)
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")


def _run_convert(cfg: ModelConfig, source_dir: Path, out: Path) -> None:
    predicate = build_predicate(cfg.recipe)
    recipe = cfg.recipe
    kwargs = dict(
        hf_path=str(source_dir),
        mlx_path=str(out),
        quantize=True,
        q_group_size=recipe.group_size,
        q_bits=recipe.default_bits,
        q_mode=recipe.mode,
        quant_predicate=predicate,
    )

    if cfg.model_type is ModelType.VLM:
        from mlx_vlm.convert import convert

        convert(**kwargs, trust_remote_code=True)
        return

    from mlx_lm.convert import convert

    convert(**kwargs, trust_remote_code=True)


def quantize(cfg: ModelConfig, source_dir: Path) -> StageResult:
    out = mlx_dir(cfg)
    if _mlx_complete(out):
        return StageResult(
            stage="quantize",
            ok=True,
            message=f"using cached mlx output at {out}",
            artifact_path=out,
        )

    if not _mlx_complete(source_dir) and not (source_dir / "config.json").is_file():
        return StageResult(
            stage="quantize",
            ok=False,
            message=f"source snapshot incomplete or missing: {source_dir}",
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)

    try:
        _run_convert(cfg, source_dir, out)
        _write_quantization_config(out, cfg.recipe)
    except ImportError as exc:
        return StageResult(
            stage="quantize",
            ok=False,
            message=f"convert dependencies missing (run scripts/setup.sh): {exc}",
        )
    except Exception as exc:  # noqa: BLE001 — mlx convert failures
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        return StageResult(stage="quantize", ok=False, message=f"convert failed: {exc}")

    if not _mlx_complete(out):
        return StageResult(
            stage="quantize",
            ok=False,
            message=f"convert produced incomplete output at {out}",
        )

    return StageResult(
        stage="quantize",
        ok=True,
        message=f"quantized {cfg.name} with {cfg.recipe.name} ({cfg.model_type.value})",
        artifact_path=out,
        metrics={
            "model_type": cfg.model_type.value,
            "default_bits": cfg.recipe.default_bits,
            "group_size": cfg.recipe.group_size,
        },
    )
