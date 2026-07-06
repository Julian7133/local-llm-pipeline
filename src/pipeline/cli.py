"""CLI entrypoint: `quantize <stage|run> configs/<model>.yaml`.

Owner: T1. Wires config parsing + stage dispatch. Stages raise NotImplemented
until their owning thread lands; `run` executes fetch->quantize->fixups->
validate->deploy in order, stopping on the first failed StageResult.

TODO(T1):
  - load_config(path) -> ModelConfig (parse yaml, map recipe name -> QuantRecipe,
    model_type -> ModelType, nested memory -> MemoryBudget).
  - argparse subcommands: fetch, quantize, fixups, validate, deploy, run.
  - `run` threads artifact paths between stages; honor --force / --from <stage>.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from .recipe import UNSLOTH_UD_3BIT
from .types import MemoryBudget, ModelConfig, ModelType

RECIPE_BY_NAME = {
    "unsloth-ud-3bit": UNSLOTH_UD_3BIT,
}


def load_config(path: Path) -> ModelConfig:
    """Parse a YAML model spec into a :class:`ModelConfig`."""
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")

    name = raw.get("name") or path.stem

    recipe_name = raw.get("recipe", "unsloth-ud-3bit")
    if not isinstance(recipe_name, str):
        raise ValueError(f"recipe must be a string, got {type(recipe_name).__name__}")
    recipe = RECIPE_BY_NAME.get(recipe_name)
    if recipe is None:
        known = ", ".join(sorted(RECIPE_BY_NAME))
        raise ValueError(f"unknown recipe {recipe_name!r}; known: {known}")

    memory_raw = raw.get("memory") or {}
    if not isinstance(memory_raw, dict):
        raise ValueError("memory must be a mapping when present")
    memory = MemoryBudget(
        total_ram_gb=float(memory_raw.get("total_ram_gb", 24.0)),
        reserve_gb=float(memory_raw.get("reserve_gb", 3.0)),
        kv_cache_bits=int(memory_raw.get("kv_cache_bits", 4)),
        min_context_tokens=int(memory_raw.get("min_context_tokens", 8192)),
    )

    work_dir_raw = raw.get("work_dir", "~/.cache/llm-quant-pipeline")
    work_dir = Path(str(work_dir_raw)).expanduser()

    return ModelConfig(
        name=str(name),
        hf_repo=str(raw["hf_repo"]),
        model_type=ModelType(str(raw["model_type"])),
        org=str(raw["org"]),
        output_name=str(raw["output_name"]),
        recipe=recipe,
        memory=memory,
        strip_mtp_config=bool(raw.get("strip_mtp_config", False)),
        min_tok_s=float(raw.get("min_tok_s", 5.0)),
        work_dir=work_dir,
    )


def main(argv: list[str] | None = None) -> int:  # noqa: ARG001
    raise NotImplementedError("T1: argparse subcommands + run orchestration")


if __name__ == "__main__":
    sys.exit(main())
