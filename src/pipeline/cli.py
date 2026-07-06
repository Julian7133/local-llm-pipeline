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

from .types import ModelConfig


def load_config(path: Path) -> ModelConfig:  # noqa: ARG001
    raise NotImplementedError("T1: parse yaml -> ModelConfig")


def main(argv: list[str] | None = None) -> int:  # noqa: ARG001
    raise NotImplementedError("T1: argparse subcommands + run orchestration")


if __name__ == "__main__":
    sys.exit(main())
