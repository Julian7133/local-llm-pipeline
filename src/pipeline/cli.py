"""CLI entrypoint: `quantize <stage|run> configs/<model>.yaml`.

Owner: T1. Wires config parsing + stage dispatch. Stages raise NotImplemented
until their owning thread lands; `run` executes fetch->quantize->fixups->
validate->deploy in order, stopping on the first failed StageResult.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .deploy import deploy
from .fetch import fetch, _snapshot_complete
from .fixups import apply_fixups
from .quantize import quantize, _mlx_complete
from .recipe import UNSLOTH_UD_3BIT
from .types import MemoryBudget, ModelConfig, ModelType, StageResult
from .validate import validate

STAGES = ("fetch", "quantize", "fixups", "validate", "deploy")

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
    if isinstance(recipe_name, str):
        recipe_key = recipe_name
    else:
        raise ValueError(f"recipe must be a string, got {type(recipe_name).__name__}")
    recipe = RECIPE_BY_NAME.get(recipe_key)
    if recipe is None:
        known = ", ".join(sorted(RECIPE_BY_NAME))
        raise ValueError(f"unknown recipe {recipe_key!r}; known: {known}")

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


def base_dir(cfg: ModelConfig) -> Path:
    return cfg.work_dir / cfg.name / "base"


def mlx_dir(cfg: ModelConfig) -> Path:
    return cfg.work_dir / cfg.name / "mlx"



def _run_fetch(cfg: ModelConfig, *, force: bool) -> StageResult:
    cached = base_dir(cfg)
    if not force and _snapshot_complete(cached):
        return StageResult(
            stage="fetch",
            ok=True,
            message=f"skipped (cached at {cached})",
            artifact_path=cached,
        )
    result = fetch(cfg)
    if result.ok and result.artifact_path is None:
        return StageResult(
            stage=result.stage,
            ok=result.ok,
            message=result.message,
            artifact_path=cached,
            metrics=result.metrics,
        )
    return result


def _run_quantize(cfg: ModelConfig, source: Path, *, force: bool) -> StageResult:
    out = mlx_dir(cfg)
    if not force and _mlx_complete(out):
        return StageResult(
            stage="quantize",
            ok=True,
            message=f"skipped (cached at {out})",
            artifact_path=out,
        )
    result = quantize(cfg, source)
    if result.ok and result.artifact_path is None:
        return StageResult(
            stage=result.stage,
            ok=result.ok,
            message=result.message,
            artifact_path=out,
            metrics=result.metrics,
        )
    return result


def _run_fixups(cfg: ModelConfig, mlx: Path, *, force: bool) -> StageResult:
    # Fixups mutate the mlx tree in place; re-run when forced or config changed.
    if not force and (mlx / "config.json").exists():
        return StageResult(
            stage="fixups",
            ok=True,
            message=f"skipped (config present at {mlx})",
            artifact_path=mlx,
        )
    result = apply_fixups(cfg, mlx)
    if result.ok and result.artifact_path is None:
        return StageResult(
            stage=result.stage,
            ok=result.ok,
            message=result.message,
            artifact_path=mlx,
            metrics=result.metrics,
        )
    return result


def _run_validate(cfg: ModelConfig, mlx: Path) -> StageResult:
    report = validate(cfg, mlx)
    ok = report.passed(cfg)
    message = report.eval_notes or (
        f"ppl regression {report.regression:.1%} "
        f"(base={report.base_perplexity:.2f}, quant={report.quant_perplexity:.2f}); "
        f"weights≈{report.est_weights_gb:.1f}GB kv≈{report.est_kv_gb_at_min_ctx:.1f}GB "
        f"fits={report.fits_budget}"
    )
    return StageResult(
        stage="validate",
        ok=ok,
        message=message,
        artifact_path=mlx,
        metrics={
            "base_perplexity": report.base_perplexity,
            "quant_perplexity": report.quant_perplexity,
            "regression": report.regression,
            "est_weights_gb": report.est_weights_gb,
            "est_kv_gb_at_min_ctx": report.est_kv_gb_at_min_ctx,
            "fits_budget": report.fits_budget,
        },
    )


def _run_deploy(cfg: ModelConfig, mlx: Path, *, force: bool) -> StageResult:
    deploy_target = Path("~/.omlx/models").expanduser() / cfg.org / cfg.output_name
    if not force and deploy_target.exists():
        return StageResult(
            stage="deploy",
            ok=True,
            message=f"skipped (already deployed at {deploy_target})",
            artifact_path=deploy_target,
        )
    result = deploy(cfg, mlx)
    if result.ok and result.artifact_path is None:
        return StageResult(
            stage=result.stage,
            ok=result.ok,
            message=result.message,
            artifact_path=deploy_target,
            metrics=result.metrics,
        )
    return result


def run_pipeline(
    cfg: ModelConfig,
    *,
    from_stage: str | None = None,
    force: bool = False,
) -> StageResult:
    """Execute pipeline stages in order, stopping on the first failure."""
    if from_stage is not None:
        if from_stage not in STAGES:
            known = ", ".join(STAGES)
            raise ValueError(f"unknown stage {from_stage!r}; known: {known}")
        start = STAGES.index(from_stage)
        stages = STAGES[start:]
    else:
        stages = STAGES

    source = base_dir(cfg) if base_dir(cfg).exists() else None
    mlx = mlx_dir(cfg) if mlx_dir(cfg).exists() else None

    for stage in stages:
        if stage == "fetch":
            result = _run_fetch(cfg, force=force)
            if result.artifact_path is not None:
                source = result.artifact_path
        elif stage == "quantize":
            if source is None:
                source = base_dir(cfg)
            result = _run_quantize(cfg, source, force=force)
            if result.artifact_path is not None:
                mlx = result.artifact_path
        elif stage == "fixups":
            if mlx is None:
                mlx = mlx_dir(cfg)
            result = _run_fixups(cfg, mlx, force=force)
            if result.artifact_path is not None:
                mlx = result.artifact_path
        elif stage == "validate":
            if mlx is None:
                mlx = mlx_dir(cfg)
            result = _run_validate(cfg, mlx)
        elif stage == "deploy":
            if mlx is None:
                mlx = mlx_dir(cfg)
            result = _run_deploy(cfg, mlx, force=force)
        else:
            raise AssertionError(f"unhandled stage: {stage}")

        if not result.ok:
            return result

    return StageResult(stage="run", ok=True, message="all stages completed", artifact_path=mlx)


def _dispatch(command: str, cfg: ModelConfig, *, force: bool) -> StageResult:
    if command == "fetch":
        return _run_fetch(cfg, force=force)
    if command == "quantize":
        source = base_dir(cfg)
        return _run_quantize(cfg, source, force=force)
    if command == "fixups":
        return _run_fixups(cfg, mlx_dir(cfg), force=force)
    if command == "validate":
        return _run_validate(cfg, mlx_dir(cfg))
    if command == "deploy":
        return _run_deploy(cfg, mlx_dir(cfg), force=force)
    raise ValueError(f"unknown command: {command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantize",
        description="UD-style 3-bit MLX quantization pipeline for oMLX serving",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    force_stages = {"fetch", "quantize", "fixups", "deploy", "run"}
    for name in STAGES + ("run",):
        stage_parser = sub.add_parser(name, help=f"Run the {name} stage")
        stage_parser.add_argument(
            "config",
            type=Path,
            help="Path to configs/<model>.yaml",
        )
        if name in force_stages:
            stage_parser.add_argument(
                "--force",
                action="store_true",
                help="Re-run even when cached artifacts exist",
            )
        if name == "run":
            stage_parser.add_argument(
                "--from",
                dest="from_stage",
                choices=STAGES,
                metavar="STAGE",
                help="Start at this stage (earlier artifacts resolved from cache paths)",
            )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    force = bool(getattr(args, "force", False))

    try:
        if args.command == "run":
            result = run_pipeline(cfg, from_stage=args.from_stage, force=force)
        else:
            result = _dispatch(args.command, cfg, force=force)
    except (ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if result.message:
        stream = sys.stdout if result.ok else sys.stderr
        print(f"[{result.stage}] {result.message}", file=stream)
    if not result.ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
