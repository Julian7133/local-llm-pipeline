"""Stage 4: architecture fixups after quantization.

Owner: T2. See PLAN.md §2.4 and §0 (arch quirks).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .fetch import base_dir
from .recipe import UNSLOTH_UD_3BIT
from .types import ModelConfig, StageResult

_SKIP_PATTERNS = ("A_log", "dt_bias", "conv1d", "norm")
_GATE_PATH_RE = re.compile(r"\.(gate|shared_expert_gate)(\.|$)")
_MTP_KEY_RE = re.compile(r"mtp", re.IGNORECASE)

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)
_VLM_FILES = ("preprocessor_config.json", "processor_config.json", "chat_template.json")


def _iter_weight_keys(mlx_dir: Path) -> list[str]:
    keys: list[str] = []
    index = mlx_dir / "model.safetensors.index.json"
    if index.is_file():
        with index.open(encoding="utf-8") as fh:
            data = json.load(fh)
        keys.extend(data.get("weight_map", {}).keys())
    for shard in sorted(mlx_dir.glob("*.safetensors")):
        try:
            from safetensors import safe_open
        except ImportError:
            continue
        with safe_open(shard, framework="numpy") as fh:
            keys.extend(fh.keys())
    return keys


def _is_quantized(keys: set[str], stem: str) -> bool:
    return f"{stem}.scales" in keys or f"{stem}.biases" in keys


def _strip_mtp_keys(obj: dict) -> tuple[dict, list[str]]:
    removed: list[str] = []
    cleaned: dict = {}
    for key, value in obj.items():
        if _MTP_KEY_RE.search(key):
            removed.append(key)
            continue
        if isinstance(value, dict):
            nested, nested_removed = _strip_mtp_keys(value)
            cleaned[key] = nested
            removed.extend(f"{key}.{r}" for r in nested_removed)
        else:
            cleaned[key] = value
    return cleaned, removed


def _apply_mtp_fixup(cfg: ModelConfig, config: dict) -> list[str]:
    if not cfg.strip_mtp_config:
        return []
    cleaned, removed = _strip_mtp_keys(config)
    config.clear()
    config.update(cleaned)
    return removed


def _assert_deltanet_unquantized(keys: list[str]) -> None:
    key_set = set(keys)
    violations: list[str] = []
    for key in keys:
        if not any(pat in key for pat in _SKIP_PATTERNS):
            continue
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        stem = key.removesuffix(".weight") if key.endswith(".weight") else key
        if _is_quantized(key_set, stem):
            violations.append(key)
    if violations:
        sample = ", ".join(violations[:5])
        raise AssertionError(
            f"DeltaNet/norm tensors must stay unquantized; found scales for: {sample}"
        )


def _gate_bits_from_config(config: dict, key: str) -> int | None:
    quant = config.get("quantization", {})
    if not isinstance(quant, dict):
        return None
    entry = quant.get(key)
    if isinstance(entry, dict) and "bits" in entry:
        return int(entry["bits"])
    return None


def _assert_gate_8bit(keys: list[str], config: dict) -> None:
    key_set = set(keys)
    violations: list[str] = []
    for key in keys:
        if not _GATE_PATH_RE.search(key):
            continue
        if not key.endswith(".weight"):
            continue
        stem = key.removesuffix(".weight")
        if not _is_quantized(key_set, stem):
            violations.append(f"{key} (not quantized)")
            continue
        bits = _gate_bits_from_config(config, stem)
        if bits is not None and bits != 8:
            violations.append(f"{stem} ({bits}-bit)")

    if violations:
        sample = ", ".join(violations[:5])
        raise AssertionError(f"router/shared_expert_gate must be 8-bit; issues: {sample}")


def _copy_sidecar_files(source: Path, dest: Path, names: tuple[str, ...]) -> list[str]:
    copied: list[str] = []
    for name in names:
        src = source / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            copied.append(name)
    return copied


def apply_fixups(cfg: ModelConfig, mlx_dir: Path) -> StageResult:
    config_path = mlx_dir / "config.json"
    if not config_path.is_file():
        return StageResult(
            stage="fixups",
            ok=False,
            message=f"missing config.json in {mlx_dir}",
        )

    applied: list[str] = []
    try:
        with config_path.open(encoding="utf-8") as fh:
            config = json.load(fh)

        removed = _apply_mtp_fixup(cfg, config)
        if removed:
            applied.append(f"stripped_mtp:{len(removed)}")

        weight_keys = _iter_weight_keys(mlx_dir)
        _assert_deltanet_unquantized(weight_keys)
        applied.append("deltanet_unquantized:ok")

        _assert_gate_8bit(weight_keys, config)
        applied.append("gate_8bit:ok")

        source = base_dir(cfg)
        copied = _copy_sidecar_files(source, mlx_dir, _TOKENIZER_FILES)
        if cfg.model_type.value == "vlm":
            copied.extend(_copy_sidecar_files(source, mlx_dir, _VLM_FILES))
        if copied:
            applied.append(f"copied:{','.join(copied)}")

        with config_path.open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
            fh.write("\n")
    except AssertionError as exc:
        return StageResult(stage="fixups", ok=False, message=str(exc))
    except Exception as exc:  # noqa: BLE001
        return StageResult(stage="fixups", ok=False, message=f"fixup failed: {exc}")

    return StageResult(
        stage="fixups",
        ok=True,
        message="; ".join(applied) if applied else "no fixups required",
        artifact_path=mlx_dir,
        metrics={"fixups": applied},
    )


def expected_gate_bits() -> int:
    """Recipe contract used by gate assertions (router + shared_expert_gate)."""
    return UNSLOTH_UD_3BIT.bits_for("model.layers.0.mlp.gate.weight") or 8
