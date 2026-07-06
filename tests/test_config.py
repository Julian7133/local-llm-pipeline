"""Config parsing tests — YAML -> ModelConfig via cli.load_config."""

from pathlib import Path

import pytest

from pipeline.cli import load_config
from pipeline.recipe import UNSLOTH_UD_3BIT
from pipeline.types import ModelType

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

CONFIG_FILES = [
    "qwen3.6-35b-a3b.yaml",
    "agents-a1.yaml",
    "qwen-agentworld-35b.yaml",
]


@pytest.mark.parametrize("filename", CONFIG_FILES)
def test_config_parses(filename: str) -> None:
    cfg = load_config(CONFIGS_DIR / filename)
    assert cfg.recipe is UNSLOTH_UD_3BIT
    assert cfg.recipe.default_bits == 3
    assert cfg.memory.usable_gb == pytest.approx(21.0)
    assert cfg.memory.kv_cache_bits == 4
    assert cfg.memory.min_context_tokens == 8192
    assert cfg.strip_mtp_config is False
    assert cfg.min_tok_s == 5.0


def test_qwen_vlm_config_fields() -> None:
    cfg = load_config(CONFIGS_DIR / "qwen3.6-35b-a3b.yaml")
    assert cfg.name == "qwen3.6-35b-a3b"
    assert cfg.hf_repo == "unsloth/Qwen3.6-35B-A3B"
    assert cfg.model_type is ModelType.VLM
    assert cfg.org == "unsloth"
    assert cfg.output_name == "Qwen3.6-35B-A3B-UD-MLX-3bit"


def test_agents_vlm_config_fields() -> None:
    cfg = load_config(CONFIGS_DIR / "agents-a1.yaml")
    assert cfg.name == "agents-a1"
    assert cfg.hf_repo == "InternScience/Agents-A1"
    assert cfg.model_type is ModelType.VLM
    assert cfg.org == "InternScience"
    assert cfg.output_name == "Agents-A1-UD-MLX-3bit"


def test_agentworld_text_config_fields() -> None:
    cfg = load_config(CONFIGS_DIR / "qwen-agentworld-35b.yaml")
    assert cfg.name == "qwen-agentworld-35b"
    assert cfg.model_type is ModelType.TEXT
    assert cfg.output_name == "Qwen-AgentWorld-35B-A3B-UD-MLX-3bit"


def test_nested_memory_mapping(tmp_path: Path) -> None:
    cfg_path = tmp_path / "custom.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: custom",
                "hf_repo: org/model",
                "model_type: text",
                "org: org",
                "output_name: out",
                "memory:",
                "  total_ram_gb: 32.0",
                "  reserve_gb: 4.5",
                "  kv_cache_bits: 4",
                "  min_context_tokens: 16384",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.memory.total_ram_gb == pytest.approx(32.0)
    assert cfg.memory.reserve_gb == pytest.approx(4.5)
    assert cfg.memory.usable_gb == pytest.approx(27.5)
    assert cfg.memory.kv_cache_bits == 4
    assert cfg.memory.min_context_tokens == 16384


def test_recipe_name_maps_to_quant_recipe(tmp_path: Path) -> None:
    cfg_path = tmp_path / "recipe.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "name: recipe-test",
                "hf_repo: org/model",
                "model_type: text",
                "org: org",
                "output_name: out",
                "recipe: unsloth-ud-3bit",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.recipe is UNSLOTH_UD_3BIT
    assert cfg.recipe.name == "unsloth-ud-3bit"
    assert cfg.recipe.max_perplexity_regression == pytest.approx(0.10)


def test_unknown_recipe_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "\n".join(
            [
                "name: bad",
                "hf_repo: org/model",
                "model_type: text",
                "org: org",
                "output_name: out",
                "recipe: not-a-recipe",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown recipe"):
        load_config(bad)
