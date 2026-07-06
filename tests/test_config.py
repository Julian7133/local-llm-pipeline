"""Config parsing tests — T1 contract for configs/*.yaml -> ModelConfig."""

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
