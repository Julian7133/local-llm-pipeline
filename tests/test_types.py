"""Unit tests for types.py contracts: MemoryBudget and ValidationReport."""

from dataclasses import replace

import pytest

from pipeline.recipe import UNSLOTH_UD_3BIT
from pipeline.types import MemoryBudget, ModelConfig, ModelType, ValidationReport


def test_memory_budget_usable_gb_defaults() -> None:
    budget = MemoryBudget()
    assert budget.total_ram_gb == pytest.approx(24.0)
    assert budget.reserve_gb == pytest.approx(3.0)
    assert budget.usable_gb == pytest.approx(21.0)


def test_memory_budget_usable_gb_custom() -> None:
    budget = MemoryBudget(total_ram_gb=32.0, reserve_gb=5.0)
    assert budget.usable_gb == pytest.approx(27.0)


def _model_config(*, max_regression: float = 0.10) -> ModelConfig:
    return ModelConfig(
        name="test",
        hf_repo="org/model",
        model_type=ModelType.TEXT,
        org="org",
        output_name="out",
        recipe=replace(UNSLOTH_UD_3BIT, max_perplexity_regression=max_regression),
    )


def test_validation_report_passed_when_within_threshold() -> None:
    cfg = _model_config(max_regression=0.10)
    report = ValidationReport(
        base_perplexity=10.0,
        quant_perplexity=10.8,
        regression=0.08,
        est_weights_gb=17.0,
        est_kv_gb_at_min_ctx=3.0,
        fits_budget=True,
    )
    assert report.passed(cfg) is True


def test_validation_report_passed_at_exact_threshold() -> None:
    cfg = _model_config(max_regression=0.10)
    report = ValidationReport(
        base_perplexity=10.0,
        quant_perplexity=11.0,
        regression=0.10,
        est_weights_gb=17.0,
        est_kv_gb_at_min_ctx=3.0,
        fits_budget=True,
    )
    assert report.passed(cfg) is True


def test_validation_report_fails_on_regression() -> None:
    cfg = _model_config(max_regression=0.10)
    report = ValidationReport(
        base_perplexity=10.0,
        quant_perplexity=11.5,
        regression=0.15,
        est_weights_gb=17.0,
        est_kv_gb_at_min_ctx=3.0,
        fits_budget=True,
    )
    assert report.passed(cfg) is False


def test_validation_report_fails_on_memory_fit() -> None:
    cfg = _model_config(max_regression=0.10)
    report = ValidationReport(
        base_perplexity=10.0,
        quant_perplexity=10.5,
        regression=0.05,
        est_weights_gb=19.0,
        est_kv_gb_at_min_ctx=4.0,
        fits_budget=False,
    )
    assert report.passed(cfg) is False
