"""Stage 1: fetch BF16 base from HF, with a disk-space guard.

Owner: T2. See PLAN.md §2.1.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError

from .types import ModelConfig, StageResult

# 35B BF16 ~= 70 GB on disk; require 10% headroom before starting a download.
BF16_35B_GB = 70.0
DISK_HEADROOM_RATIO = 1.1


def base_dir(cfg: ModelConfig) -> Path:
    """Local HF snapshot directory for a model (matches cli.base_dir)."""
    return cfg.work_dir / cfg.name / "base"


def _repo_candidates(hf_repo: str) -> list[str]:
    """Build an ordered list of HF repo ids to try."""
    seen: set[str] = set()
    candidates: list[str] = []

    def add(repo: str) -> None:
        if repo and repo not in seen:
            seen.add(repo)
            candidates.append(repo)

    add(hf_repo)
    if hf_repo.endswith("-Base"):
        add(hf_repo[: -len("-Base")])
    if "AgentWorld" in hf_repo and "Qwen-AgentWorld-35B-A3B" not in hf_repo:
        add("Qwen/Qwen-AgentWorld-35B-A3B")
    return candidates


def resolve_hf_repo(hf_repo: str, *, api: HfApi | None = None) -> str:
    """Return the first existing repo id from ``_repo_candidates``."""
    client = api or HfApi()
    errors: list[str] = []
    for repo in _repo_candidates(hf_repo):
        try:
            client.model_info(repo)
            return repo
        except RepositoryNotFoundError:
            errors.append(f"{repo} not found")
        except Exception as exc:  # noqa: BLE001 — surface HF auth/network issues
            raise RuntimeError(f"failed to resolve {repo!r}: {exc}") from exc
    raise ValueError(
        f"no Hugging Face repo found for {hf_repo!r}; tried: {', '.join(_repo_candidates(hf_repo))}"
    )


def estimate_download_gb(cfg: ModelConfig) -> float:
    """Conservative on-disk estimate for the BF16 base snapshot."""
    _ = cfg
    return BF16_35B_GB


def _snapshot_complete(path: Path) -> bool:
    """True only when config + ALL weight shards are present.

    A partially-downloaded snapshot (interrupted mid-fetch) has config.json and
    *some* shards, which previously read as "complete" and let the pipeline run
    quantize on missing weights. When a safetensors index exists, require every
    shard it references; otherwise fall back to any single weight file.
    """
    if not path.is_dir() or not (path / "config.json").is_file():
        return False

    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            with index.open(encoding="utf-8") as fh:
                weight_map = json.load(fh).get("weight_map", {})
        except (OSError, ValueError):
            return False
        shards = set(weight_map.values())
        if not shards:
            return False
        return all((path / shard).is_file() for shard in shards)

    # No index: a single-file checkpoint. Require at least one weight file and
    # no leftover .incomplete markers from an interrupted download.
    if any(path.glob("*.incomplete")):
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def _check_disk_space(target: Path, required_gb: float) -> None:
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    free_gb = usage.free / (1024**3)
    needed_gb = required_gb * DISK_HEADROOM_RATIO
    if free_gb < needed_gb:
        raise OSError(
            f"insufficient disk space at {target}: "
            f"{free_gb:.1f} GB free, need {needed_gb:.1f} GB "
            f"({required_gb:.1f} GB estimate × {DISK_HEADROOM_RATIO})"
        )


def fetch(cfg: ModelConfig) -> StageResult:
    dest = base_dir(cfg)
    dest.mkdir(parents=True, exist_ok=True)

    if _snapshot_complete(dest):
        return StageResult(
            stage="fetch",
            ok=True,
            message=f"using cached snapshot at {dest}",
            artifact_path=dest,
        )

    required_gb = estimate_download_gb(cfg)
    try:
        _check_disk_space(dest, required_gb)
        repo_id = resolve_hf_repo(cfg.hf_repo)
        # snapshot_download resumes partial files and re-verifies existing ones,
        # so re-running after an interrupted fetch continues rather than restarts.
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        return StageResult(stage="fetch", ok=False, message=str(exc))
    except Exception as exc:  # noqa: BLE001 — HF download errors
        return StageResult(stage="fetch", ok=False, message=f"download failed: {exc}")

    if not _snapshot_complete(dest):
        return StageResult(
            stage="fetch",
            ok=False,
            message=f"download of {cfg.hf_repo} incomplete (missing config or weights)",
        )

    resolved = repo_id if repo_id != cfg.hf_repo else cfg.hf_repo
    return StageResult(
        stage="fetch",
        ok=True,
        message=f"downloaded {resolved} -> {dest}",
        artifact_path=dest,
        metrics={"hf_repo": resolved, "estimated_gb": required_gb},
    )
