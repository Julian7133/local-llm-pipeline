"""Stage 6: deploy to ~/.omlx/models and smoke-test under oMLX.

Owner: T3. See PLAN.md §2.6.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .types import ModelConfig, ModelType, StageResult
from .validate import is_coherent_output

OMLX_MODELS_ROOT = Path("~/.omlx/models").expanduser()
LOADED_MODEL_RE = re.compile(
    r"Loaded model:\s*(?P<model>[^\s(]+).*?actual:\s*(?P<resident>[\d.]+)\s*GB",
    re.IGNORECASE,
)

# 1×1 PNG for VLM smoke tests.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def deploy_target(cfg: ModelConfig) -> Path:
    return OMLX_MODELS_ROOT / cfg.org / cfg.output_name


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def atomic_copytree(src: Path, dst: Path) -> None:
    """Copy *src* to *dst* via a sibling temp dir + rename (atomic on one volume)."""
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"source model dir not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(tmp, dst)


def parse_loaded_model_line(line: str) -> tuple[str, float] | None:
    match = LOADED_MODEL_RE.search(line)
    if match is None:
        return None
    return match.group("model"), float(match.group("resident"))


def wait_for_loaded_model(
    log_path: Path,
    model_id: str,
    *,
    timeout_s: float = 900.0,
    poll_s: float = 0.5,
) -> float:
    """Tail *log_path* until oMLX logs 'Loaded model: <model_id>'. Returns resident GB."""
    deadline = time.monotonic() + timeout_s
    cursor = 0
    while time.monotonic() < deadline:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if len(text) > cursor:
                for line in text[cursor:].splitlines():
                    parsed = parse_loaded_model_line(line)
                    if parsed is not None and parsed[0] == model_id:
                        return parsed[1]
                cursor = len(text)
        time.sleep(poll_s)
    raise TimeoutError(f"timed out waiting for Loaded model: {model_id} in {log_path}")


def _http_json(
    method: str,
    url: str,
    payload: dict | None = None,
    *,
    timeout_s: float = 120.0,
) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _wait_for_health(base_url: str, *, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            _http_json("GET", f"{base_url}/health", timeout_s=5.0)
            return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(0.5)
    raise TimeoutError(f"omlx server did not become healthy at {base_url}")


def _smoke_messages(cfg: ModelConfig) -> list[dict]:
    if cfg.model_type is ModelType.VLM:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one short sentence."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{TINY_PNG_B64}"},
                    },
                ],
            }
        ]
    return [{"role": "user", "content": "Say hello in one short sentence."}]


def _extract_tok_s(response: dict, elapsed_s: float) -> float:
    usage = response.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if completion_tokens > 0 and elapsed_s > 0:
        return completion_tokens / elapsed_s

    # oMLX extended timing fields (when present).
    for key in ("generation_tokens_per_second", "tokens_per_second", "tok_s"):
        if key in response and response[key]:
            return float(response[key])

    choices = response.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        approx_tokens = max(1, len(text.split()))
        return approx_tokens / max(elapsed_s, 1e-6)
    return 0.0


def _chat_completion(base_url: str, model_id: str, messages: list[dict]) -> tuple[str, float]:
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 64,
        "temperature": 0.2,
    }
    started = time.monotonic()
    response = _http_json("POST", f"{base_url}/v1/chat/completions", payload)
    elapsed = time.monotonic() - started

    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"chat completion returned no choices: {response}")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not is_coherent_output(content):
        raise RuntimeError(f"incoherent deploy smoke output: {content[:200]!r}")

    tok_s = _extract_tok_s(response, elapsed)
    return content, tok_s


def _terminate_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def deploy(cfg: ModelConfig, mlx_dir: Path) -> StageResult:
    mlx_dir = Path(mlx_dir).expanduser().resolve()
    if not mlx_dir.exists():
        return StageResult(
            stage="deploy",
            ok=False,
            message=f"quantized model dir not found: {mlx_dir}",
        )

    target = deploy_target(cfg)
    try:
        atomic_copytree(mlx_dir, target)
    except OSError as exc:
        return StageResult(stage="deploy", ok=False, message=f"deploy copy failed: {exc}")

    omlx_bin = shutil.which("omlx")
    if omlx_bin is None:
        return StageResult(
            stage="deploy",
            ok=False,
            message="omlx not found on PATH; install omlx before deploy",
            artifact_path=target,
        )

    port = find_free_port()
    log_path = target.parent / f".{cfg.output_name}.deploy-{os.getpid()}.log"
    base_url = f"http://127.0.0.1:{port}"
    model_id = cfg.output_name

    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            omlx_bin,
            "serve",
            "--model-dir",
            str(OMLX_MODELS_ROOT),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "info",
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    resident_gb = 0.0
    tok_s = 0.0
    sample = ""
    try:
        _wait_for_health(base_url)
        # Chat triggers on-demand load; also try explicit admin load when available.
        try:
            _http_json("POST", f"{base_url}/admin/api/models/{model_id}/load", timeout_s=30.0)
        except urllib.error.HTTPError:
            pass

        resident_gb = wait_for_loaded_model(log_path, model_id)
        sample, tok_s = _chat_completion(base_url, model_id, _smoke_messages(cfg))
        if tok_s < cfg.min_tok_s:
            return StageResult(
                stage="deploy",
                ok=False,
                message=(
                    f"tok/s {tok_s:.2f} below floor {cfg.min_tok_s:.2f}; "
                    f"sample={sample[:120]!r}"
                ),
                artifact_path=target,
                metrics={"resident_gb": resident_gb, "tok_s": tok_s},
            )
    except (TimeoutError, urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
        return StageResult(
            stage="deploy",
            ok=False,
            message=f"omlx smoke test failed: {exc}",
            artifact_path=target,
            metrics={"resident_gb": resident_gb, "tok_s": tok_s},
        )
    finally:
        log_fh.close()
        _terminate_process(proc)
        if log_path.exists():
            log_path.unlink(missing_ok=True)

    return StageResult(
        stage="deploy",
        ok=True,
        message=f"deployed to {target}; sample={sample[:120]!r}",
        artifact_path=target,
        metrics={"resident_gb": resident_gb, "tok_s": tok_s},
    )
