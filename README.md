# local-llm-pipeline

Adapt large MoE / VLM models to run on a 24 GB Apple Silicon Mac by quantizing
them with a **UD-style 3-bit per-layer mixed-bit recipe** into **MLX** format,
then serving them under **oMLX**.

3-bit is the target (not 4-bit): on 24 GB unified memory, 4-bit weights plus a
usable KV cache do not fit. The pipeline treats *weights + KV memory fit* as a
hard validation gate. See [`PLAN.md`](PLAN.md) for the full spec and the recipe
table ported from `unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit`.

## Quick start

```bash
bash scripts/setup.sh                 # mlx-lm (git main), mlx-vlm, hf_hub
make all MODEL=qwen-agentworld-35b    # fetch -> quantize -> validate -> deploy
make test                             # recipe unit tests
```

Models deploy to `~/.omlx/models/<org>/<name>/` and serve via `omlx`.

## Status

- `src/pipeline/recipe.py` — **complete** (the deterministic UD recipe + tests).
- Everything else — skeleton with `NotImplementedError` + thread-tagged TODOs.
  See `PLAN.md` §4 for the T1–T4 work breakdown.
