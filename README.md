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
bash scripts/verify_omlx.sh           # confirm omlx + ~/.omlx/models (after oMLX install)
make test                             # recipe, config, types, predicate tests
make all MODEL=qwen-agentworld-35b    # fetch -> quantize -> validate -> deploy
```

Models deploy to `~/.omlx/models/<org>/<name>/` and serve via `omlx`.

## Status

| Module | Status |
|--------|--------|
| `recipe.py` | **Complete** — UD 3-bit recipe + predicate |
| `types.py`, `cli.load_config` | **Complete** — contracts + YAML parsing |
| `fetch/quantize/fixups/validate/deploy` | **Implemented** — HF fetch, mlx-vlm/mlx-lm convert, fixups, perplexity+memory-fit validation, oMLX deploy smoke test |
| `tests/` | Config, types, recipe, predicate unit tests |

See `PLAN.md` §4 for the T1–T4 work breakdown.

## Per-model quirks

- **Qwen3.5/3.6 MoE hybrid stack** — 40 layers alternating Gated DeltaNet
  (`linear_attn.*`) and full attention (`self_attn.*`). DeltaNet state tensors
  (`A_log`, `conv1d`, `dt_bias`, norms) are skipped; projections quantize at
  8-bit via the `in_proj` / `out_proj` rules.
- **MLX expert naming** — after convert, routed experts may appear as
  `mlp.switch_mlp.*` rather than `mlp.experts.*`. The recipe detects the router
  by the module being *named* `gate` (suffix match), so `switch_mlp.gate_proj`
  is correctly treated as an expert projection (3-bit, 4-bit on layer 39) and
  not misrouted to the 8-bit router rule.
- **Vision tower attention** — vision `q_proj`/`k_proj`/etc. match the global
  attention rule (8-bit) before the vision_tower rule (6-bit) due to rule order.
- **VLM prefix** — mlx-vlm VLMs carry a `language_model.` prefix on language
  weights; the recipe matches on suffix substrings so the same rules apply.
- **MTP heads** — Qwen3.6 configs may declare MTP layers without shipping
  weights. Default: leave the config entry; oMLX skips strict MTP load at runtime.
