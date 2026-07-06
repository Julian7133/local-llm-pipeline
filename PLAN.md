# local-llm-pipeline — Plan & Spec

Build a reproducible CLI pipeline that adapts large MoE / VLM models to run on a
24 GB Apple Silicon Mac by quantizing them with a **UD-style per-layer mixed-bit
recipe** into **MLX** format, then serving them under **oMLX**.

This document is the authoritative spec. Child threads implement against it.

---

## 0. Context (verified facts)

### Serving stack: oMLX (`omlx`)
`omlx` is a serving daemon that wraps **mlx-vlm** and adds:
- Model loading from `~/.omlx/models/<org>/<name>/`
- Paged SSD KV cache + TurboQuant 4-bit KV cache (long context on 24 GB)
- Vision feature cache (SSD) for VLMs
- OpenAI-compatible server (`omlx.server`), tool calling (`qwen3_coder` parser)
- Runtime patch that tolerates a config declaring MTP heads when the checkpoint
  ships no `mtp.*` weights (MTPModule attachment skipped for strict load)

Reference run: `Qwen3.6-35B-A3B-UD-MLX-3bit` → 16.27 GB resident, ~7.4 tok/s.

### The UD recipe (extracted from the reference model's config.json)
Target architecture: Qwen3.5/3.6 `*_moe`, 40 layers, hybrid
`3×(Gated DeltaNet + MoE) then 1×(Gated Attention + MoE)`, 256 experts / 3B active,
1 shared expert + routed experts per MoE layer.

| Component                              | Bits | Notes                    |
|----------------------------------------|------|--------------------------|
| Switchable MoE experts (default)       | 3    | group_size 64, affine    |
| Switchable MoE experts, last layer 39  | 4    | last-layer bump          |
| Shared-expert projections              | 4    |                          |
| Attention projections (q/k/v/o)        | 8    |                          |
| Router / gate layers, shared_expert_gate | 8  |                          |
| Embeddings (`embed_tokens`)            | 6    |                          |
| `lm_head`                              | 6    |                          |
| DeltaNet `A_log`, conv1d, norms        | —    | NOT quantized            |
| Vision tower (VLM)                     | 6/8  | keep high; validate      |

This exact map is implemented in `src/pipeline/recipe.py` (`UNSLOTH_UD_3BIT`).

---

## 1. Target models

| Config                        | Model                                   | Type | Path    |
|-------------------------------|-----------------------------------------|------|---------|
| `qwen3.6-35b-a3b.yaml`        | unsloth/Qwen3.6-35B-A3B (base BF16)     | VLM  | mlx-vlm |
| `agents-a1.yaml`              | InternScience/Agents-A1                 | VLM  | mlx-vlm |
| `qwen-agentworld-35b.yaml`    | Qwen-AgentWorld-35B-A3B (base BF16)     | text | mlx-lm  |

VLM vs text selects the convert/serve backend. All share the same MoE recipe.

---

## 2. Pipeline stages (CLI: `quantize run configs/<model>.yaml`)

1. **fetch** — `huggingface_hub` download of the BF16 base; disk-space guard
   (35B BF16 ≈ 70 GB; require free space before download).
2. **recipe** — build the per-layer quant map from `UNSLOTH_UD_3BIT`, resolved
   against the model's actual layer names → an mlx `quant_predicate` +
   per-layer `config.json` overrides identical in shape to the reference.
3. **quantize** — dispatch by type:
   - VLM  → `mlx_vlm.convert` (quantizes language + vision, writes config)
   - text → `mlx_lm.convert`
   Apply the recipe via predicate; write mixed-bit overrides into config.json.
4. **fixups** — MTP head handling (strip config vs. leave for oMLX runtime
   patch — validate both, default: leave); ensure `A_log`/DeltaNet/norms are
   skipped; verify `shared_expert_gate` + router at 8-bit.
5. **validate** — perplexity delta vs BF16 base on a small calibration set +
   quick functional eval (a text task; for VLMs one image→text task). Fail the
   run if perplexity regression exceeds config threshold.
6. **deploy** — copy to `~/.omlx/models/<org>/<name>/`, then `omlx` load +
   chat smoke test asserting coherent output and a min tok/s.

Each stage is independently runnable and idempotent; artifacts cached on disk.

---

## 3. Repo layout

```
configs/<model>.yaml           per-model spec (source repo, type, recipe, thresholds)
src/pipeline/
  __init__.py
  fetch.py       # HF download + disk guard
  recipe.py      # UD recipe → mlx predicate + config overrides   [CORE, provided]
  quantize.py    # mlx-vlm / mlx-lm dispatch
  fixups.py      # MTP, A_log/DeltaNet skips, gate bits
  validate.py    # perplexity + quick eval
  deploy.py      # copy to ~/.omlx/models + omlx smoke test
  cli.py         # `quantize run/fetch/quantize/validate/deploy`
  types.py       # dataclasses: ModelConfig, QuantRecipe, StageResult
tests/           # unit tests per module (recipe map, config parsing, predicate)
scripts/         # helper shell scripts (install omlx, env setup)
Makefile         # make fetch/quantize/validate/deploy/all MODEL=agents-a1
pyproject.toml
```

---

## 4. Child-thread work breakdown

- **T1 — core & config** (`types.py`, `cli.py`, `configs/*.yaml`, `pyproject`,
  `Makefile`): CLI wiring, config schema, dataclasses. Owns the contracts other
  threads code against. `recipe.py` is already provided — do not rewrite it,
  only wire it in.
- **T2 — fetch & quantize** (`fetch.py`, `quantize.py`, `fixups.py`): HF
  download + disk guard; mlx-vlm/mlx-lm dispatch; apply recipe predicate; MTP &
  DeltaNet fixups. The hardest thread (arch quirks).
- **T3 — validate & deploy** (`validate.py`, `deploy.py`, oMLX smoke test):
  perplexity + quick eval; deploy to `~/.omlx/models`; omlx load & chat test.
- **T4 — tests & docs** (`tests/*`, `README.md`, `scripts/`): unit tests for the
  recipe map + config parsing + predicate resolution; install/setup scripts.

Contracts live in `types.py` (T1). Threads T2–T4 depend on it; T1 lands first.

### Per-model quirks (discovered during predicate testing)

- **MLX `switch_mlp` vs HF `experts.*`** — mlx-lm exports routed experts as
  `mlp.switch_mlp.{gate,up,down}_proj` rather than per-expert `mlp.experts.N.*`.
  The recipe detects the router by the module *named* `gate` (suffix match, not
  the `mlp.gate` substring), so `switch_mlp.gate_proj` is correctly an expert
  projection (3-bit; 4-bit on the last layer). `_looks_like_expert` recognizes
  both `experts.*` and `switch_mlp.*` so the last-layer bump applies to each.
- **Vision tower attention** — vision `q/k/v/o_proj` paths match the global
  attention rule (8-bit) before the vision_tower rule (6-bit).
- **VLM `language_model.` prefix** — mlx-vlm VLMs prefix language weights with
  `language_model.`; substring rules still match.
- **MTP config without weights** — leave `strip_mtp_config: false` (default);
  oMLX tolerates the declared-but-missing MTP block at serve time.

---

## 5. Definition of done

`make all MODEL=qwen-agentworld-35b` downloads, quantizes to the UD recipe,
validates within the perplexity threshold, deploys to `~/.omlx/models`, and the
oMLX smoke test returns coherent output. Repeat for the two VLM configs.
