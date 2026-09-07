# MA-Hard baselines

Layout, matching `baseline-plans.md`:

```
benchmarks/
  common.py             shared: item selection, Lean extraction, blv verify, CriticLean judging, metrics
  generators.py         chat backends (OpenAI-compatible endpoint, or offline vLLM)
  single-pass/          existing control arm
  iterative/            B1: compile-repair loop
  agentic/              A1: Claude Code in a real Lake project
```

Both runners write the same output shape as `single-pass/run_benchmark.py`
(`uuid`, `file_id`, `type`, `text`, `parsed_output`, `compiler_output`, `verified`,
`aligned`, ...), so `scripts/run_alignment_score.py` still works on their output if
you want to re-judge separately.

## Running everything at once

`scripts/run_ma_hard_baselines.sh` runs all four lanes and writes `BASELINE-RESULTS.md`.
The two GPU-hosted generators run sequentially on the H200 pair while the API model and
the Claude Code agent (neither needs a GPU) run in parallel alongside them; judging comes
last, once the GPUs are free for CriticLean.

```
GPUs:    [ gpt-oss-120b ][ Qwen3-30B-A3B ]          [ CriticLean judge ]
no GPU:  [ gpt-5-mini ........ ][ claude code ..... ]
```

```bash
# smoke test first -- 6 items, real models, a few dollars
N_EXAMPLES=6 SUBSET_FILE="" scripts/run_ma_hard_baselines.sh

# full run
SUBSET_FILE=ma_hard_uuids.json MATH_ATLAS_MCP=./math-atlas-mcp.json \
    scripts/run_ma_hard_baselines.sh
```

Everything is configured by environment variable (see the top of the script):
`SUBSET_FILE`/`FILTER`/`N_EXAMPLES` for selection, `MAX_ROUNDS`, `GPUS`, `TP_SIZE`,
`LEAN_PROJECT`, `MAX_BUDGET_USD`, `JUDGE_MODEL` / `JUDGE_MODEL_PATH`, and `SKIP_GPT_OSS` / `SKIP_QWEN` /
`SKIP_API` / `SKIP_AGENT` / `SKIP_JUDGE` to run one lane at a time.

The GPU-hosted models are served from the official vLLM docker image, one container at a
time (`VLLM_IMAGE`, default `vllm/vllm-openai:latest`; `VLLM_CONTAINER`, default
`ma-hard-vllm`). The container gets `--gpus '"device=$GPUS"'`, `--ipc=host`, port
`$PORT` published, and `$HF_CACHE` (default `$HF_HOME`) mounted at
`/root/.cache/huggingface` so weights are shared with the host. `HF_TOKEN` is forwarded
when set, and `VLLM_DOCKER_ARGS` passes anything else through (e.g. `--shm-size=32g`).
Container logs land in `$LOG_DIR/server.<model>.log`.

Notes:
- Output names are derived from the model, so changing `QWEN_MOE`/`GPT_OSS`/`API_MODEL`
  renames the run rather than overwriting the previous model's results:
  `Qwen/Qwen3.8-Flash-Next` -> `outputs/iterative/qwen3.8-flash-next.ma-hard.json`.
- It refuses to start if `blv` isn't reachable, and prompts before the agent lane's spend
  (`ASSUME_YES=1` for unattended runs, e.g. under `nohup`).
- Finished outputs are skipped, so re-running resumes rather than redoing work. This is
  also how you judge later: run once with `SKIP_JUDGE=1`, then again without it.
- `RUN_CONTROL=1` (default) additionally runs each iterative model at `--max-rounds 1`,
  giving the single-pass control under identical scoring.
- Summarize at any time without re-running:
  `python benchmarks/summarize_results.py outputs/iterative outputs/agentic --output BASELINE-RESULTS.md`

## Prerequisites

1. **Lean verification** — `blv` needs Redis + `rq` workers running (`redis` on
   `localhost:6379` by default; override with `--redis-host/--redis-port/--redis-db`).
2. **Alignment judge** — serve `m-a-p/CriticLeanGPT-Qwen3-32B-RL` on an OpenAI-compatible endpoint and pass
   `--judge-model-url`. Skip with `--skip-judge` to get compile rate only.
3. **A1 only** — a Lake project with Mathlib (`--project`), plus `uvx lean-lsp-mcp`
   on PATH and the `claude` CLI logged in.

## Selecting MA-Hard

MA-Hard can be addressed three ways, and they compose:

```bash
--dataset offendo/math-atlas-hard            # its own dataset
--dataset offendo/math-atlas --split hard    # its own split
--filter split=hard --filter difficulty=hard # a column filter (repeatable)
--subset-file ma_hard_uuids.json             # explicit uuid list (JSON list / JSONL / txt)
```

`--item-type definition --item-type theorem` narrows by type; `--n-examples N` subsamples
for debugging.

## B1 — iterative compile-repair

Generate → compile → feed the Lean errors back → repeat, up to `--max-rounds`.
Round 1 runs at `--temperature`, repairs at `--retry-temperature` (0.0 tends to
re-emit the same broken code). `--max-history-rounds` bounds context growth on
32k-window local models.

```bash
python benchmarks/iterative/run_iterative.py \
    --model openai/gpt-oss-120b --model-url http://localhost:8000/v1 \
    --dataset offendo/math-atlas --filter split=hard \
    --max-rounds 5 --temperature 0.0 --retry-temperature 0.7 \
    --judge-model m-a-p/CriticLeanGPT-Qwen3-32B-RL --judge-model-url http://localhost:8001/v1 \
    --output outputs/iterative/gpt-oss-120b.ma-hard.json
```

Omit `--model-url` to load the model in-process with vLLM (`--tensor-parallel-size 2`
for the H200 pair). `--max-rounds 1` reproduces the single-pass control under identical
scoring, which is the honest way to compute the B1 delta.

## A1 — Claude Code

Each item becomes its own module (`<Lib>/MAHard/Item_<uuid>.lean`) in a real Lake
project, so the agent gets true Lean feedback through the lean-lsp MCP and can import
items it formalized earlier. `ma_hard_index.json` in the project root catalogues those
modules for reuse.

```bash
python benchmarks/agentic/run_claude_code.py \
    --dataset offendo/math-atlas --filter split=hard \
    --project ~/src/mathatlas-formalization --model sonnet \
    --mcp-config ./math-atlas-mcp.json \
    --max-budget-usd 0.75 --timeout 900 \
    --judge-model m-a-p/CriticLeanGPT-Qwen3-32B-RL --judge-model-url http://localhost:8001/v1 \
    --output outputs/agentic/claude-code-sonnet.ma-hard.json
```

- The project is created with `lake init <Name> math` if `--project` doesn't exist.
  `--no-build-project` skips the `lake exe cache get` + `lake build` warm-up.
- `--mathatlas-project ~/src/mathatlas-formalization` wires in the MathAtlas MCP (see
  below). `--mcp-config` is repeatable for any further servers, and `--strict-mcp-config`
  (default) keeps ambient user-level MCP servers out of the run.
- Budget caps: `--max-budget-usd` per item and `--timeout` wall clock. **Report both
  with your numbers** — an uncapped agentic score isn't comparable to anything.
- `--concurrency` > 1 runs agents in parallel over one shared project; expect `lake`
  lock contention and cross-item races. Default is 1.
- `--reset-items` wipes previous item modules to measure the no-reuse condition.
- `--resume` skips uuids already in `--output` and merges.

### The MathAtlas MCP

`benchmarks/agentic/mathatlas_mcp.py` is a thin wrapper over the read-only data layer in
the formalization project (`atlas.mcp.mathatlas.data`), launched inside that project's
environment. It exposes all six upstream tools — `mathatlas_get_item`, `get_context`,
`get_dependencies`, `get_proofs`, `list_items`, `stats` — over the 71,064-item dataset.

The one difference from upstream: **the mathlib grounding of the item under test is
withheld.** `get_item` normally returns `mathlib_suggestion`, the Mathlib declaration the
dataset already grounded that concept to (the first item in the dataset, "fractional
ideal", resolves to `Mathlib.RingTheory.FractionalIdeal`). Handing that to an agent scored
on formalizing the same item is answer leakage — compile rate rises for a reason unrelated
to the model, and A1 stops being comparable to the single-pass control. Dependencies keep
their grounding, so reuse of prerequisites still works.

The item under test is baked into each item's generated MCP config (`MA_HARD_ITEM_UUID`)
rather than inherited from the runner's environment, so the redaction cannot silently
no-op and stays correct under `--concurrency > 1`.

Point `--mathatlas-data` / `--mathatlas-textbooks` elsewhere to run against a different
snapshot; by default the server resolves them from the project's own `data/`.

### Scoring under reuse

If the agent imports an earlier item module, `blv` (which runs outside the project)
would fail it. `--inline-imports` (default) splices reused project-local modules into
the snippet before verification, so genuine reuse gets credit while the item is still
checked against a plain Mathlib environment.

## Judging separately

Both runners judge inline, but with one GPU pair you serve the generator and the judge at
different times. Run generation with `--skip-judge`, then:

```bash
python benchmarks/judge_results.py \
    --input outputs/iterative/gpt-oss-120b.ma-hard.json \
    --judge-model m-a-p/CriticLeanGPT-Qwen3-32B-RL --judge-model-url http://localhost:8000/v1
```

It attaches `aligned`/`alignment_output`, recomputes the metrics in place, and keeps the
generation config recorded in `<output>.metrics.json`.

## Metrics

Written to `<output>.metrics.json` alongside the run config:

- `compile_rate` — fraction that compiles under `import Mathlib` / `import Aesop`.
- `aligned_rate_of_compiling` — CriticLean verdict among compiling items.
- **`joint_rate`** — compiles AND aligned. This is the headline; compile rate alone is
  trivially inflated by degenerate output.
- `degenerate_rate` — cheap syntactic guard (`: True`, `:= trivial`, near-empty). Not
  used to drop rows; report it, since both iterative and agentic systems drift toward
  satisfying the compiler.
- `compile_at_k` (B1) — fraction solved within k rounds, i.e. the saturation curve.
- `total_cost_usd` / `mean_turns` / `total_duration_s` (A1).
