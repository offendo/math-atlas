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

## Prerequisites

1. **Lean verification** — `blv` needs Redis + `rq` workers running (`redis` on
   `localhost:6379` by default; override with `--redis-host/--redis-port/--redis-db`).
2. **Alignment judge** — serve CriticLean-32B on an OpenAI-compatible endpoint and pass
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
    --judge-model criticleangpt-qwen3-32b-rl --judge-model-url http://localhost:8001/v1 \
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
    --judge-model criticleangpt-qwen3-32b-rl --judge-model-url http://localhost:8001/v1 \
    --output outputs/agentic/claude-code-sonnet.ma-hard.json
```

- The project is created with `lake init <Name> math` if `--project` doesn't exist.
  `--no-build-project` skips the `lake exe cache get` + `lake build` warm-up.
- `--mcp-config` is repeatable and merges into the generated lean-lsp config — this is
  where the MathAtlas dependency/context MCP goes. `--strict-mcp-config` (default) keeps
  ambient user-level MCP servers out of the run.
- Budget caps: `--max-budget-usd` per item and `--timeout` wall clock. **Report both
  with your numbers** — an uncapped agentic score isn't comparable to anything.
- `--concurrency` > 1 runs agents in parallel over one shared project; expect `lake`
  lock contention and cross-item races. Default is 1.
- `--reset-items` wipes previous item modules to measure the no-reuse condition.
- `--resume` skips uuids already in `--output` and merges.

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
    --judge-model criticleangpt-qwen3-32b-rl --judge-model-url http://localhost:8000/v1
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
