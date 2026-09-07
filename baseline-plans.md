# MA-Hard Baseline Plan (iterative + agentic)

Target: ~700 MA-Hard items (definitions, theorem statements, examples). Inputs guaranteed to
every system: `text` + `type`. Everything else (local context, dependency neighbors, Mathlib
search) is *optional* and supplied via MCP — which is exactly the variable we want to isolate.

Existing single-pass numbers (Kimina-7B, Herald, Goedel-V2-8B/32B, gpt-oss-20b/120b,
Qwen3-30B-A3B, GPT-5.x) are the control arm. Everything below is measured as a delta against
the matching single-pass row.

## What the design is trying to separate

Single-pass → agentic conflates four things. The baseline set is chosen so each is attributable:

1. **Sequential feedback** (compiler errors fed back) — B1.
2. **Parallel sampling** (more tries, no feedback) — B3. Without this, any B1 gain is
   unattributable: a 5-round repair loop spends ~5x tokens, and best-of-8 often matches it.
3. **Retrieval / context** (Mathlib names, surrounding book text, dep graph) — B4.
4. **Harness** (planning, self-directed tool use, long horizon) — A1/A2.

B3 is the one people skip and the one reviewers ask for. Keep it.

## Iterative tier

### B1 — Compile-repair loop (the canonical baseline)
Generate → compile → feed verbatim Lean error + failing snippet → regenerate. K=5 rounds max,
temperature 0.7 on retries (0.0 first shot), stop on first compile. Report compile-rate vs.
rounds so you get the saturation curve for free (it usually flattens at K≈3).

Models: `gpt-oss-120b` (local workhorse), `Qwen3-30B-A3B-Thinking-2507` (efficiency point),
`gpt-5-mini` (cheap API), `claude-sonnet-5` (strong API). Skip Opus here — B1 is a scaffold
test, not a frontier-model test.

### B2 — Specialist formalizer in the repair loop
Same loop, but the generator is `Goedel-Formalizer-V2-32B` and `Kimina-Autoformalizer-7B`.
Worth one row each: these are SFT'd for single-pass emission and generally do *not* improve
(sometimes regress) when handed error strings they were never trained on. That negative result
is cheap to obtain and is the main argument for why MA-Hard needs non-specialist baselines.
Optional variant: specialist generates, `gpt-oss-120b` repairs. This is the strongest
cost/performance config in the iterative tier if it works.

### B3 — Sample-and-select (budget-matched to B1)
n=8 at T=0.8, no feedback. Filter to compiling candidates, rank with your existing
`m-a-p/CriticLeanGPT-Qwen3-32B-RL` judge, take top-1. Same total token budget as B1's 5 rounds so the
comparison is honest. Models: `gpt-oss-120b`, `claude-sonnet-5`.

### B4 — Retrieval-augmented repair
B1 + MCP retrieval, no autonomy: fixed pre-generation retrieval of (a) preceding document
context, (b) MathAtlas dependency neighbors' formal statements, (c) Mathlib name lookup
(LeanSearch/Loogle or a local embedding index over Mathlib declarations). Justification:
on definitions, the dominant single-pass failure is unknown/incorrect Mathlib identifiers,
which no amount of error-feedback iteration fixes. Models: `gpt-oss-120b`, `claude-sonnet-5`.

B4 is the ablation that tells you how much of the agentic gain is just "it had the right
context," and it is much cheaper to run than any agent.

## Agentic tier

### A1 — Claude Code + MathAtlas MCP + lean-lsp MCP
The headline upper bound, and the config you can actually ship today. Tools: dependency/context
MCP, Lean LSP (goal state, diagnostics, `lean_run_code`), Mathlib search. Budget cap per item
(e.g. 40 tool calls / 15 min wall clock) — report the cap, since uncapped agents make the
number meaningless.

Models: `claude-sonnet-5` on all ~700; `claude-opus-5` on the 150-item stratified subset only.

### A2 — Minimal ReAct scaffold, same tools, model-swappable
~200 lines of your own harness: tool loop over the identical MCP tool set, same budget cap.
This exists so A1's result isn't confounded with Claude Code's proprietary scaffolding — with
A2 you can say "tool access buys X, harness buys Y." Run `gpt-5.2`, `claude-sonnet-5`, and
`gpt-oss-120b` so you also get an open-weights agentic point.

### A3 — OpenHands (one config)
Popular OSS agent; reviewers expect it and it is the comparability anchor. One model
(`gpt-5.2` or `claude-sonnet-5`), same tools and cap. Don't spend more than one row on it —
generic software agents underperform on formalization and the finding is not interesting twice.

### A4 (optional, availability-gated) — a formalization-specific agent
Merlean / Clawristotle / Aristotle-class. Add *only if* you can run it unmodified against
MA-Hard within a week; otherwise cite it and move on. These are the systems most likely to top
the table, but access and reproducibility are the risk.

## Recommended run set

| # | Config | Models | Scope |
|---|---|---|---|
| B1 ✅ | compile-repair K=5 | gpt-oss-120b, Qwen3-30B-A3B, gpt-5-mini, sonnet-5 | full 700 |
| B2 | specialist in loop | Goedel-V2-32B, Kimina-7B (+ hybrid repair) | full 700 |
| B3 | best-of-8 + judge | gpt-oss-120b, sonnet-5 | full 700 |
| B4 | retrieval + repair | gpt-oss-120b, sonnet-5 | full 700 |
| A1 ✅ | Claude Code + MCP | sonnet-5 (full), opus-5 (subset) | 700 / 150 |
| A2 | ReAct scaffold | gpt-5.2, sonnet-5, gpt-oss-120b | 150 subset |
| A3 | OpenHands | gpt-5.2 or sonnet-5 | 150 subset |

That's 15 runs, ~4 of which are frontier-API-cost. Run everything expensive on a fixed,
stratified 150-item subset drawn once and reused across all rows, so subset numbers are
comparable to each other and to the full-set rows restricted to those 150.

✅ = runner implemented; see `benchmarks/README.md` for prerequisites and invocation.
`benchmarks/iterative/run_iterative.py` (B1) also covers B2 (point `--model` at a
specialist) and, with `--max-rounds 1`, reproduces the single-pass control under
identical scoring.

## Model notes

- **Local (2×H200, 282GB)**: `gpt-oss-120b` is the right default workhorse — MoE, fast, already
  in your outputs so it links directly to single-pass. `Qwen3-30B-A3B-Thinking-2507` is the
  efficiency point. `Qwen3-32B` dense if you want a non-MoE control. Goedel-V2-32B and
  Kimina-7B for B2.
- **API**: `gpt-5.2` and `gpt-5-mini` (you already have `gpt-5.2` single-pass numbers, so the
  delta is free); `claude-sonnet-5` as the default agentic model, `claude-opus-5` for the
  ceiling. Skip Haiku — the loop cost isn't the bottleneck, quality is.

## Reporting

Per row: `compile_rate`, `aligned_rate` (CriticLean judge), and **joint verified∧aligned** —
report the joint as the headline, since iterative methods trivially inflate compile rate by
emitting degenerate-but-compiling statements (`theorem foo : True := trivial`). Also log total
tokens, wall clock, tool calls, and $ per item; a +6pt gain at 30x cost should be visible in
the table, not buried.

## Two things to guard

1. **Leakage.** The dependency MCP must not expose the gold formal statement of the item under
   test (or of a near-duplicate). Audit the tool responses on a sample before the real runs —
   this is the failure mode that silently invalidates the whole agentic tier.
2. **Degenerate compiling output.** Add a cheap syntactic guard (statement must reference at
   least one non-trivial hypothesis / the expected symbols) alongside the judge, and report how
   often it fires. Iterative and agentic systems both learn to satisfy the compiler.

## Out of scope for now

ProofFlow (in-repo) targets proof formalization with a proof dependency DAG; MA-Hard as
specified is statement-level. Revisit if you add a proof subset.
