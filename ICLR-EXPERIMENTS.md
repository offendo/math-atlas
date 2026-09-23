# ICLR revision experiments (MA-Hard)

What exists, what was added for the ICLR revision, how to run it, and what is still
missing. The motivation for each experiment is in `paper/CLAUDE-REVIEW.md` (§4.4);
reviewer IDs (R1–R3) refer to `paper/reviews/`.

**Results:** `outputs/iclr/analysis/report.md` (auto-generated) and the *Results*
section at the bottom of this file, which is filled in as runs finish.

---

## 0. Quick start

```bash
# one-time: verification stack and agent Lake projects (Lean/Mathlib v4.28.0 = blv's version)
N_WORKERS=24 docker compose -f ~/src/blv/compose.yaml --project-directory ~/src/blv up -d
scripts/setup_agent_projects.sh ~/src/ma-hard-iclr dep none opt

# everything (resumable: finished outputs are skipped on re-run)
ASSUME_YES=1 nohup scripts/run_iclr_experiments.sh > logs/iclr-driver.log 2>&1 &

# analyses (CPU only), once the runs finish
scripts/run_iclr_analysis.sh            # -> outputs/iclr/analysis/report.md etc.
```

Smoke test (3 items per run, throwaway projects/outputs):
```bash
BASE=~/src/ma-hard-iclr/base scripts/setup_agent_projects.sh ~/src/ma-hard-iclr-smoke dep none opt
N_EXAMPLES=3 JV_N=6 OUT_DIR=/tmp/iclr-smoke AGENT_ROOT=~/src/ma-hard-iclr-smoke ASSUME_YES=1 \
    scripts/run_iclr_experiments.sh
```

Environment notes (all handled in code, listed so they don't bite again):
- `OPENAI_BASE_URL` is set to the **empty string** on this machine. The OpenAI SDK then
  posts to `""` and every call fails with a bare "Connection error". `common.default_openai_url()`
  and `OpenAIGenerator` now treat empty as unset.
- `~/src/MathProjectTemplate` (the Sep-6 A1 project) no longer exists, and
  `mathatlas-formalization` is on Lean v4.30.0-rc1 while blv verifies against **v4.28.0**. The new
  agent projects are pinned to v4.28.0 so what the agent checks equals what blv scores.
- There is no `ANTHROPIC_API_KEY`. Sonnet single-pass runs go through the logged-in `claude`
  CLI with every tool disabled (`generators.ClaudeCLIGenerator`, `--backend claude-cli`). The
  CLI exposes no temperature or seed, so Sonnet runs are single samples at default sampling.
  `sonnet` resolves to `claude-sonnet-5` (recorded per run in `models_seen`).

---

## 1. Experiments that already existed (before this revision)

| ID | What | Where | MA-Hard joint | Notes |
|---|---|---|---|---|
| Paper T2/T4 | single-pass fine-tuned + prompted systems, full set | `outputs/*.json`, `criticlean/*.json` | gpt-oss-120b 2.6% (T4) | only prompted gpt-oss was run on MA-Hard |
| Paper T1 | faithfulness judges on MA-Align / ConsistencyCheck / CriticLeanBench | `outputs/*-ma-*.json`, `scripts/run_alignment_benchmark.py` | – | **CriticLean-32B runs are 100% parse failures** (see §4) |
| B1 | compile-repair K=5 (gpt-oss-120b, gpt-5-mini) + K=1 controls | `outputs/iterative/*` | 4.4 / 11.2 (K=1: 2.7 / 5.6) | Sep 6, `scripts/run_ma_hard_baselines.sh` |
| A1 | Claude Code (Sonnet) + lean-lsp + MathAtlas MCP | `outputs/agentic/claude-code-sonnet.ma-hard.json` | 29.2 | $175.75, concurrency 24, $0.50/item cap; **the MathAtlas MCP was available, but only 13.9% of items used it (8.2% called `get_dependencies`)**; 0 leak hits (audited, §2 E2b) |

So a "dependency-aware" agent was *wired* but not *exercised*: the prompt only
suggested the tools "when the informal text is ambiguous". E2b fixes that.

---

## 2. New experiments (code added in this revision)

All MA-Hard runs use `offendo/math-atlas-official`, split `hard` (698 items: 622
statements, 76 definitions), and are scored the same way: blv (Lean v4.28.0) compile,
then the CriticLean-32B judge on compiling items. **Joint = compiles AND faithful**
is the headline. Outputs go to `outputs/iclr/<kind>/<run>.json` with `<run>.metrics.json`.

### E0 — Judge validation (fixes the broken Table 1)
- **Code:** `benchmarks/judge_validation.py` (runs through the production judging path,
  `common.judge_alignment`, with the same prompts and structured output as the baselines).
- **Runs:** CriticLean-32B, gpt-oss-120b, gpt-5.2 × {MA-Align defs, MA-Align stmts,
  ConsistencyCheck, CriticLeanBench} → `outputs/iclr/judge-validation/<judge>.metrics.json`.
- **Reports:** accuracy, **balanced accuracy, MCC, Cohen's κ, sensitivity, specificity**,
  majority baseline, parse/call errors. MA-Align statements are 25/75, so plain accuracy has a
  75% trivial floor.
- **Why:** R2-W3, R3-W4/Q2, R1. The paper's CriticLean-32B MA-Align numbers could not be
  reproduced from disk: the old script's parser didn't strip ```` ```json ```` fences.
- **Paper:** Table 1 is replaced (balanced metrics + majority baseline); sensitivity and
  specificity feed E3c.

### E1 — Complete the MA-Hard model ladder
- **E1a, `benchmarks/analysis/slice_full_runs.py`:** every paper system's *full-set* output
  sliced to MA-Hard, **re-verified with current blv**, then CriticLean-judged (ReForm, Goedel
  8B/32B, Kimina, Herald, ATLAS, gpt-oss variants). No new generation. It uses the processed
  `.verified`/`.aligned` files, because the raw generation files still hold unfilled placeholders
  (Kimina writes `theorem {thm_example} …`). It records both the stored and the re-verified
  compile verdicts, and writes `pairwise_identity.json` (byte-identical outputs across systems).
- **E1b/c:** Sonnet (`claude-cli`) and gpt-5.2 single-pass; Sonnet compile-repair K=5
  (`sonnet.k5.dep-none`).
- **Why:** R1-W4/Q3, R2-W1/W2/Q1/Q2. It separates model strength (single-pass) from feedback
  (K=5) from tools/graph (A1).
- **Paper:** new §4.x, "How far can stronger systems get on MA-Hard?": a model × scaffold table
  with cost.

### E2 — Dependency access (the key experiment)
- **E2a, single-pass dependency retrieval.** `benchmarks/dependency_context.py`, exposed as
  `run_iterative.py --dependency-context {none,informal,mathlib,both,random}`.
  - The prompt gets the item's *direct* prerequisites from the MathAtlas graph (the same NDJSON
    the MCP serves): their informal text, and/or the Mathlib declarations the dataset grounded
    them to.
  - `random` is a **matched control**: the same number of random definitions from the same
    textbook, rendered identically. Local context is known to hurt, so "more text" has to be
    ruled out.
  - The item's own Mathlib grounding is never shown. A dependency whose grounding equals the
    item's own is shown without it (4/698 items).
  - Coverage: all 698 items have ≥1 in-dataset dependency (median 5); 606 have ≥1
    Mathlib-grounded dependency.
  - Runs:
    - gpt-oss-120b: all 5 modes, single-pass; plus K=5 × {none, both}.
    - Sonnet: {none, both, random}.
    - gpt-5.2: {none, both}.
- **E2b, agent arms.** Same model (Sonnet), budget ($0.50/item), timeout (900 s) and
  concurrency (24). Each arm gets its own fresh v4.28.0 project, so no arm can see another's item
  files.

  | Arm | MathAtlas MCP | Prompt |
  |---|---|---|
  | `none` | ✗ (lean-lsp only) | `agent_task.txt` |
  | `opt` | ✓ | `agent_task.txt` (tools optional: the Sep-6 condition, re-run on v4.28.0) |
  | `dep` | ✓ | **`agent_task_dependency_aware.txt`**: must call `get_dependencies` + `get_context` first, ground each prerequisite in Mathlib or formalize it above the target, then formalize the item |

  - `benchmarks/analysis/agent_tool_usage.py` audits every transcript: how often each arm
    *actually* used MathAtlas tools, and **leak hits** (any tool call touching the raw
    dataset/output files).
- **Why:** R3-W5/Q1 (main question), R2-W4 (novelty). It shows the graph is useful, not just
  correlated with failure.
- **Paper:** the showcase figure is correctness by condition with CIs, in two panels:
  single-pass none/random/informal/mathlib/both and agent none/opt/dep.

### E3 — Judge validity
- **E3a, human verification of "correct".** `benchmarks/analysis/make_annotation_set.py` builds
  a blind sheet: judged-faithful and judged-unfaithful compiling outputs from the agent and
  single-pass arms, plus guidelines. `--score` computes judge precision/NPV per run, κ between
  annotators, and failure categories. **Needs two human annotators (≈4 h each).**
- **E3b, multi-judge re-scoring.** Every MA-Hard output (this campaign's and the Sep-6 runs) is
  re-judged by gpt-oss-120b and gpt-5.2 into `outputs/iclr/rejudge/<judge>/`. `report.py` gives
  joint % per judge, Kendall τ of the system rankings, and item-level κ.
- **E3c, judge-error-corrected joint.** Rogan–Gladen with the E0 MA-Align sensitivity and
  specificity per item type, with a bootstrap CI over items *and* validation counts
  (`report.py`).
- **Why:** R2-W3, R3-W4/Q2, R1-W3. Agents optimise against the compiler, so the 29% headline
  needs human-verified precision.

### E4 — Confounds and robustness (CPU only, full set)
- **E4b, `scripts/graph_depth.py`.** Depth recomputed correctly (SCC condensation plus
  longest path; exact mass), compared with the paper's depths. It is then re-computed under
  extraction noise (9% of edges; noise models `drop` / `rewire-backward` / `rewire`, 100
  replicates each). Reports the Spearman ρ of depth and the **Jaccard overlap of MA-Hard
  membership**. Saves perturbed depth vectors for E4a.
- **E4a, `benchmarks/analysis/confounds.py`.** On the paper's own full-set per-item correctness:
  - pooled logit;
  - logit with controls (mass, length, #refs, Mathlib, type);
  - **linear probability model with textbook fixed effects** and textbook-clustered SEs;
  - the FE depth coefficient under each perturbed depth;
  - Mathlib effect: pooled χ² vs **CMH stratified by textbook**;
  - binned depth plot with Wilson CIs, including within-textbook depth tertiles.
- **Why:** R1-W3/Q1/Q2, R3-Q3.

### E5 — Failure analysis
- **E5a, `benchmarks/analysis/error_taxonomy.py`.** First Lean error per non-compiling output,
  bucketed (unknown identifier, typeclass synthesis, type mismatch, syntax, …) for every run.
  Shows what feedback fixes and what dependency context fixes.
- **E5b.** Failure categories for compiled-but-unfaithful outputs come from the E3a sheet
  (`failure_category`).
- **Why:** R3-W3/Q3.

### E6 — Tuned-example ablation (prompt × example domain)
- gpt-oss-120b on MA-Hard with the paper's own single-pass prompt files. Two new cells were built
  by splicing the existing files:
  - `theorem_few_shot_base_prompt_tuned_examples.txt`
  - `theorem_few_shot_tuned_prompt_lw_examples.txt`

  Theorems: {base, tuned prompt} × {none, LeanWorkbook, graduate examples}.
  Definitions: {base, tuned} × {none, graduate}.
- **Why:** R3-W2.
- **Caveat found while building it:** the graduate ("tuned") *theorem* examples are themselves
  unfaithful. Example 1 (cyclic vector) and Example 2 (metric) formalize to tautologies
  (`P ↔ P`, hypotheses restating the conclusion). Example 3 is unrelated to its text. Example 4
  asserts `inner v v = 1`. They teach vacuous patterns, which may explain why statement
  faithfulness *drops* with them (paper: 43.9 → 35.9%). Three examples also appear verbatim in
  MathAtlas (Milne, Robbin, Knapp), none of them in MA-Hard.

---

## 3. Orchestration

`scripts/run_iclr_experiments.sh`: three concurrent lanes, each sequential inside, all resumable.

| Lane | Order | Approx. wall clock |
|---|---|---|
| GPU (vLLM docker, GPUs 1,2, TP=2) | gpt-oss-120b: E2a ×5, E2a×K5 ×2, E6 ×10 → CriticLean-32B: E0, judge everything (polls until the other lanes finish) → gpt-oss-120b: E0 + E3b re-judge | 2 h gen + judge wait + ~1 h |
| Claude (`claude` CLI) | Sonnet sp: none/both/random → A1-dep → A1-none → Sonnet K5 → A1-opt | ~5–6 h |
| API/CPU | E1a slicing (+ blv re-verify, ~4 min/system) → gpt-5.2 sp none/both → gpt-5.2 E0 | ~1.5 h |
| After all lanes | gpt-5.2 E3b re-judging | ~30 min |

Estimated spend: 3 agent arms ≈ 3 × $175 (list price; the Sep-6 arm cost $175.75), Sonnet
single-pass/K5 ≈ $15–30, gpt-5.2 generation + judging ≈ $20–40.

---

## 4. Problems found in existing artifacts (fix before reporting numbers)

1. **CriticLean-32B MA-Align/CriticLeanBench outputs are 100% JSON-parse failures.**
   `scripts/run_alignment_benchmark.py:try_parse` does not strip ```` ```json ```` fences, so
   every verdict defaulted to "misaligned". The paper's 75.0 on MA-Align statements equals the
   all-misaligned majority baseline exactly. E0 replaces these numbers.
2. **The paper's depth is order-dependent.** `scripts/graph-stats.py` caches partial results
   when it hits a cycle. Recomputed with SCC condensation: **64% of items change depth**
   (Spearman 0.93 with the old values; 252 items lie in cycles, the largest SCC has 182 items),
   and the corrected top-698 overlaps MA-Hard with Jaccard **0.40**.
3. **MA-Hard membership is fragile under extraction noise** (2-replicate pilot; the full 100
   replicates run in `run_iclr_analysis.sh`). Dropping 9% of edges gives MA-Hard Jaccard
   0.17–0.42; rewiring 9% backward gives 0.03–0.11. The depth *rank* is more stable
   (ρ ≈ 0.77–0.92). Recommendation: define MA-Hard by a robust rule (e.g. corrected depth
   *and* mass quantiles, or per-textbook caps) and report its stability.
4. **Kimina raw generations contain unfilled placeholders** (`theorem {thm_example} …`). Only
   the processed `.verified` file compiles (27.3% full set, matching the paper; 12.4% on
   MA-Hard). Any re-scoring must use processed files.
5. **The ATLAS output file** (`outputs/xiaoyangliu-sjtu.atlas_translator_q.statements.json`)
   starts with a record identical to ReForm's (same `<round>`-style reasoning and formal
   statement). `slice_full_runs.py` writes `pairwise_identity.json` to settle whether the
   paper's ATLAS row is actually ReForm output.
6. **The graduate few-shot theorem examples are unfaithful** (see E6).
7. **The Sep-6 A1 run used the MathAtlas MCP in only 14% of items** (see §1).

---

## 5. Remaining gaps (not covered by this campaign)

- **E3a needs humans.** The sheets are generated automatically, but two annotators (≈4 h each)
  must fill them before judge precision on agent outputs can be reported. Until then, the 29%
  (and any new agent number) is judge-only.
- **Seeds.** Every configuration is a single run. gpt-oss runs are deterministic at T=0
  (first round). Sonnet via the CLI cannot be seeded. A1 arms are one sample each (≈$175 per
  repeat). Report CIs (item-level Wilson) and state that system-level variance is unmeasured.
- **No non-Claude agent** (A2 ReAct scaffold, A3 OpenHands from `baseline-plans.md`). A1 is
  therefore confounded with Claude Code's harness; E2b only varies MathAtlas access *within* it.
- **Best-of-n sampling (B3)** is not run, so compile-repair gains are not budget-matched against
  plain resampling.
- **Opus / frontier ceiling** is not run (only Sonnet); left out to hold the budget.
- **MA-Align is still 100+100 items with no reported IAA.** E0 fixes the metric, not the
  sample size; the corrected-joint CIs are wide for this reason.
- **Dependency extraction for R3-Q4** (smaller extractor models) is not run.
- ~~Open-Split table~~ → now `benchmarks/analysis/open_split_table.py` (run by
  `run_iclr_analysis.sh`). Open = uuids in the public release (`offendo/math-atlas-official`
  test + hard; `open-books.txt` titles do not match `file_id`s). Pilot: Kimina full-set row
  reproduces the paper exactly (27.3/8.3/2.3), but the open subset is **~50% of items, not the
  70% the paper states**, and its row (24.3/8.9/2.2) differs from paper Table 5 (27.1/8.1/2.2).
  Which file backs each paper row is inferred from filenames — confirm the mapping.
- **Proof formalization** remains out of scope.
- **MA-Hard redefinition** (item 3 above) is analysed but not applied; changing the split is
  a paper decision.

---

## 6. Results

*(Filled in from `outputs/iclr/analysis/*` as runs finish.)*
