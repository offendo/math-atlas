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
- **E2b, agent arms.** Same model (Sonnet), budget (**$0.75/item**), timeout (900 s) and
  concurrency (24). Each arm gets its own fresh v4.28.0 project, so no arm can see another's item
  files. The budget is above Sep-6's $0.50 because the dependency protocol hit $0.50 on 2 of 3 smoke
  items. The within-campaign `opt` re-run, not the Sep-6 run, is therefore the comparator.

  | Arm | MathAtlas MCP | Prompt |
  |---|---|---|
  | `none` | ✗ (lean-lsp only) | `agent_task.txt` |
  | `opt` | ✓ | `agent_task.txt` (tools optional: the Sep-6 condition, re-run on v4.28.0) |
  | `dep` | ✓ | **`agent_task_dependency_aware.txt`**: must call `get_dependencies` + `get_context` first, ground each prerequisite in Mathlib or formalize it above the target, then formalize the item. **Hook-enforced** (`--require-dependencies`) |

  - With the prompt alone, 1 of 3 smoke agents skipped the protocol entirely. The `dep` arm
    therefore runs a PreToolUse hook (`benchmarks/agentic/hooks/require_dependencies.py`) that
    blocks Write/Edit of the item file until `get_dependencies` has been called. Calls made by a
    subagent also count, since agents sometimes delegate the lookup. An agent can still write
    through Bash or run out of budget first, so the audit below reports actual compliance.

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
   starts with a record that looks like ReForm's. After extraction, however, its code overlaps
   ReForm's on **0/622** MA-Hard items, so it is not a copy. It is an *unprocessed* file (no
   stored verdict): re-verified compile is 0.8% on MA-Hard, against 21.4% for the paper's
   full-set ATLAS row. The processed ATLAS file the paper used is not in the repo; find it
   before reporting ATLAS on MA-Hard.
5b. **Duplicate gpt-oss outputs.** `criticlean/openai.gpt-oss-120b.statements.json` and
   `criticlean/openai.gpt-oss-20b.statements.aligned.json` contain **byte-identical code on 622/622
   MA-Hard items** (`outputs/iclr/sliced/pairwise_identity.json`). The "20b" judged file was most
   likely produced by judging the 120b generations. Any paper row built from it is really gpt-oss-120b.
6. **The graduate few-shot theorem examples are unfaithful** (see E6).
7. **The Sep-6 A1 run used the MathAtlas MCP in only 14% of items** (see §1).
8. **Agent checkpoint collision (fixed).** `run_claude_code.py` derived its checkpoint with
   `with_suffix`, so `claude-code-sonnet.dep.json` / `.none.json` / `.opt.json` all shared
   `claude-code-sonnet.partial.jsonl`. In the smoke test, the second arm "resumed" from the
   first arm's checkpoint and silently produced nothing. It is fixed, and the orchestrator now
   treats a run that exits without an output file as failed.

---

## 4b. Current state (Sep 23, 14:15): generation only, judging deferred

At the user's request, **judging is deferred.** Every judge will be re-run after the judge
itself is improved. The campaign now produces raw generations only.
- **The CriticLean-32B container died** some time between 10:12 and 13:24 on Sep 23 (it ran
  with `--rm`, so there are no logs). From then on, every judge call failed with "Connection
  error".
  - The only file it affected is `claude-code-sonnet.none.json`: its bogus all-"misaligned"
    verdicts were stripped. The original is kept as `…none.json.bogus-judge.bak`, and the metrics
    are now compile-only.
  - The orchestrator (`run_iclr_experiments.sh`) was stopped, so no other file gets garbage
    verdicts. GPUs 1–2 are now in use by another job.
- **Judged numbers already in the tables** (everything judged before 10:12, including
  `claude-code-sonnet.dep`) used CriticLean-32B with the production prompts. They will be
  superseded by the new judge.
- **To judge everything with the new judge:** serve it, validate it with
  `benchmarks/judge_validation.py --tag <tag>`, then run
  `scripts/judge_all.sh <model> <url> <tag> [--judge-prompt-file … --judge-definition-prompt-file …]`.
  This writes to `outputs/iclr/judged/<tag>/`, never touches the raw files, skips finished
  files, and refuses to start if the endpoint is down. Then run `report.py` / `run_iclr_analysis.sh`
  on that directory. For multi-judge agreement (E3b), run `judge_all.sh` once per judge and pass
  the parent directory to `report.py --rejudge-dir`.

### Raw generation: complete (Sep 23, 15:28)
Every generation run is done, with no empty rows. The three agent arms (Sonnet, $0.75/item cap,
v4.28.0 projects):

| arm | compile | used any MathAtlas tool | called `get_dependencies` | mean turns | cost (list) | leak hits |
|---|---|---|---|---|---|---|
| `none` (lean-lsp only) | 95.8% | 0% | 0% | 7.0 | $101.3 | 0 |
| `opt` (tools optional) | 98.4% | 22.5% | 5.3% | 7.3 | $101.6 | 0 |
| `dep` (protocol, hook-enforced) | 97.3% | 100% | 100% | 9.8 | $133.1 | 2 (grep of the textbook source; benign) |

**Judge (user decision): Qwen3.8-27B for everything; no proprietary judge.** Settings:
`--judge-max-tokens 32000 --judge-concurrency 40 --judge-reasoning-effort medium`, production
prompts. `scripts/judge_all.sh qwen http://localhost:8000/v1 qwen38-27b …` writes to
`outputs/iclr/judged/qwen38-27b/`. Validation tag: `qwen38-27b-medium`. On MA-Align (from the
user's earlier `qwen38-27b` run):
- Definitions: bal. acc 0.87, κ 0.74. Significantly better than CriticLean (McNemar p<0.001) and
  on par with gpt-5.2.
- Statements: bal. acc 0.81, specificity 0.61. No better than CriticLean (0.59; p=0.66). It still
  over-accepts about 40% of misaligned statements, so report the judge-adjusted column.

### Final MA-Hard results, Qwen3.8-27B judge (Sep 23, 21:46)
Full table: `paper/experiments.tex` (`tab:iclr-mahard`, regenerated by
`benchmarks/analysis/make_latex_tables.py`) and `outputs/iclr/analysis/report.md`. Qwen validation
(`qwen38-27b-medium`, 0 parse errors):

| benchmark | acc | balanced | sensitivity | specificity | κ | CriticLean acc |
|---|---|---|---|---|---|---|
| MA-Align defs | 88.0 | 0.88 | 0.94 | 0.82 | 0.76 | 67.0 |
| MA-Align stmts | 66.0 | 0.77 | 1.00 | 0.55 | 0.38 | 68.0 |
| ConsistencyCheck | 87.1 | 0.84 | 0.94 | 0.73 | 0.70 | 76.5 |
| CriticLeanBench | 87.0 | 0.87 | 0.90 | 0.84 | 0.74 | 82.0 |

Headline rows (correct %, 95% CI; judge-adjusted in parentheses):

| run | Qwen | CriticLean (earlier) |
|---|---|---|
| gpt-oss-120b single-pass / K5 | 2.7 [1.7, 4.2] / 7.9 (1.3) | 2.0 / 4.2 |
| gpt-5.2 single-pass | 3.2 [2.1, 4.7] | 2.1 |
| Sonnet single-pass / K5 | 38.7 [35.1, 42.3] / 73.6 [70.2, 76.8] (53.9) | 12.6 / – |
| Claude Code, no MathAtlas | 74.6 [71.3, 77.7] (59.3) | – |
| Claude Code, tools optional | 76.2 [72.9, 79.2] (60.0) | – |
| Claude Code, dependency-aware | 77.9 [74.7, 80.9] (64.2) | 27.1 |
| paper fine-tuned systems (ReForm, Goedel, Kimina, Herald) | 0.3–1.9 | 0.2–1.9 |

Paired McNemar tests (Holm-adjusted over 18 contrasts):
- **Dependency access does not significantly help.**
  - Agents: `dep` vs `none` is +3.3 pp (p=0.08 raw, 1.0 Holm); `dep` vs `opt` is +1.7 pp (n.s.).
  - Single pass: every gpt-oss dependency arm is within ±1 pp of none (n.s.); gpt-5.2 +both is
    +1.3 pp (p=0.09).
  - For Sonnet, the **random-definition control beats both no context (+9.9 pp, p<0.001) and real
    dependencies (+8.7 pp, p=0.001)**. Extra same-textbook context helps Sonnet, but the *specific*
    prerequisites add nothing over random ones. Worth checking whether the effect is textbook
    notation/conventions, or the judge.
- **Compiler feedback is what matters:** gpt-oss K5 vs single-pass +5.2 pp (p<0.001), Sonnet K5
  vs single-pass +35.0 pp (p<0.001). Sonnet K5 (73.6) is indistinguishable from the Claude Code
  agent without MathAtlas (74.6; p=0.64), so the agent harness adds little over a plain
  compile-repair loop.
- The `opt` re-run beats the Sep-6 A1 run by +18.8 pp (p<0.001). Budget ($0.75 vs $0.50) and
  Mathlib version both changed, so agent numbers are sensitive to setup.
- **Absolute numbers are judge-dependent by ~3×** for strong systems: the dependency-aware
  agent scores 27.1% (CriticLean) vs 77.9% (Qwen). Qwen's statement specificity is 0.55, and the
  judge-adjusted value is 64%. Rankings are stable, levels are not. The human check (E3a) decides
  which level is right. Sheets: `outputs/iclr/annotation/` (174 rows, guidelines included).

### MA-Align labels re-verified (Sep 24)
All 200 MA-Align pairs were re-judged by hand, blind to the original label:
`benchmarks/labels/ma-align-relabel.tsv` (label, confidence, reasoning per item). Existing judge
outputs are re-scored against it, without re-running any judge:
`benchmarks/analysis/rescore_ma_align.py` → `outputs/iclr/judge-validation-relabel/`
(`summary.md`, plus `<tag>.metrics.json` in judge_validation's schema).
- **Defs: 25/100 labels change (50 → 71 aligned).** 23 are misaligned → aligned. The original
  labels follow the strict rubric in `prompts/definition_alignment.txt`: reusing a matching Mathlib
  definition is misaligned (empty set, `Complex.exp`, `Int.ModEq`), and so is any harmless
  generalization (k=0 allowed, `Ring` for `Field`, no `a ≠ 0` on gcd). Some original rationales are
  also wrong: (−1)•x gives negatives in a module over a field (item 9), and `(⊤ : Ideal B).jacobson`
  is ⊤, not the radical (item 89, aligned → misaligned).
- **Stmts: 9/100 change (25 → 32 aligned).** Item 16 (existence of a minimizer for an arbitrary I)
  becomes misaligned; eight exact transcriptions become aligned.
- Qwen-medium: defs acc 88 → 75 (majority 71), sens 0.94 → 0.72, spec 0.82 → 0.83; stmts acc
  66 → 71, sens 1.00 → 0.97, spec 0.55 → 0.59. Most of the defs drop comes from the judge prompt
  itself, which instructs the strict rubric. Under the new labels, the definition prompt, not
  only the model, sets the judge's sensitivity.

## 5. Remaining gaps (not covered by this campaign)

- **OpenAI credits ran out at 02:16 on Sep 23** (`insufficient_quota`).
  - The gpt-5.2 *generation* runs had already finished (698/698 non-empty) and are valid.
  - gpt-5.2 judge validation is valid for MA-Align only. Its ConsistencyCheck/CriticLeanBench rows
    are call errors and are marked invalid in `report.md`.
  - The final gpt-5.2 re-judging step (E3b third judge) preflights the quota and skips itself.
  - **To finish it:** add credits, delete `outputs/iclr/judge-validation/gpt-5.2.*`, and re-run
    `scripts/run_iclr_experiments.sh` (finished work is skipped).

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

### E0 — judge validation, production judging path (final for CriticLean-32B and for gpt-5.2 on MA-Align)
| judge | benchmark | paper acc | **acc** | balanced acc | sensitivity | specificity | κ | majority | parse err |
|---|---|---|---|---|---|---|---|---|---|
| CriticLean-32B | MA-Align defs (50/50) | 80.0 | **67.0** | 0.670 | 0.660 | 0.680 | 0.34 | 0.50 | 2 |
| CriticLean-32B | MA-Align stmts (25/75) | 75.0 | **68.0** | 0.773 | 0.960 | **0.587** | 0.39 | 0.75 | 1 |
| CriticLean-32B | ConsistencyCheck | 82.6 | 76.5 | 0.774 | 0.746 | 0.803 | 0.51 | 0.66 | 3 |
| CriticLean-32B | CriticLeanBench | 86.4 | 82.0 | 0.820 | 0.764 | 0.876 | 0.64 | 0.50 | 18 |
| gpt-5.2 | MA-Align defs | 86.0 | **79.0** | 0.790 | 0.740 | 0.840 | 0.58 | 0.50 | 0 |
| gpt-5.2 | MA-Align stmts | 80.0 | **85.0** | 0.820 | 0.760 | 0.880 | 0.62 | 0.75 | 0 |
| gpt-5.2 | ConsistencyCheck / CriticLeanBench | – | invalid | – | – | – | – | – | out of API credits |

(Parse errors are counted as "misaligned", the production behaviour.)

- **All four CriticLean-32B numbers in the paper are overstated.** MA-Align definitions drop from
  80 to 67; statements drop to 68, *below* the 75% always-misaligned baseline.
- **The production judge over-accepts statements:** sensitivity 0.96, specificity 0.59. Among
  compiling statements, "faithful" is inflated, so statement correctness across the paper is an
  overestimate. E3c's Rogan–Gladen column quantifies how much.
- gpt-5.2 beats CriticLean on MA-Align (79/85 vs 67/68). The paper's qualitative conclusion ("judges
  degrade on graduate math; a strong proprietary judge does better") survives, with different numbers.
- κ for CriticLean on MA-Align is 0.34–0.39 ("fair"). Calling it "strongly correlated with human
  judgment" (rebuttal) is not supportable.

### E1a — paper systems on MA-Hard, re-verified with blv v4.28.0 (compile only; judging pending)
Re-verified compile % [stored verdict from the original pipeline], n = 622 statements or 76 definitions:
ReForm 5.0 [5.0] · Goedel-8B 6.6 [6.6] · Goedel-32B 8.2 [13.7] · Kimina 12.2 [12.4] · Herald 3.4 [3.5] ·
ATLAS 0.8 [unprocessed file, see §4] · gpt-oss-120b stmts: default 15.1, zero-shot 5.6, tuned-prompt 7.1 ·
gpt-oss-120b defs: default 11.8, zero-shot 0.0, tuned-prompt 7.9, tuned-examples 2.6, context-300 6.6 ·
gpt-oss-20b defs default 9.2. Stored and re-verified verdicts agree except for Goedel-32B, which
loses 5.5 pp under Lean v4.28 (Mathlib renames). **The paper's best statement system on the full
set (ReForm) compiles on only 5% of MA-Hard**, below Kimina (12%) and gpt-oss default (15%).

### E4b — depth correctness and robustness (final; `outputs/iclr/depth/`)
- Corrected (SCC) vs paper depth: Spearman ρ = 0.93, **64% of items change depth**, 252 items in
  cycles (largest SCC: 182). The top-698 by corrected depth overlaps MA-Hard with Jaccard **0.40**.
- 100 replicates per noise model, 9% of edges perturbed:

  | noise | depth rank ρ vs clean (mean; 2.5%) | MA-Hard Jaccard (mean; 2.5%) |
  |---|---|---|
  | drop | 0.90; 0.86 | 0.32; 0.08 |
  | rewire-backward | 0.81; 0.74 | 0.10; 0.01 |
  | rewire (any, same book) | 0.71; 0.66 | 0.06; 0.01 |

  **Depth ranks are fairly robust; MA-Hard membership is not.** MA-Hard should be redefined with
  a noise-robust rule, or reported with this caveat.

### E4a — confounds on the paper's full-set outputs (final; `outputs/iclr/analysis/confounds/`)
Linear probability model of correct, per SD of depth, in pp (textbook FE = textbook fixed effects
plus controls for mass, length, #refs, Mathlib, type; SEs clustered by textbook):

| system (split, n) | correct % | pooled, corrected depth | textbook FE, corrected depth [95% CI] | textbook FE, paper depth | textbook FE under noise (mean [2.5, 97.5]) |
|---|---|---|---|---|---|
| ReForm 8B (stmts, 45,259) | 9.2 | −3.3 | **−4.6** [−6.3, −3.0] | −2.1 (p=3e-6) | −0.5 [−1.3, +0.2] |
| Goedel 8B (stmts, 45,260) | 7.1 | −3.2 | **−3.3** [−4.7, −1.9] | −0.7 (n.s.) | −0.8 [−1.5, −0.3] |
| gpt-oss-120b "tuned exs" file (defs, 12,913) | 15.3 | −6.2 | **−6.7** [−9.7, −3.8] | −2.0 (p=.007) | −1.3 [−2.5, −0.1] |
| gpt-oss-120b default file (defs, 12,913) | 16.7 | −6.6 | **−6.1** [−8.9, −3.3] | −1.5 (n.s.) | −1.2 [−3.1, −0.0] |

- **The depth effect survives textbook fixed effects** when depth is computed correctly. With the
  paper's depth it is weaker (not significant for 2 of 4 systems). On noise-perturbed graphs it
  shrinks by roughly 4×. So the claim "depth predicts failure beyond subfield" holds, with a
  smaller and more noise-sensitive effect than Fig. 4 suggests.
- **Mathlib (definitions): the effect survives stratification.** In-Mathlib 21–24% vs 12–13%
  correct. CMH pooled OR = 1.51 / 1.56 (p ≈ 0) across 141 textbooks; FE model +4–5 pp. Present
  "leakage" as one hypothesis. The confound check R1 asked for passes.
- **Bookkeeping:** ReForm's full-set correctness recomputes to **9.2%** (the paper says 9.8%). The
  gpt-oss definitions file named `definitions.aligned.json` gives **16.7%** (the paper's "+tuned
  exs." headline), while the file named `…few_shot_tuned_examples` gives 15.3% (the paper's
  "+tuned prompt" value). File names and paper rows are mismatched somewhere; fix the table
  provenance.
