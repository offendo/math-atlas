# MathAtlas — Review Analysis & Revision Plan

*Sources: `paper/neurips_2026.tex`, `paper/reviews/*`, `paper/BASELINE-RESULTS.md`, and the result/data files in `outputs/`, `math-atlas.json`, `benchmarks/`.*

Scores:

| Reviewer | Score | Confidence |
|---|---|---|
| R1 | 4 (borderline accept) | 3 |
| R2 | 3 (borderline reject) | 2 |
| R3 | 3 (borderline reject) | 4 |

The most confident reviewer is negative, and all three share the same core objection.

---

## 0. TL;DR

1. **All three reviewers value the dataset.** The rejections are about the *evaluation*:
   - only single-pass runs, mostly of small open models;
   - a noisy judge;
   - no CIs;
   - no method ever uses the dependency graph.
2. **The rebuttals deferred almost everything.** They promised "future work" or "camera-ready" and gave almost no new numbers. Two answers contradict each other: R1 was told frontier models *will* be run, while R2 was told the authors are "unable to test large models". One claim is contradicted by the repo: the R3 rebuttal says the tuned examples are not from MathAtlas, but three of them appear verbatim in `math-atlas.json` (§3, R3).
3. **Your existing MA-Hard results already answer the biggest objection (A).**

   | System on MA-Hard | Joint correct |
   |---|---|
   | gpt-oss-120b, single-pass | 2.7% |
   | gpt-5-mini, compile-repair K=5 | 11.2% |
   | Claude Code (Sonnet) + MathAtlas MCP + Lean LSP | **29.2%** |

   The Claude Code run took about 68 minutes of real time and cost $176, a median of $0.22 per item. These results answer R1-Q3, R2-Q1 and R2-Q2. They also mean the headline "best model achieves only 2.6%" has to change.
4. **Four validity problems need resolving before anything else (§4.6).** None of the reviewers caught them.
   - The on-disk CriticLean-32B MA-Align runs are 100% JSON-parse failures.
   - MA-Align statements are 75/25 imbalanced, so "75.0% accuracy" is exactly the majority-class rate.
   - Judge errors are directional (over-accepting).
   - ReForm's 9.8% headline doesn't match its own row (19.0% × 48.4% = 9.2%).
5. **2-day plan, MA-Hard only (§4.4).** The core is E0–E3; E4–E6 are cheaper or optional.
   - **E0:** CIs, plus a fix for the judge.
   - **E1:** complete the model ladder.
   - **E2:** a **dependency-access ablation**, the experiment that turns the graph from a statistic into a demonstrated resource.
   - **E3:** judge validity: human verification of "correct" outputs, plus a sensitivity analysis.
   - **E4:** confound and robustness analysis on existing outputs.
   - **E5:** a failure taxonomy.
   - **E6:** optional prompt 2×2.

---

## 1. Overall: what the reviewers' main problems were

| # | Theme | Who | Core complaint |
|---|---|---|---|
| **A** | **The difficulty claim is not established** | R1, R2, (R3) | Only single-pass prompting of small open models, up to 32B plus gpt-oss-120b. No frontier models, no feedback loops, no agents. The low numbers might reflect weak baselines rather than a hard benchmark. |
| **B** | **Measurement validity** | R1, R2, R3 | Correctness depends on CriticLean, which your own MA-Align puts at only 75–80% accuracy. There is no sensitivity analysis, no CIs or significance tests, and every result is a single run. At 2.6% correctness, comparisons between systems can't be interpreted. |
| **C** | **Pipeline noise and confounded analyses** | R1, R3 | The dataset is fully LLM-extracted and was validated on small samples. Extraction error propagates into depth and MA-Hard membership. The depth and Mathlib effects are confounded with subfield, textbook and difficulty, and the "training-data leakage" reading is unsupported. |
| **D** | **The key innovation (the dependency graph) is unused** | R3 (main), R2 (novelty) | The graph only appears as a correlate of failure (Fig. 4). No baseline uses it, so the benchmark's distinguishing feature is never shown to be *useful*. R2 sees the recipe as extract → prompt → judge. |
| **E** | **Framing, completeness and polish** | R1, R3 | MA-Align is undersold. Limitations covers scope but not measurement. The Open Split and copyright bias get no discussion. There is no failure-mode analysis. The tuned-examples ablation is ambiguous. There are typos and inconsistent numbers. |

**Bottom line:** A and D are what move scores. B and C give a confident reviewer (R3) grounds to reject. E is cheap to fix.

---

## 2. Specific problems, reviewer suggestions, and my notes

Format: **Problem** → *Reviewer suggestion* → my note / suggestion.

### Reviewer 1 (score 4)

**R1.1 Claims and assumptions are not made explicit; Limitations covers scope only.**
→ *Split Limitations into scope vs. measurement limitations: pipeline noise, faithfulness-metric error, coverage bias, the Mathlib-leakage interpretation, dataset reproducibility.*
→ Agree. Also add a short "What MathAtlas does / does not measure" paragraph to §4 listing the assumptions:
- extraction fidelity;
- CriticLean as a proxy for faithfulness;
- statement-only evaluation, with no proofs.

**R1.2 MA-Align is undersold.**
→ *Give it co-headline status next to the benchmark.*
→ Agree, but a co-headline needs more rigour than MA-Align has now (§4.6):
- an annotation protocol and inter-annotator agreement (IAA);
- balanced metrics (balanced accuracy, MCC, κ) and confusion matrices;
- ideally, judge accuracy vs. depth *within* MA-Align, which directly tests "judges degrade as the math gets harder".

**R1.3 Pipeline noise propagates into depth, MA-Hard and Fig. 4; the validation samples are small.**
→ *Bootstrap the depth distributions under plausible extraction error and show the depth–correctness trend holds.*
→ Agree. It needs no GPU (E4b). Also report how stable MA-Hard membership is (Jaccard overlap) under perturbation.

**R1.4 Statistical rigour is thin; single runs make small gaps uninterpretable.**
→ *No explicit suggestion.*
→ Three steps:
- Wilson 95% CIs on every table cell.
- Paired tests (McNemar or paired bootstrap), since all systems run on the same items.
- 2–3 seeds for the top MA-Hard configurations.

**R1.5 The Mathlib "leakage" interpretation is confounded with difficulty and subfield.**
→ *Stratify the χ² by subfield or entity length. If the gap persists the claim strengthens; if it shrinks, soften it.*
→ Agree. Use Cochran–Mantel–Haenszel stratified by textbook, or logistic regression with textbook fixed effects (E4a). Also report linker precision: the Mathlib grounding is never evaluated, yet the claim depends on it.

**R1.6 No frontier proprietary models on the main task.**
→ *Run about 200 entities through one frontier model.*
→ **Largely done already** on MA-Hard (Claude Code Sonnet, gpt-5-mini). Add frontier single-pass and compile-repair runs so model strength can be separated from agentic scaffolding (E1).

**R1.7 The dataset is not bit-reproducible because of LLM sampling.**
→ *State it as a limitation.*
→ State it, and release the exact pipeline outputs, prompts, model revisions and seeds, so the *released artifact* is the reference.

**R1.8 Typos.** "~51k" vs. "52,052"; footnote numbering; "andwincludes"; "Dedeking" (should be Dedekind).
→ Fix. The full list is in §4.5.

### Reviewer 2 (score 3)

**R2.1 Only one-pass prompting: no execution feedback, no agents (OpenHands, Claude Code).**
→ *(Q1) Measure how much difficulty remains under an agentic setup with feedback.*
→ **You now have this.** B1 compile-repair takes gpt-oss-120b from 2.7% to 4.4% and gpt-5-mini from 5.6% to 11.2%; A1 Claude Code reaches 29.2%. Present these as a ladder in the main text (E1).

**R2.2 Only small open formalizers; no flagship models.**
→ *(Q2) Measure flagship performance and whether it changes the conclusions.*
→ Same as R1.6. Answer honestly: frontier models with tools lift MA-Hard from ~3% to ~29%. The benchmark is far from saturated, but it is no longer "~3% for everyone".

**R2.3 The LLM judge gives the absolute numbers a large error band.**
→ *No explicit suggestion.*
→ E3:
- (a) rescore with 2–3 judges and show the rankings are stable;
- (b) report correctness corrected for judge error (Rogan–Gladen, using MA-Align sensitivity and specificity), with CIs;
- (c) human-verify a sample of "correct" outputs.

**R2.4 Limited methodological novelty (extract → prompt → judge).**
→ *No explicit suggestion.*
→ Reframe the contribution as *new evaluation affordances*:
1. definitions as first-class targets;
2. a dependency graph that enables controlled ablations, such as dependency access and depth-stratified evaluation;
3. the MA-Align finding that judges transfer poorly to graduate-level math.

The dependency ablation (E2) is what makes (2) demonstrated rather than claimed.

### Reviewer 3 (score 3, confidence 4)

**R3.1 The construction pipeline depends on a "proprietary" model; smaller models are untested.**
→ *(Q4) Try open alternatives such as Qwen3 and report the minimum capability needed.*
→ The premise is wrong: gpt-oss-120b is open-weight, and your rebuttal said so correctly. The underlying question is fair but low priority. Optionally, rerun entity extraction on the pages behind the 250 annotated entities with Qwen3-30B-A3B and compare validity.

**R3.2 "Tuned examples" conflates prompt quality with in-domain examples.**
→ *Run a cleaner ablation that separates the two.*
→ The ablation is cumulative (few-shot → +tuned prompt → +tuned examples), so the two variables are never isolated. Add the missing cell, **base prompt + graduate examples**, as a 2×2 (E6).

Also fix the provenance claim. The examples *are* partly MathAtlas text: Milne "semilocal" (definition prompts), plus Robbin and Knapp theorems in the theorem prompt. None of them are in MA-Hard. Either swap in held-out examples and re-run, or disclose the overlap and exclude those items from evaluation.

**R3.3 No failure-mode analysis.**
→ *(Q3) Break failures down into compile vs. semantic, by field, and by object type (structures vs. typeclasses).*
→ Agree, and it's cheap (E5): an automatic taxonomy of Lean errors, plus about 60 hand-labelled outputs that compile but are unfaithful, plus per-textbook correctness.

**R3.4 Tension in the faithfulness metric: the best judge (gpt-5.2) is impractical, and the judge actually used (CriticLean) is noisy.**
→ *(Q2) A sensitivity analysis or confidence-adjusted correctness.*
→ Same as R2.3 (E3). It matters more than the reviewers realise, because the judges' error is **directional**: they over-accept, which inflates correctness (§4.6).

**R3.5 No dependency-aware baseline, even though the graph is the key innovation.**
→ *(Q1) A simple "retrieve Mathlib definitions first, then formalize" pipeline.*
→ **The highest-value experiment** (E2). It needs no ARIA-style agent. The graph edges (`object_links`) and Mathlib groundings (`mathlib_links`) are already in `math-atlas.json`, and the A1 harness can already turn the MathAtlas MCP on or off with `--mathatlas-project`.

**R3.6 Open-Split results are buried in the appendix.**
→ *Discuss consistency with the full set in the main text.*
→ Add 2–3 sentences and the full-vs-open deltas with CIs. **Regenerate Table 5 first**: several of its rows are identical to full-set rows (§4.6).

**R3.7 (Limitations) The copyright-induced selection bias is not characterised.**
→ *Describe what is lost.*
→ Add an appendix table comparing the full set and the Open Split by field, entity type and depth.

**R3.8 (Limitations) No proof formalization; Lean only.**
→ *Nothing beyond noting it.*
→ One sentence each in Limitations. On the worry that Lean-only evaluation "measures Lean-ecosystem familiarity": the grounded vs. ungrounded Mathlib analysis is relevant evidence, so say so.

**R3.9 (Limitations) Metric reliability at low correctness; no CIs or significance tests.**
→ *Report them.*
→ Same as R1.4. With n≈700 and p≈2.6%, the 95% CI is roughly ±1.2pp, so gpt-oss-20b (2.1%) and gpt-oss-120b (2.6%) **cannot be distinguished**. Say so.

---

## 3. Responses to the rebuttals (written as each reviewer)

### R1's response to your rebuttal

> Thank you. I accept the commitments to expand the Limitations section, soften the leakage claim, and elevate MA-Align. However, most of my substantive questions were answered with promises rather than results.
>
> - **Noise propagation (W3/Q1):** The CI on extraction accuracy (90.4 ± 3.6) answers a different question. I asked whether the *depth–correctness trend and MA-Hard membership* survive extraction error. That analysis needs no GPU, yet it was deferred. The proposed post-hoc filter worries me as well: its 92% F1 is measured on the same 250 annotations used to develop it, and changing the dataset after review makes every reported number stale. Also, your answer to Q1 ends mid-sentence ("However, we note that in human analysis of the data.").
> - **Mathlib confound (Q2):** A stratified χ² takes a few lines of code on existing outputs. I expected to see the result in the rebuttal.
> - **Frontier models (W4/Q3):** "200 items formalized from two algebra textbooks" has no denominator, no correctness criterion and no comparison condition, so I cannot interpret it. It does not tell me whether MathAtlas is hard for frontier systems. Running frontier models on MA-Hard is exactly the right plan, and I'd like to see the numbers.
>
> **Verdict:** Framing and limitations are addressed; the core empirical concerns remain open. I keep my score at 4.

**Critical analysis:** R1 was your ally, and the rebuttal gave them nothing new to argue with in the discussion. The ±3.6 CI is computed correctly (n=250, p=.904 gives ±3.65), but it answers a question R1 didn't ask. The filtering proposal adds risk: if you apply it, every table has to be regenerated.

### R2's response to your rebuttal

> - **W1 (no feedback or agents):** The cost argument doesn't convince me. A compile-repair loop needs a few thousand tokens per item, and MA-Hard has only 700 items. You also describe a Claude-based system that cost $600 over five days, so the budget exists. And the anecdote that 200 theorems were formalized cuts *against* the paper: if an agent can formalize 200 items, the "extremely challenging" headline has to be reconciled with that, not deferred.
> - **W2 (no flagship models):** "Unable to test large models due to computational constraints" is hard to reconcile with three facts: frontier models are served through APIs, not self-hosted; the paper already uses gpt-5.2 on MA-Align; and this same rebuttal describes an Opus-based system. The reply to R1 even says frontier models *will* be run on MA-Hard. The two answers are inconsistent.
> - **W3 (judge noise):** "Strongly correlated with human judgment" is not backed by any correlation or agreement statistic, and 75–80% accuracy on a 100-item set is not strong. I didn't ask you to solve faithfulness evaluation, only to quantify how much judge error moves your numbers.
> - **W4 (novelty):** A reasonable answer for a benchmark paper, and I accept that simple extraction is a feature. It would be more convincing if one experiment showed the graph *enabling* something.
>
> **Verdict:** W4 partially addressed; W1–W3 not addressed. I keep my score at 3.

**Critical analysis:** This was the weakest rebuttal. The "computational constraints" line does the most damage: an AC reading all three rebuttals will notice that it contradicts the reply to R1. `BASELINE-RESULTS.md` now contains exactly what R2 asked for.

### R3's response to your rebuttal

> - **W1 (proprietary pipeline):** Thank you for the correction; gpt-oss-120b is open-weight and self-hosted. Resolved. My question about smaller extractors remains, but it is minor.
> - **W2 (tuned examples):** Thanks for clarifying the examples' provenance; please fix the text. The ablation is still cumulative, so the domain effect is not isolated. The data also show that the in-domain gain is mostly on definitions. On statements, gpt-oss-120b goes from 7.5% to 7.8% correct while faithfulness drops from 43.9% to 35.9%. Partially resolved.
> - **W3 (failure modes):** Not addressed. The response pivots to dependency depth and efficiency. A compile-error taxonomy and a small manual sample would have been cheap. Your answer to Q3 also concedes that depth correlates with field. That is exactly the confound R1 raised, and it weakens the "depth drives difficulty" reading unless it's controlled for.
> - **W4 (metric noise):** I agree gpt-5.2 shouldn't be the default judge. But I asked for a *sensitivity analysis of your own numbers*, and you call that out of scope (Q2). It is in scope: it decides whether 9.8% really means 9.8%.
> - **W5 (dependency-aware methods):** Not addressed. I didn't ask for ARIA or an agent. A single-pass "retrieve dependencies, then formalize" baseline needs neither, and your graph already does the retrieval. "Extremely slow and expensive" doesn't describe a single-pass variant.
> - **W6 (Open Split):** A commitment only. Fine.
> - My Limitations points (copyright bias, CIs) were not addressed.
>
> **Verdict:** One factual correction resolved and one wording issue clarified. My two main requests, a dependency-aware baseline and a metric-sensitivity analysis, were declined. I keep my score at 3.

**Critical analysis:**
- **R3 is your most confident and most detailed reviewer, and you declined both of their main requests (Q1, Q2).** These are the two cheapest high-value experiments you have.
- **The W2 claim is inaccurate as written.** The rebuttal says the examples "were not taken directly from MathAtlas, but rather from other graduate level mathematics work which we left out of MathAtlas". In fact, the tuned-example prompt files contain text that appears verbatim in `math-atlas.json`:

  | Textbook | Entity | Depth |
  |---|---|---|
  | Milne | *Definition 12.8* (semilocal) | 15 |
  | Robbin | metric theorem and example | 1–2 |
  | Knapp | representation theorem | 68 |

  None of them fall in MA-Hard (depth ≥ 72), so the numerical impact is negligible. But the camera-ready must not repeat the rebuttal's claim. Either replace these examples or disclose the overlap and exclude the items.

### On the overall rebuttal

- "We believe we have thoroughly addressed all concerns … we ask the reviewers to update their scores" isn't supported by the content, since nearly every request was deferred. Reviewers read that as overclaiming.
- The cost argument works against the benchmark's own design. MA-Hard (700 items) exists precisely so that expensive systems *can* be evaluated. **Turn it into a positive in the revision:** "MA-Hard is the budget-friendly frontier split. A full agentic run costs about $175 and takes about an hour."
- Replace "strongly correlate with human judgments" with actual numbers: κ, balanced accuracy, and CIs.

---

## 4. My own assessment

### 4.1 Is the research contribution worth publishing?

**Yes, provided the evaluation is repaired.** The artifact itself is undisputed:
- 52k graduate-level entities taken from textbooks "in the wild";
- definitions as first-class targets;
- a dependency graph with 178k edges;
- Mathlib groundings;
- a test set for faithfulness metrics.

Graduate-level autoformalization is where the field is heading, and no resource of this scale exists. What's missing is evidence that (a) the difficulty holds for strong systems and (b) the graph is *useful*, not merely correlated with failure.

Your existing agentic result is **good news, if you frame it correctly**. The story becomes:

> *Single-pass systems solve ~3% of MA-Hard. Compiler feedback helps modestly (≤11%). A frontier agent with access to MathAtlas's dependency graph and Lean tools reaches ~29% for about $0.25 per item. [E2: X pp of that comes from dependency access, which only MathAtlas provides.] Even so, about 70% of MA-Hard remains unsolved, and [E3] judge-corrected numbers are lower still.*

That paper is stronger than "everything scores 2.6%". It makes R2's objection your headline and answers R3's with the graph.

### 4.2 Strengths

- **Scale and realism:** graduate text taken as-is across 87 subfields, roughly 500× larger than FATE-scale graduate benchmarks.
- **Definitions are included and evaluated.** That is rare, and the Intro's argument for why definitions matter (¶4) is compelling.
- **The dependency graph and Mathlib grounding** enable analyses no other benchmark supports.
- **MA-Align** is a genuine evaluation-science finding: judge rankings on ConsistencyCheck and CriticLeanBench don't transfer.
- **Honest reporting of negative results:** local context hurts, and Kimina shows a large compile-vs-faithful gap.
- **Separate compile, faithful and correct metrics** are the right protocol.
- **A practical release:** the Open Split plus code to regenerate the full set.

### 4.3 Weaknesses (beyond what the reviewers said)

1. **The headline difficulty claim rests on weak systems** (addressed by E1).
2. **No method uses the graph** (E2).
3. **Measurement validity is worse than the reviewers realised** (§4.6): the judge over-accepts, MA-Align statements are imbalanced, and the on-disk CriticLean-32B runs are broken.
4. **The depth effect is probably confounded with textbook identity.**
   - Both KDEs in Fig. 4 are **bimodal**, with clusters at depth 0–20 and 55–75.
   - MA-Hard is exactly the non-proof items with depth ≥ 72. It covers 37 textbooks, and **161 of its 698 items (23%) come from one book (Vakil)**.
   - So "depth predicts failure" may largely mean "these books are hard". This needs a within-textbook analysis (E4a) and per-textbook macro-averages on MA-Hard.
   - The KS test is uninformative at n ≈ 30k, where everything comes out significant. Report effect sizes.
5. **The depth computation has a correctness issue.**
   - `scripts/graph-stats.py` runs a memoized DFS over `object_links`, and cycles are handled by giving the back-edge depth 0. With memoization, **the depth of nodes on or above a cycle depends on traversal order**.
   - The 16.3% of object references with no matched target become leaves, which *underestimates* depth.
   - The mass computation double-counts shared descendants in `branch_sum`.
   - Fix: condense strongly connected components, then take the longest path on the resulting DAG. Report this in the paper (E4b).
6. **The quality evaluation measures precision only.**
   - Entity recall is never measured; the `\jmf{add stuff about precision and recall}` todo is still in the source.
   - Relation recall is computed only over retrieved candidates, which gives an upper bound.
   - The Mathlib linker is never evaluated, although the 27.9 vs. 16.8 claim depends on it.
7. **MA-Hard is under-specified and under-evaluated.**
   - The selection rule (depth ≥ 72, excluding proofs) isn't stated. The paper says "14 subjects", but the data show 37 textbooks.
   - Only prompted gpt-oss models were evaluated on it; ReForm, the best statement system, was not. ReForm's full-set run covers the 622 MA-Hard statements, and only 5.0% of them compile.
   - Only 14 of the 698 items have a Mathlib link.
8. **Missing related work.**
   - RAutoformalizer, the dependency-retrieval autoformalizer from the paper you already cite for BEq (`liu2025rethinking`). A reviewer who knows it will ask why it wasn't a baseline.
   - miniCTX, on context-dependent formal math; directly relevant to your local-context result.
   - Recent agentic formalization systems, now that you report an agent.
   - *(Check each citation before adding it.)*

### 4.4 Additional experiments (2 days, MA-Hard only)

Budget facts from the repo:
- A1 (Claude Code, concurrency 24) ran MA-Hard in about 68 minutes of real time for $176; 58 items hit the $0.50 cap.
- B1 gpt-oss-120b with K=5 took about 37 minutes, and the single-pass control about 8.
- CriticLean judged all five runs in about 17 minutes.
- Per-item outputs exist for every MA-Hard run and for all full-set single-pass runs (sliceable by uuid).

That means **everything below fits in 2 days**. The limit is human annotation time, not compute.

> Ordering is by value per hour. The E0 judge fix gates everything. E4 is analysis-only and runs in parallel.

---

#### E0 — Fix the judge and add statistics (must-do, ~3 h, no new generation)
- **What:**
  1. Fix the CriticLean-32B output parser: strip ```` ```json ```` fences and `<think>` blocks. Then rerun it on MA-Align defs and stmts, ConsistencyCheck and CriticLeanBench (about 20 minutes).
  2. Report balanced accuracy, MCC, κ and confusion matrices, with the majority baseline shown.
  3. Add Wilson 95% CIs to every cell of Tables 2 and 4 and the MA-Hard ladder.
  4. Run McNemar tests for adjacent rungs of the ladder.
- **Why:** R1.4, R3.9, R2.3. It also resolves §4.6-1/2. If the fixed 32B judge differs from the paper's numbers, the tables must be rescored *before* anything else, because every "faithful" column depends on it.
- **Story:** Table 1 gains balanced metrics. Every table gains CIs. The checklist's error-bar answer becomes "Yes".

#### E1 — Complete the MA-Hard model ladder (~½ day of mostly unattended compute, ≤$60)
- **What:**
  - (a) **Slice existing full-set outputs** to MA-Hard uuids (ReForm, Goedel 8B/32B, Kimina, Herald, ATLAS, and the gpt-oss prompt variants), and re-judge them uniformly with `benchmarks/judge_results.py`. This needs no new generation. It fixes "best model 2.6%" being based on only two systems, and it resolves the question of which judge scored ReForm (§5).
  - (b) **Frontier single-pass:** `run_iterative.py --max-rounds 1` with Sonnet, and gpt-5.2 if the budget allows.
  - (c) **Frontier compile-repair:** the same with `--max-rounds 5`.
- **Result:** a model × scaffold grid.
  - Rows: gpt-oss-120b, gpt-5-mini, Sonnet.
  - Columns: single-pass, compile-repair, and agent with tools and dependencies.
  - Sonnet gets all three columns. That separates *model strength* from *feedback* from *tool/graph access*, which is precisely R2's Q1.
- **Why:** R1.6/Q3, R2.1/Q1, R2.2/Q2.
- **Story:** a new §4.x, "How far can stronger systems get on MA-Hard?", and a new abstract sentence: "single-pass ≤ X%; the best agent reaches Y%, leaving Z% unsolved". Report cost per item beside every row.

#### E2 — Dependency-access ablation (**the key experiment**, ~1 day)
- **E2a — Single-pass dependency retrieval, a controlled ablation.** A new formatter or prompt hook; none exists today (`object_links` and `mathlib_links` are not used by any formatter). For each MA-Hard item, the prompt includes one of:
  1. nothing (the control, which already exists);
  2. **informal text** of the depth-1 `object_links` targets;
  3. the **Mathlib names** from those dependencies' `mathlib_links` (redact the item's own link);
  4. both (2) and (3);
  5. **a matched random-context control:** the same number of random definitions from the same textbook.

  Control (5) is essential. Your local-context result shows that simply adding more text *hurts*, so without it an improvement from (2)–(4) can't be attributed to the dependencies themselves.
  - Models: gpt-oss-120b (about 8 minutes per variant) and Sonnet single-pass (cheap).
  - Before running, check how many MA-Hard dependencies actually have Mathlib links. Only 14 of 698 *items* do, but their shallower dependencies should have many more.
- **E2b — Agent ablation.** Rerun A1 **without** `--mathatlas-project` (no dependency or context MCP), keeping the model, budget cap and Lean-LSP tools unchanged. About 70 minutes and $175. That gives "access to the graph buys X pp in an agent".
  - **Leak-guard caveat:** A1 has unrestricted Bash, and the unredacted dataset (including the item's own `mathlib_suggestion`) is readable at `~/src/mathatlas-formalization/data/`.
  - Before E2b, either sandbox that path and rerun *both* arms, or grep the existing A1 transcripts for reads of that path and report the result.
  - Otherwise a reviewer can argue that the 29.2% includes leakage. The dataset holds no gold Lean, only Mathlib-name hints, so the risk is moderate, but it must be closed.
- **Why:** R3.5/Q1 (their main question), R2.4 (novelty), and the central claim that the graph "facilitates … dependency-aware autoformalization".
- **Story:** Either outcome is publishable.
  - If dependencies help: MathAtlas *rewards* dependency-aware methods, which is the reason the graph exists.
  - If naive stuffing doesn't help single-pass but graph access helps the agent: *structured* access is what matters, which is consistent with the local-context finding.
  - Either way, this becomes the paper's showcase figure: correctness by condition, with CIs.

#### E3 — Judge validity at the frontier (~1 day, mostly human time)
- **E3a — Human verification of "correct".** Two annotators label about 100 outputs the judge called faithful: 50 from A1, 25 from compile-repair and 25 from single-pass. They also label about 50 outputs that compile but were judged unfaithful, from A1. Report judge precision, NPV and κ. This is **critical now that 29.2% is the headline**: agents with tools optimise against the compiler and can produce plausible-looking but unfaithful statements, and the judge over-accepts. Budget about 4 hours per annotator.
- **E3b — Multi-judge rescoring.** Rescore all ladder outputs that compile with gpt-oss-120b (ReForm prompt) and, if the budget allows, gpt-5.2. That's only a few hundred calls. Report Kendall τ between the system rankings each judge produces.
- **E3c — Judge-error-corrected correctness.** Apply Rogan–Gladen using MA-Align sensitivity and specificity (per type), with bootstrap CIs. Add a column or a footnote to the ladder table.
- **Why:** R2.3, R3.4/Q2, R1 W3, and §4.6-3.
- **Story:** "Rankings are stable across judges (τ = …). Absolute correctness is an upper bound. Human-verified precision of 'correct' is …". This turns the judge weakness into a direct use of MA-Align, which in turn strengthens MA-Align's claim to be a core contribution (R1.2).

#### E4 — Confound and robustness analysis (CPU only, ~½ day, runs in parallel)
MA-Hard spans only depths 72–80, so the depth question *has* to use the existing full-set outputs. No new generation is needed.

- **E4a — Confounds.**
  - Fit a logistic regression: `correct ~ depth + log(mass) + length + #obj_refs + in_mathlib + type`, with **textbook fixed effects** (or a random intercept).
  - Run a Cochran–Mantel–Haenszel test for the Mathlib effect, stratified by textbook.
  - Replace the Fig. 4 KDEs with binned correctness rates plus CIs, and add a within-textbook version.
- **E4b — Robustness.**
  - Recompute depth with SCC condensation, which fixes the cycle issue.
  - Then perturb the graph 100 times: drop about 9% of edges (1 − precision) and add spurious edges at the measured error rate.
  - Report Spearman ρ of depth against the original, Jaccard overlap of MA-Hard membership, and the spread of the regression coefficient.
- **Why:** R1.3/Q1, R1.5/Q2, and R3's Q3 concession.
- **Story:** "The depth effect survives textbook fixed effects (β = …) and extraction-noise perturbation." Or, if it doesn't, soften the claim to "depth marks hard subfields". The Mathlib paragraph gets written to match whichever result you find.

#### E5 — Failure taxonomy on MA-Hard (~½ day)
- **What:**
  - (a) Automatically bucket the Lean errors from existing outputs: unknown identifier or constant, failed typeclass synthesis, type mismatch, parse error, timeout. Compare the buckets across ladder rungs to show what feedback fixes and what retrieval fixes.
  - (b) Hand-label about 60 outputs that compile but are unfaithful, from A1 and gpt-oss. Suggested categories: invented or placeholder definitions, dropped hypotheses, wrong generality, vacuous or degenerate statements, wrong object.
  - (c) A1 correctness per textbook and per entity type.
- **Why:** R3.3/Q3.
- **Story:** a short subsection plus an appendix of examples. If "unknown identifier" dominates, that directly motivates E2.
- **Cost:** very low. Do (b) together with E3a, since it's the same annotators.

#### E6 — Tuned-example 2×2 (optional, ~1 h)
- **What:** Run gpt-oss-120b on MA-Hard with {base, tuned} prompt × {LeanWorkbook, graduate} examples, replacing the three MathAtlas-overlapping examples with held-out ones.
- **Why:** R3.2.
- **Story:** one sentence in §4.2 plus an appendix table.

#### Suggested schedule
| When | Work |
|---|---|
| Day 1 AM | E0 parser fix + rerun (1 h) → CIs. Write the E2a formatter. Launch E1a slicing and judging. Sandbox the dataset path and launch **E2b** (A1 without MCP) and E1b/c (Sonnet single-pass and repair). |
| Day 1 PM | E2a runs on gpt-oss and Sonnet. Run E4a/b. Judge all new outputs. |
| Day 1 evening | E3b multi-judge, E3c correction, E5a error taxonomy. |
| Day 2 | E3a + E5b human annotation (2 annotators). Optional E6. Build tables and figures, then write. |

Estimated spend: about $175 for E2b, $30–60 for Sonnet single-pass and repair, and under $30 for gpt-5.2 judging, so **roughly $250–300**. Add another $175 if you rerun A1 with the leak closed.

#### How the paper's story changes
1. **Abstract and intro:** MathAtlas + MA-Align are co-contributions. Single-pass ≤ X%; the best agent reaches Y% (judge-corrected Y′%); dependency access explains Z pp of the gap.
2. **§2:** state the depth definition precisely, with the SCC fix, and the MA-Hard selection rule.
3. **§3 → a standalone MA-Align section:** balanced metrics, fixed CriticLean-32B numbers, human verification (E3a), and the judge-sensitivity result.
4. **§4:** the existing single-pass tables with CIs → the **MA-Hard ladder (E1)** → **dependency ablation (E2)** → confounds (E4) → failure modes (E5).
5. **Limitations:** rewritten along the scope vs. measurement split.

### 4.5 Writing: weak sections and concrete fixes

**Abstract**
- Restore the MA-Align sentence (currently commented out).
- Correct the numbers.
- Add the ladder result.
- "is high quality but" is unsupported as stated. Back it with numbers or drop it.

**Introduction**
- The numbers disagree with the abstract ("10.5% … 20.3%" vs. 9.8 / 16.7). Keep one source of truth.
- Restore the commented-out contributions list as four bullets: MathAtlas, the dependency graph, MA-Hard with the ladder, MA-Align.
- Define "in the wild" in one sentence: unfiltered textbook text, with implicit context, multiple objects per entity, and prerequisites that aren't formalized.
- "~51k" → "~52k".

**§2 Dataset**
- Figure 1: "Dedeking" → "Dedekind".
- Define depth and mass formally: graph, cycle handling, unmatched references, cross-book edges.
- Move the statistics prose into a table (entity types, references per entity, depth mean/median/IQR, % grounded in Mathlib).
- Add a depth histogram by field.
- Typos: "it's formalization" → "its"; "an items dependency tree" → "an item's"; "contains of 52,052" → "contains 52,052".

**§2.3 Quality**
- Add entity recall, or state explicitly that it isn't measured. Add IAA.
- **Fix the relation-evaluation description.** It says candidates were retrieved "with LeanSearch", but relation extraction uses ChromaDB/E5 over MathAtlas entities; LeanSearch is the *Mathlib* linker.
- Add a precision figure for the Mathlib linker.
- Table 1(b) says "Precision" in its caption but reports "Valid %". Pick one.

**§3 Metrics / MA-Align — the weakest section**
- Line 313 promises "our improvements", but none are presented.
- Line 332 ends mid-sentence ("…is open source, ").
- There's no annotation protocol, annotator count or IAA.
- The labels aren't balanced (statements are 25/75). Report balanced accuracy, MCC or κ, and show the 75% majority baseline.
- The Table 1 caption says the gpt-5.2 prompt uses "examples taken from MathAtlas". But `prompts/statement_alignment.txt` (byte-identical to `definition_alignment.txt`) uses toy examples (prime, injective, …), and neither the gpt-5.2 prompt nor its MA-Align outputs are in the repo. Locate them, confirm there's **no overlap with MA-Align items**, and fix the caption.
- Promote MA-Align to its own section (see the story above).

**§4 Experiments**
- The local-context description says "500 tokens" in §4.2 and "300 tokens" in §4.3. The drops cited in the text (20.3→17.4, 10.5→8.6) appear in no table; Table 2 shows 16.7→15.2 and 7.8→6.9.
- Reword the description of the tuned examples (R3.2).
- Say in the text, not just the caption, that the Faithful column is *conditional on compiling*.
- Mathlib paragraph: present leakage as one hypothesis among several, and state which system and subset Fig. 3 uses (see §4.6-6).
- Fig. 4: switch to a binned plot of correctness rate with CIs. The per-class normalised KDEs hide the base rates and spill below depth 0.
- The claim that "fine-tuned models were unable to generalize to definitions" needs numbers, at least in the appendix.
- Specify MA-Hard's prompt in the Table 4 caption, and add the ladder (E1) and the E2 results.

**§5 Related work**
- "andwincludes" → "and includes".
- Add RAutoformalizer, miniCTX and agentic formalization systems.
- Make the FATE comparison quantitative.

**§6 Conclusion**
- "In contrast to prior benchmarks, MathAtlas~with a dependency-graph structure…" is missing a verb.
- "that that" → "that".
- Update it to the new story.

**Limitations**
- Rewrite as scope vs. measurement (R1.1). Cover:
  - extraction noise, with numbers;
  - judge error and its direction;
  - coverage and copyright bias;
  - non-bit-reproducibility;
  - statement-only evaluation;
  - dependence on Lean and Mathlib;
  - contamination risk, since public textbooks may already be in model training data.

**Checklist**
- The error-bar item says "No" because of compute. CIs over existing outputs cost nothing, so switch it to "Yes" after E0.
- "No human subjects" is fine if the annotators were authors; say so.

### 4.6 Consistency and validity issues — **verify these first**

1. **CriticLean-32B on MA-Align.**
   - `outputs/criticleangpt-qwen3-32b-rl-ma-{stmts,defs}.json` and `…-criticleanbench.json` hit a JSON-parse error on **every item**: 100/100, 100/100 and 500/500. The model wrapped its answers in ```` ```json ```` fences or `<think>` blocks, so every prediction defaulted to "misaligned".
   - The `.metrics` files report 0.75 (stmts), 0.50 (defs) and 0.50 (CriticLeanBench). The paper reports 75.0 / 80.0 / 86.4.
   - **The paper's 75.0 on statements equals that degenerate score exactly.**
   - If the paper's numbers came from a different run that parsed correctly, it isn't in the repo. If they didn't, the paper's primary metric was never actually validated on MA-Align.
2. **MA-Align statements are 75 misaligned / 25 aligned** (definitions are 50/50), yet the paper says sampling was done "to better balance the label distribution". On statements, every judge except gpt-5.2 scores **at or below the trivial 75% baseline**:

   | Judge | Accuracy on MA-Align stmts |
   |---|---|
   | CriticLean-14B | 56 |
   | gpt-oss-120b | 61 / 65 |
   | Qwen3 variants | 57–71 |
   | *Always "misaligned"* | *75* |

3. **The judge error is directional.** The confusion matrices on disk for MA-Align statements:

   | Judge | Sensitivity (aligned) | Specificity (misaligned) |
   |---|---|---|
   | CriticLean-14B | 0.92 | 0.44 |
   | gpt-oss-120b | 0.92 | 0.48 |

   Low specificity means over-acceptance, which inflates "faithful" and "correct". As an illustration, plugging ReForm's 48.4% observed faithful rate into Rogan–Gladen with those sensitivity and specificity values gives an estimate *below zero*. The real 32B judge will differ, but if it behaves similarly, **the statement correctness numbers are upper bounds.** E0 and E3 must settle this.
4. **ReForm headline.** 19.0% compile × 48.4% faithful = **9.2%**, not 9.8%. Every other row of Table 2 satisfies Compiles × Faithful = Correct, and the Open-Split ReForm row is 9.2%. The abstract, intro and conclusion all quote 9.8%. The inventory also suggests `outputs/reform.statements.aligned.json` may have been scored by a JSON-prompt judge rather than CriticLean; please confirm.
5. **Table 5 (Open Split) looks partly copy-pasted.**
   - Several rows match full-set rows exactly: gpt-oss-120b defs few-shot (24.2/57.0/13.8) and +tuned prompt (26.0/58.8/15.3); gpt-oss-20b statements few-shot, +tuned prompt and +tuned examples; gpt-oss-120b statements +tuned prompt. Identical triples on a 70% subset are implausible.
   - Meanwhile gpt-oss-20b zero-shot defs Faithful jumps from 20.9% to 48.4%.
   - Regenerate the table from the outputs.
6. **Fig. 3 (Mathlib)** shows 16.8% correct for "missing" and 27.9% for "present". Any mixture of the two must be at least 16.8%, but the best definitions system scores 16.7% overall. So the figure must use a different system or subset; say which.
7. **The local-context numbers and token counts disagree** with each other and with Table 2 (see §4.5). The intro's "10.5% / 20.3%" look stale.
8. **MA-Hard definition and coverage.**
   - The selection rule is undocumented. It reconstructs as non-proof items with depth ≥ 72.
   - The paper says "14 subjects", but the data show 37 textbooks.
   - Only prompted gpt-oss models were evaluated.
   - The depth DFS depends on traversal order when cycles are present (§4.3-5).
9. **Tuned-example overlap with MathAtlas** (Milne, Robbin, Knapp; see §3, R3) contradicts both the rebuttal and the paper's "annotated, graduate-level examples" framing.

### 4.7 Appendix additions

- **Datasheet:** a per-textbook table (title, field, license, open or Springer, #entities, mean depth), plus a comparison of the full-set and Open-Split distributions (R3.7).
- **Depth and mass computation:** the algorithm, SCC and cycle handling, how unmatched references are treated, and depth histograms by field.
- **MA-Hard construction:** the rule (depth ≥ 72, no proofs), composition by textbook, field and type (note the 23% from Vakil), and per-textbook macro-averaged results.
- **MA-Align construction:**
  - sampling procedure and annotation guidelines;
  - annotator background and count, IAA;
  - per-split label distribution, and confusion matrices for every judge;
  - 3–4 worked examples (aligned and misaligned);
  - confirmation of no overlap with the judge's few-shot examples.
- **Quality-annotation guidelines** for entities, references and relations, with example errors, plus a precision sample for the Mathlib linker.
- **All formalization and judge prompts.** §4.2 says "see App. B for full prompts", but App. B only contains the extraction and grounding prompts. The zero-shot, few-shot, tuned, local-context and judge prompts are missing.
- **Experimental details:**
  - Mathlib commit (only the Lean version is given) and compile timeout;
  - how outputs with multiple statements are compiled and judged;
  - how degenerate outputs are handled (`True`, `sorry`, axioms);
  - sampling parameters for each model, and seeds.
- **Fine-tuned models on definitions:** the numbers behind "unable to generalize".
- **Agentic and iterative setup:** tool list, MCP interface, per-item budget cap (58 of 698 items hit the $0.50 cap), leak audit, and cost, time and tokens per item.
- **Full CIs and paired tests** for every table.
- **Qualitative examples:** successes and failures at shallow vs. deep depth, and one example per failure category (E5).
- **Reproducibility statement:** nondeterminism, released intermediate outputs, model revisions.

---

## 5. Open questions for you

1. **Where did the CriticLean-32B MA-Align numbers (80.0 / 75.0 / 86.4) come from?** Every on-disk run is a parse failure (§4.6-1). The answer decides whether E0 is a formality or a must-fix that forces rescoring every table.
2. **ReForm 9.8% vs. 9.2%:** which is correct, and which judge scored ReForm?
3. **The gpt-5.2 MA-Align prompt and outputs:** where are they, and do its few-shot examples overlap MA-Align?
4. **A1 details:** which Sonnet version was used, and did any A1 transcripts read `~/src/mathatlas-formalization/data/` (the unredacted Mathlib hints)?
5. **Venue:** is this for the NeurIPS camera-ready or a resubmission (e.g. ICLR, D&B)? The answer affects how much restructuring is feasible, such as a standalone MA-Align section.
