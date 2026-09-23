#!/usr/bin/env python3
"""E0/E3 -- the ICLR results tables, with uncertainty.

For every MA-Hard output file:
  * compile / faithful-of-compiling / joint rates with Wilson 95% CIs
  * joint rate excluding degenerate (`: True`-style) outputs
  * statements vs definitions joint rates
  * **judge-error-corrected joint** (Rogan-Gladen): the observed "aligned" rate
    among compiling items of each type is corrected with the judge's MA-Align
    sensitivity/specificity for that type (from judge_validation.py), with a
    bootstrap CI that resamples both the items and the validation confusion
    counts. Because MA-Align is small, this CI is wide -- which is the point.
  * cost and turns where recorded

Plus:
  * paired exact McNemar tests (on per-item joint correctness) for the planned
    contrasts, Holm-adjusted
  * multi-judge agreement (E3b): per-run joint under each judge, Kendall tau of
    the system rankings, and per-item Cohen kappa between judges

    python benchmarks/analysis/report.py \
        --runs 'outputs/iclr/iterative/*.json' --runs 'outputs/iclr/agentic/*.json' \
        --runs 'outputs/iclr/sliced/*.json' --runs 'outputs/iterative/*.json' --runs 'outputs/agentic/*.json' \
        --validation-dir outputs/iclr/judge-validation --judge-tag criticlean-32b \
        --rejudge-dir outputs/iclr/rejudge --output-dir outputs/iclr/analysis
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy import stats

app = typer.Typer(pretty_exceptions_show_locals=False)
DEF_TYPES = {"definition"}
SKIP_SUFFIXES = (".metrics.json", "coverage.json", "pairwise_identity.json")

# (A, B): is A better than B?  Names are output-file stems.
CONTRASTS = [
    ("gpt-oss-120b.sp.dep-informal", "gpt-oss-120b.sp.dep-none"),
    ("gpt-oss-120b.sp.dep-mathlib", "gpt-oss-120b.sp.dep-none"),
    ("gpt-oss-120b.sp.dep-both", "gpt-oss-120b.sp.dep-none"),
    ("gpt-oss-120b.sp.dep-random", "gpt-oss-120b.sp.dep-none"),
    ("gpt-oss-120b.sp.dep-both", "gpt-oss-120b.sp.dep-random"),
    ("gpt-oss-120b.k5.dep-none", "gpt-oss-120b.sp.dep-none"),
    ("gpt-oss-120b.k5.dep-both", "gpt-oss-120b.k5.dep-none"),
    ("sonnet.sp.dep-both", "sonnet.sp.dep-none"),
    ("sonnet.sp.dep-random", "sonnet.sp.dep-none"),
    ("sonnet.sp.dep-both", "sonnet.sp.dep-random"),
    ("sonnet.k5.dep-none", "sonnet.sp.dep-none"),
    ("gpt-5.2.sp.dep-both", "gpt-5.2.sp.dep-none"),
    ("claude-code-sonnet.none", "sonnet.k5.dep-none"),
    ("claude-code-sonnet.opt", "claude-code-sonnet.none"),
    ("claude-code-sonnet.dep", "claude-code-sonnet.none"),
    ("claude-code-sonnet.dep", "claude-code-sonnet.opt"),
    ("gpt-oss-120b.sp.dep-none", "gpt-oss-120b.control"),  # re-run vs Sep-6 control (should be ~equal)
    ("claude-code-sonnet.opt", "claude-code-sonnet.ma-hard"),  # re-run vs Sep-6 A1 (Mathlib version changed)
]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(p: float, ci: tuple[float, float] | None = None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "–"
    s = f"{100 * p:.1f}"
    return s + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else "")


def load_run(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_json(path)
    except ValueError:
        return None
    if "uuid" not in df.columns or "verified" not in df.columns:
        return None
    df["verified"] = df["verified"].fillna(False).astype(bool)
    if "aligned" in df.columns:
        df["aligned"] = df["aligned"].fillna(False).astype(bool)
    df["is_def"] = df["type"].isin(DEF_TYPES)
    df["correct"] = df["verified"] & df["aligned"] if "aligned" in df.columns else np.nan
    return df.drop_duplicates("uuid")


def rogan_gladen(obs: float, sens: float, spec: float) -> float:
    denom = sens + spec - 1
    if denom <= 0:
        return float("nan")
    return float(np.clip((obs + spec - 1) / denom, 0.0, 1.0))


def corrected_joint(df: pd.DataFrame, val: dict | None, n_boot: int = 2000, seed: int = 0):
    """Judge-error-corrected joint rate + bootstrap CI (items and validation counts resampled)."""
    if val is None or "aligned" not in df.columns:
        return None, None
    conf = {"def": val.get("ma-align-defs", {}).get("confusion"), "stmt": val.get("ma-align-stmts", {}).get("confusion")}
    if not conf["def"] or not conf["stmt"]:
        return None, None
    rng = np.random.default_rng(seed)
    n = len(df)
    groups = {"def": df[df["is_def"]], "stmt": df[~df["is_def"]]}

    def estimate(frames, sens_spec):
        total = 0.0
        for key, g in frames.items():
            comp = g[g["verified"]]
            if len(comp) == 0:
                continue
            total += len(comp) * rogan_gladen(comp["aligned"].mean(), *sens_spec[key])
        return total / n

    def ss(c):
        return c["tp"] / (c["tp"] + c["fn"]), c["tn"] / (c["tn"] + c["fp"])

    point = estimate(groups, {k: ss(c) for k, c in conf.items()})
    boots = []
    for _ in range(n_boot):
        frames = {k: g.sample(len(g), replace=True, random_state=int(rng.integers(1 << 31))) for k, g in groups.items()}
        sspec = {}
        for k, c in conf.items():
            pos, neg = c["tp"] + c["fn"], c["tn"] + c["fp"]
            sspec[k] = (rng.binomial(pos, c["tp"] / pos) / pos, rng.binomial(neg, c["tn"] / neg) / neg)
        b = estimate(frames, sspec)
        if not math.isnan(b):
            boots.append(b)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else None
    return point, ci


def run_row(name: str, df: pd.DataFrame, cfg: dict, val: dict | None) -> dict:
    n = len(df)
    k_c = int(df["verified"].sum())
    row = {"run": name, "n": n, "compile": k_c / n, "compile_ci": wilson(k_c, n)}
    if "aligned" in df.columns:
        k_j = int(df["correct"].sum())
        row.update(
            faithful_of_compiling=df.loc[df["verified"], "aligned"].mean() if k_c else float("nan"),
            joint=k_j / n,
            joint_ci=wilson(k_j, n),
            joint_nondegenerate=float((df["correct"] & ~df.get("degenerate", pd.Series(False, index=df.index)).fillna(False).astype(bool)).mean()),
            joint_stmt=df.loc[~df["is_def"], "correct"].mean() if (~df["is_def"]).any() else float("nan"),
            joint_def=df.loc[df["is_def"], "correct"].mean() if df["is_def"].any() else float("nan"),
        )
        row["joint_corrected"], row["joint_corrected_ci"] = corrected_joint(df, val)
    if "degenerate" in df.columns:
        row["degenerate"] = float(df["degenerate"].fillna(False).astype(bool).mean())
    if "cost_usd" in df.columns and df["cost_usd"].notna().any():
        row["cost_total"] = float(df["cost_usd"].sum())
    elif cfg.get("generation_cost_usd"):
        row["cost_total"] = float(cfg["generation_cost_usd"])
    if "agent_error" in df.columns:
        row["budget_capped"] = float((df["agent_error"] == "error_max_budget_usd").mean())
        row["agent_errors"] = float(df["agent_error"].notna().mean())
    if "num_turns" in df.columns and df["num_turns"].notna().any():
        row["mean_turns"] = float(df["num_turns"].mean())
    row["config"] = {k: cfg.get(k) for k in ("baseline", "model", "dependency_context", "max_rounds", "prompt_file",
                                             "theorem_prompt_file", "definition_prompt_file", "mcp_servers", "judge_model")}
    return row


def mcnemar(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    m = a[["uuid", "correct"]].merge(b[["uuid", "correct"]], on="uuid", suffixes=("_a", "_b"))
    ca, cb = m["correct_a"].astype(bool), m["correct_b"].astype(bool)
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    p = stats.binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"n_paired": len(m), "a_only": n10, "b_only": n01, "delta": float(ca.mean() - cb.mean()), "p": float(p)}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


@app.command()
def run(
    runs: list[str] = typer.Option(..., help="Glob(s) of MA-Hard output JSONs; repeatable."),
    validation_dir: Path = typer.Option(Path("outputs/iclr/judge-validation")),
    judge_tag: str = typer.Option("criticlean-32b", help="Validation tag of the primary judge."),
    rejudge_dir: Path = typer.Option(Path("outputs/iclr/rejudge")),
    output_dir: Path = typer.Option(Path("outputs/iclr/analysis")),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    val_path = validation_dir / f"{judge_tag}.metrics.json"
    val = json.loads(val_path.read_text())["benchmarks"] if val_path.exists() else None

    frames: dict[str, pd.DataFrame] = {}
    rows = []
    for pattern in runs:
        for path in sorted(glob.glob(pattern)):
            if path.endswith(SKIP_SUFFIXES):
                continue
            df = load_run(path)
            if df is None:
                continue
            name = Path(path).stem
            if name in frames:  # same stem in two dirs: keep the first (campaign dirs come first)
                continue
            mpath = Path(path).with_suffix(".metrics.json")
            cfg = json.loads(mpath.read_text()).get("config", {}) if mpath.exists() else {}
            frames[name] = df
            rows.append(run_row(name, df, cfg, val))

    # ---- judge validation table (E0)
    lines = ["# MA-Hard results (auto-generated by benchmarks/analysis/report.py)", ""]
    vrows = []
    for mf in sorted(validation_dir.glob("*.metrics.json")):
        v = json.loads(mf.read_text())
        for bench, m in v["benchmarks"].items():
            vrows.append({"judge": v["tag"], "benchmark": bench, **{k: m[k] for k in (
                "n", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "cohen_kappa", "mcc",
                "majority_baseline", "parse_errors", "judge_call_errors")}})
    if vrows:
        vt = pd.DataFrame(vrows)
        lines += ["## E0 -- judge validation", "",
                  vt.to_markdown(index=False, floatfmt=".3f"), ""]

    # ---- main table
    table = []
    for r in rows:
        table.append({
            "run": r["run"], "n": r["n"],
            "compile % [95% CI]": fmt(r["compile"], r["compile_ci"]),
            "faithful|compile %": fmt(r.get("faithful_of_compiling")),
            "joint % [95% CI]": fmt(r.get("joint"), r.get("joint_ci")),
            "joint, judge-corrected % [95% CI]": fmt(r.get("joint_corrected"), r.get("joint_corrected_ci")) if r.get("joint_corrected") is not None else "–",
            "joint non-degenerate %": fmt(r.get("joint_nondegenerate")),
            "stmts %": fmt(r.get("joint_stmt")), "defs %": fmt(r.get("joint_def")),
            "degenerate %": fmt(r.get("degenerate")),
            "cost $": f"{r['cost_total']:.2f}" if r.get("cost_total") else "–",
            "turns": f"{r['mean_turns']:.1f}" if r.get("mean_turns") else "–",
            "budget-capped %": fmt(r.get("budget_capped")) if "budget_capped" in r else "–",
        })
    lines += ["## Runs (joint = compiles AND judged faithful)", "",
              pd.DataFrame(table).to_markdown(index=False), "",
              "Judge-corrected: Rogan-Gladen with the primary judge's MA-Align sensitivity/specificity "
              f"(`{judge_tag}`), per item type; CI bootstraps items and validation counts.", ""]

    # ---- contrasts
    crow = []
    for a, b in CONTRASTS:
        if a in frames and b in frames and "aligned" in frames[a] and "aligned" in frames[b]:
            crow.append({"A": a, "B": b, **mcnemar(frames[a], frames[b])})
    if crow:
        for r, p in zip(crow, holm([r["p"] for r in crow])):
            r["p_holm"] = p
        ct = pd.DataFrame(crow)
        ct["delta"] = (100 * ct["delta"]).round(1)
        lines += ["## Paired contrasts (exact McNemar on joint correctness; delta in pp, A - B)", "",
                  ct.to_markdown(index=False, floatfmt=".4f"), ""]

    # ---- multi-judge agreement (E3b)
    judges = {"criticlean-32b (primary)": frames}
    if rejudge_dir.exists():
        for jdir in sorted(p for p in rejudge_dir.iterdir() if p.is_dir()):
            jf = {}
            for path in jdir.glob("*.json"):
                if path.name.endswith(SKIP_SUFFIXES):
                    continue
                df = load_run(str(path))
                if df is not None and "aligned" in df.columns:
                    jf[path.stem] = df
            if jf:
                judges[jdir.name] = jf
    if len(judges) > 1:
        common_runs = sorted(set.intersection(*[set(k for k, v in j.items() if "aligned" in v) for j in judges.values()]))
        jt = pd.DataFrame({j: {r: 100 * judges[j][r]["correct"].mean() for r in common_runs} for j in judges}).round(1)
        lines += ["## E3b -- joint % under each judge", "", jt.to_markdown(), ""]
        names = list(judges)
        agree = []
        for i in range(len(names)):
            for k in range(i + 1, len(names)):
                tau = stats.kendalltau(jt[names[i]], jt[names[k]]).statistic if len(common_runs) > 2 else float("nan")
                ys, xs = [], []
                for r in common_runs:
                    m = judges[names[i]][r][["uuid", "verified", "aligned"]].merge(
                        judges[names[k]][r][["uuid", "aligned"]], on="uuid")
                    m = m[m["verified"]]
                    ys += m["aligned_x"].tolist(); xs += m["aligned_y"].tolist()
                from sklearn.metrics import cohen_kappa_score
                kappa = cohen_kappa_score(ys, xs) if len(set(ys) | set(xs)) > 1 else float("nan")
                agree.append({"judge_a": names[i], "judge_b": names[k], "kendall_tau_over_runs": tau,
                              "item_kappa_on_compiling": kappa, "n_items": len(ys)})
        lines += [pd.DataFrame(agree).to_markdown(index=False, floatfmt=".3f"), ""]

    (output_dir / "report.md").write_text("\n".join(lines))
    (output_dir / "report.json").write_text(json.dumps({"runs": rows, "contrasts": crow}, indent=2, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    app()
