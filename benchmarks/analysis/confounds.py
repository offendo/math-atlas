#!/usr/bin/env python3
"""E4a -- is "depth predicts failure" just "these textbooks are hard"?

Uses the paper's own full-set per-item correctness (the runs behind Fig. 3/4)
plus dependency features, and asks whether the depth effect survives controls:

  * pooled logistic regression: correct ~ depth (standardized)
  * + controls: log mass, log text length, #object references, Mathlib-grounded, type
  * + **textbook fixed effects**: linear probability model with textbook dummies and
    textbook-clustered SEs (a fixed-effects logit is not identified for books with
    zero successes, which are common here)
  * the same FE model on each perturbed depth vector from scripts/graph_depth.py
    (coefficient spread under extraction noise)
  * Mathlib effect (definitions): pooled chi-square vs Cochran-Mantel-Haenszel
    stratified by textbook, and the FE model's coefficient
  * binned correctness-vs-depth plot with Wilson CIs, overall and within-textbook
    depth tertiles

    python benchmarks/analysis/confounds.py \
        --system reform-8b:statements=/path/reform.statements.aligned.json \
        --system gpt-oss-120b:definitions=/path/...definition_few_shot_tuned_examples.json \
        --depth outputs/iclr/depth/depth.rewire-backward.csv.gz --output-dir outputs/iclr/analysis/confounds
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy import stats

app = typer.Typer(pretty_exceptions_show_locals=False)
DEFAULT_DATA = Path("~/src/mathatlas-formalization/data/MathAtlas.json").expanduser()


def as_obj(x):
    if isinstance(x, str) and x[:1] in "{[":
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(x)
            except Exception:
                pass
    return x


def verified_of(rec) -> bool:
    co = as_obj(rec.get("compiler_output"))
    if isinstance(co, dict) and "verified" in co:
        return bool(co["verified"])
    return bool(rec.get("verified", False))


def aligned_of(rec) -> bool | None:
    a = rec.get("aligned")
    if isinstance(a, (bool, np.bool_)):
        return bool(a)
    if isinstance(a, str) and a in ("aligned", "misaligned"):
        return a == "aligned"
    ao = as_obj(rec.get("alignment_output"))
    if isinstance(ao, dict):
        if "aligned" in ao:
            return bool(ao["aligned"])
        if "result" in ao:
            return str(ao["result"]).lower() in ("aligned", "correct", "true")
    return None


def load_correctness(path: str) -> pd.DataFrame:
    with open(path) as f:
        obj = json.load(f)
    recs = pd.DataFrame(obj).to_dict("records") if isinstance(obj, dict) else obj
    rows = []
    for r in recs:
        v, a = verified_of(r), aligned_of(r)
        rows.append({"uuid": r["uuid"], "verified": v, "aligned": a})
    df = pd.DataFrame(rows).drop_duplicates("uuid")
    df["correct"] = df["verified"] & df["aligned"].fillna(False).astype(bool)
    return df


def features(data: Path) -> pd.DataFrame:
    rows = []
    with open(data) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"uuid": r["uuid"], "file_id": r.get("file_id"), "type": r.get("type"),
                         "len_chars": len(r.get("text") or ""), "n_obj_refs": len(r.get("object_references") or []),
                         "in_mathlib": bool(r.get("found_mathlib_link"))})
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fit_models(df: pd.DataFrame, depth_col: str) -> dict:
    import statsmodels.formula.api as smf

    d = df.copy()
    d["depth_z"] = (d[depth_col] - d[depth_col].mean()) / d[depth_col].std()
    d["log_mass"] = np.log1p(d["mass"])
    d["log_len"] = np.log(d["len_chars"].clip(lower=1))
    d["y"] = d["correct"].astype(int)
    d["in_mathlib_i"] = d["in_mathlib"].astype(int)
    out = {}
    multi_type = d["type"].nunique() > 1
    ctrl = "log_mass + log_len + n_obj_refs + in_mathlib_i" + (" + C(type)" if multi_type else "")
    try:
        m1 = smf.logit("y ~ depth_z", d).fit(disp=0)
        out["logit_pooled"] = {"coef_depth_z": m1.params["depth_z"], "p": m1.pvalues["depth_z"],
                               "odds_ratio_per_sd": float(np.exp(m1.params["depth_z"]))}
        m2 = smf.logit(f"y ~ depth_z + {ctrl}", d).fit(disp=0)
        out["logit_controls"] = {"coef_depth_z": m2.params["depth_z"], "p": m2.pvalues["depth_z"],
                                 "odds_ratio_per_sd": float(np.exp(m2.params["depth_z"])),
                                 "coef_in_mathlib": m2.params["in_mathlib_i"], "p_in_mathlib": m2.pvalues["in_mathlib_i"]}
    except Exception as e:  # separation etc.
        out["logit_error"] = str(e)
    groups = pd.factorize(d["file_id"])[0]
    lpm0 = smf.ols("y ~ depth_z", d).fit(cov_type="cluster", cov_kwds={"groups": groups})
    lpm = smf.ols(f"y ~ depth_z + {ctrl} + C(file_id)", d).fit(cov_type="cluster", cov_kwds={"groups": groups})
    out["lpm_pooled"] = {"pp_per_sd_depth": 100 * lpm0.params["depth_z"], "p": lpm0.pvalues["depth_z"]}
    out["lpm_textbook_fe"] = {"pp_per_sd_depth": 100 * lpm.params["depth_z"], "p": lpm.pvalues["depth_z"],
                              "ci95_pp": [100 * x for x in lpm.conf_int().loc["depth_z"].tolist()],
                              "pp_in_mathlib": 100 * lpm.params["in_mathlib_i"], "p_in_mathlib": lpm.pvalues["in_mathlib_i"],
                              "n": int(lpm.nobs), "n_textbooks": int(d["file_id"].nunique())}
    return out


def fe_depth_coef(df: pd.DataFrame, depth_col: str) -> float:
    import statsmodels.formula.api as smf

    d = df.copy()
    d["depth_z"] = (d[depth_col] - d[depth_col].mean()) / d[depth_col].std()
    d["log_mass"] = np.log1p(d["mass"]); d["log_len"] = np.log(d["len_chars"].clip(lower=1))
    d["y"] = d["correct"].astype(int); d["in_mathlib_i"] = d["in_mathlib"].astype(int)
    ctrl = "log_mass + log_len + n_obj_refs + in_mathlib_i" + (" + C(type)" if d["type"].nunique() > 1 else "")
    return 100 * smf.ols(f"y ~ depth_z + {ctrl} + C(file_id)", d).fit().params["depth_z"]


def mathlib_cmh(df: pd.DataFrame) -> dict:
    from statsmodels.stats.contingency_tables import StratifiedTable

    tables = []
    for _, g in df.groupby("file_id"):
        t = pd.crosstab(g["in_mathlib"], g["correct"]).reindex(index=[True, False], columns=[True, False], fill_value=0)
        if (t.sum(axis=1) > 0).all() and (t.sum(axis=0) > 0).all():
            tables.append(t.to_numpy())
    pooled = pd.crosstab(df["in_mathlib"], df["correct"]).reindex(index=[True, False], columns=[True, False], fill_value=0)
    chi2 = stats.chi2_contingency(pooled.to_numpy())
    out = {"rate_in_mathlib": float(df.loc[df["in_mathlib"], "correct"].mean()),
           "rate_not_in_mathlib": float(df.loc[~df["in_mathlib"], "correct"].mean()),
           "pooled_chi2_p": float(chi2.pvalue),
           "n_informative_textbooks": len(tables)}
    if tables:
        st = StratifiedTable(tables)
        out.update(mh_pooled_odds_ratio=float(st.oddsratio_pooled), cmh_p=float(st.test_null_odds().pvalue),
                   breslow_day_homogeneity_p=float(st.test_equal_odds().pvalue))
    return out


def plot(df: pd.DataFrame, depth_col: str, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = list(range(0, int(df[depth_col].max()) + 10, 10))
    df = df.assign(bin=pd.cut(df[depth_col], bins=bins, right=False))
    g = df.groupby("bin", observed=True)["correct"].agg(["sum", "count"])
    x = [b.left + 5 for b in g.index]
    rates = g["sum"] / g["count"]
    lo, hi = zip(*[wilson(k, n) for k, n in zip(g["sum"], g["count"])])
    axes[0].errorbar(x, 100 * rates, yerr=[100 * (rates - np.array(lo)), 100 * (np.array(hi) - rates)], fmt="o-", capsize=3)
    for xi, n in zip(x, g["count"]):
        axes[0].annotate(f"n={n}", (xi, 0), fontsize=6, ha="center", va="bottom")
    axes[0].set(xlabel="dependency depth (bin)", ylabel="correct % (95% CI)", title=f"{title}: pooled")
    # within-textbook depth tertiles
    df = df.copy()
    df["tertile"] = df.groupby("file_id")[depth_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if len(s) >= 3 else np.nan)
    t = df.dropna(subset=["tertile"]).groupby("tertile")["correct"].agg(["sum", "count"])
    r = t["sum"] / t["count"]
    lo, hi = zip(*[wilson(k, n) for k, n in zip(t["sum"], t["count"])])
    axes[1].errorbar(["shallow", "middle", "deep"], 100 * r, yerr=[100 * (r - np.array(lo)), 100 * (np.array(hi) - r)], fmt="o-", capsize=3)
    axes[1].set(xlabel="depth tertile *within textbook*", ylabel="correct %", title=f"{title}: within-book")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


@app.command()
def run(
    system: list[str] = typer.Option(..., help="`name:split=path` (split: statements|definitions); repeatable."),
    depth: Path = typer.Option(..., help="depth.<noise>.csv.gz from scripts/graph_depth.py"),
    data: Path = typer.Option(DEFAULT_DATA),
    output_dir: Path = typer.Option(Path("outputs/iclr/analysis/confounds")),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    feats = features(data)
    dep = pd.read_csv(depth)
    rep_cols = [c for c in dep.columns if c.startswith("depth_rep")]
    keep = ["uuid", "depth_scc", "mass_scc"] + (["depth_old", "mass_old"] if "depth_old" in dep.columns else []) + rep_cols
    feats = feats.merge(dep[keep], on="uuid", how="left")
    results = {}
    for spec in system:
        label, _, path = spec.partition("=")
        name, _, split = label.partition(":")
        corr = load_correctness(path)
        df = corr.merge(feats, on="uuid", how="inner")
        df = df[df["type"] == "definition"] if split == "definitions" else df[~df["type"].isin(["definition", "proof"])]
        res = {"n": len(df), "correct_rate": float(df["correct"].mean()),
               "aligned_missing_frac": float(corr["aligned"].isna().mean())}
        for dcol, mcol in [("depth_old", "mass_old"), ("depth_scc", "mass_scc")]:
            if dcol in df.columns and df[dcol].notna().any():
                d = df.dropna(subset=[dcol]).assign(mass=lambda x: x[mcol])
                res[dcol] = fit_models(d, dcol)
        if rep_cols:
            d = df.assign(mass=df["mass_scc"])
            coefs = [fe_depth_coef(d.dropna(subset=[c]), c) for c in rep_cols]
            res["lpm_textbook_fe_under_noise"] = {"reps": len(coefs), "pp_per_sd_mean": float(np.mean(coefs)),
                                                  "pp_per_sd_p2.5": float(np.percentile(coefs, 2.5)),
                                                  "pp_per_sd_p97.5": float(np.percentile(coefs, 97.5))}
        if split == "definitions":
            res["mathlib_effect"] = mathlib_cmh(df)
        plot(df.dropna(subset=["depth_scc"]), "depth_scc", name, output_dir / f"depth_vs_correct.{name}.png")
        results[f"{name}:{split}"] = res
        print(f"== {name}:{split}\n{json.dumps(res, indent=2, default=float)}")
    (output_dir / "confounds.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    app()
