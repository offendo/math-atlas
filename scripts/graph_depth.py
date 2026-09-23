#!/usr/bin/env python3
"""E4b -- dependency depth done right, and how robust it is to extraction noise.

`scripts/graph-stats.py` (which produced the paper's depths) runs a memoized DFS
over `object_links`. When it meets a cycle it returns depth 0 for the back edge
*and caches the partial result*, so the depth of any node on or above a cycle
depends on traversal order; `dependency_mass` also counts through that cache.

Here: same edge set (object-reference edges, all candidates, targets inside the
dataset), self-edges dropped, then
  * strongly connected components are condensed (mutually dependent items form
    one node), and depth = longest path in the condensation DAG (in SCC hops);
  * mass = number of distinct items reachable (exact, via the DAG).

Robustness (R1-Q1): relation extraction precision is ~91%, so each replicate
perturbs a fraction `--rewire` of edges and recomputes depth. Noise models
(`--noise`): `drop` (the wrong edges are simply absent), `rewire-backward` (a
wrong link points to a random *earlier* item of the same textbook -- the
realistic case, since references point back), `rewire` (any item of the same
textbook; harshest, can create forward edges and cycles). Reported:
Spearman rho of perturbed vs clean depth, and Jaccard overlap of the MA-Hard
selection (top-k non-proof items by depth, k = the real MA-Hard size). The first
`--save-reps` perturbed depth vectors are saved so confounds.py can re-fit the
depth effect on them.

    python scripts/graph_depth.py --output-dir outputs/iclr/depth
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import typer
from scipy import stats

app = typer.Typer(pretty_exceptions_show_locals=False)
DEFAULT_DATA = Path("~/src/mathatlas-formalization/data/MathAtlas.json").expanduser()
OLD_STATS = Path("/home/npatel37/src/math-atlas/math-atlas-dependency-stats.json")


def load_records(path: Path) -> dict[str, dict]:
    recs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("uuid"):
                    recs[r["uuid"]] = r
    return recs


def object_edges(recs: dict[str, dict]) -> list[tuple[str, str]]:
    edges = set()
    for u, r in recs.items():
        for cands in r.get("object_links") or []:
            for v in cands or []:
                if v and v != u and v in recs:
                    edges.add((u, v))
    return sorted(edges)


def depth_and_mass(nodes: list[str], edges: list[tuple[str, str]], with_mass: bool = True):
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    cond = nx.condensation(g)  # edge c1 -> c2: something in c1 depends on something in c2
    member = cond.graph["mapping"]
    size = {c: len(cond.nodes[c]["members"]) for c in cond.nodes}
    depth: dict[int, int] = {}
    for c in reversed(list(nx.topological_sort(cond))):  # dependencies before dependents
        succ = list(cond.successors(c))
        depth[c] = 1 + max(depth[s] for s in succ) if succ else 0
    node_depth = {n: depth[member[n]] for n in nodes}
    node_mass = None
    if with_mass:
        mass_c = {c: sum(size[d] for d in nx.descendants(cond, c)) for c in cond.nodes}
        node_mass = {n: mass_c[member[n]] + size[member[n]] - 1 for n in nodes}
    scc_size = {n: size[member[n]] for n in nodes}
    return node_depth, node_mass, scc_size


def top_k(depth: dict[str, int], eligible: list[str], k: int) -> set[str]:
    return set(sorted(eligible, key=lambda u: (-depth[u], u))[:k])


@app.command()
def run(
    data: Path = typer.Option(DEFAULT_DATA),
    old_stats: Path = typer.Option(OLD_STATS, help="The paper's depth file (graph-stats.py output)."),
    output_dir: Path = typer.Option(Path("outputs/iclr/depth")),
    reps: int = typer.Option(100, help="Perturbation replicates."),
    rewire: float = typer.Option(0.09, help="Fraction of edges perturbed per replicate (1 - relation precision)."),
    noise: str = typer.Option("rewire-backward", help="drop | rewire-backward | rewire"),
    save_reps: int = typer.Option(20, help="Save this many perturbed depth vectors for confounds.py."),
    ma_hard_dataset: str = typer.Option("offendo/math-atlas-official"),
    seed: int = typer.Option(0),
):
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    recs = load_records(data)
    nodes = list(recs)
    edges = object_edges(recs)
    print(f"{len(nodes)} items, {len(edges)} object-reference edges (self-edges dropped)")

    depth, mass, scc = depth_and_mass(nodes, edges)
    df = pd.DataFrame({"uuid": nodes, "depth_scc": [depth[n] for n in nodes], "mass_scc": [mass[n] for n in nodes],
                       "scc_size": [scc[n] for n in nodes], "type": [recs[n].get("type") for n in nodes],
                       "file_id": [recs[n].get("file_id") for n in nodes]})
    summary: dict = {"n_items": len(nodes), "n_edges": len(edges),
                     "items_in_cycles": int((df["scc_size"] > 1).sum()),
                     "largest_scc": int(df["scc_size"].max())}

    hard = set(load_dataset(ma_hard_dataset, split="hard")["uuid"])
    df["in_ma_hard"] = df["uuid"].isin(hard)
    eligible = df.loc[df["type"] != "proof", "uuid"].tolist()
    k = len(hard)

    if old_stats.exists():
        old = pd.read_json(old_stats)[["uuid", "max_depth", "dependency_mass"]]
        df = df.merge(old.rename(columns={"max_depth": "depth_old", "dependency_mass": "mass_old"}), on="uuid", how="left")
        both = df.dropna(subset=["depth_old"])
        summary["old_vs_scc"] = {
            "n_matched": len(both),
            "spearman_depth": float(stats.spearmanr(both["depth_old"], both["depth_scc"]).statistic),
            "frac_depth_changed": float((both["depth_old"] != both["depth_scc"]).mean()),
            "spearman_mass": float(stats.spearmanr(both["mass_old"], both["mass_scc"]).statistic),
        }
        # How MA-Hard (old depth >= 72, non-proof) looks under the corrected depth.
        new_top = top_k(depth, eligible, k)
        summary["ma_hard_jaccard_old_vs_scc_topk"] = len(hard & new_top) / len(hard | new_top)

    # ---- perturbation robustness
    rng = random.Random(seed)
    by_book: dict[str, list[str]] = {}
    for n in nodes:
        by_book.setdefault(recs[n].get("file_id"), []).append(n)
    def start(n):
        try:
            return int(recs[n].get("item_start") or 0)
        except (TypeError, ValueError):
            return 0
    for book in by_book.values():
        book.sort(key=start)
    pos = {n: i for book in by_book.values() for i, n in enumerate(book)}
    base_top = top_k(depth, eligible, k)
    rhos, jacc, saved = [], [], {}
    base_vec = np.array([depth[u] for u in eligible])
    for r in range(reps):
        new_edges = []
        for u, v in edges:
            if rng.random() < rewire:
                if noise == "drop":
                    continue
                book = by_book[recs[u].get("file_id")]
                hi = pos[u] if noise == "rewire-backward" else len(book)
                if hi == 0:
                    continue
                w = book[rng.randrange(hi)]
                if w != u:
                    new_edges.append((u, w))
            else:
                new_edges.append((u, v))
        d, _, _ = depth_and_mass(nodes, new_edges, with_mass=False)
        vec = np.array([d[u] for u in eligible])
        rhos.append(float(stats.spearmanr(base_vec, vec).statistic))
        pt = top_k(d, eligible, k)
        jacc.append(len(pt & base_top) / len(pt | base_top))
        if r < save_reps:
            saved[f"depth_rep{r}"] = [d[n] for n in nodes]
        print(f"rep {r}: rho={rhos[-1]:.3f} jaccard(MA-Hard)={jacc[-1]:.3f}")
    summary["perturbation"] = {
        "reps": reps, "rewire_fraction": rewire, "noise": noise,
        "spearman_mean": float(np.mean(rhos)), "spearman_p2.5": float(np.percentile(rhos, 2.5)),
        "jaccard_ma_hard_mean": float(np.mean(jacc)), "jaccard_ma_hard_p2.5": float(np.percentile(jacc, 2.5)),
    }
    for kname, vals in saved.items():
        df[kname] = vals

    df.to_csv(output_dir / f"depth.{noise}.csv.gz", index=False)
    (output_dir / f"depth_summary.{noise}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
