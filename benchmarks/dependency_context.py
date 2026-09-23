"""Dependency-graph context for single-pass / iterative baselines (E2a).

Builds a prompt block describing an item's *direct* prerequisites, read from the
MathAtlas dependency graph -- the same NDJSON the agentic MathAtlas MCP serves
(`~/src/mathatlas-formalization/data/MathAtlas.json`), so the single-pass arms and
the A1 agent see the same graph.

Modes (one per ablation arm):

    none      no block (control)
    informal  informal text of each direct dependency
    mathlib   only the Mathlib declarations the dataset grounded those
              dependencies to (dependencies without a grounding are omitted)
    both      informal text + Mathlib grounding per dependency
    random    matched control for `both`: the same number of *random* definitions
              from the same textbook (excluding the item and its real
              dependencies), rendered identically. Separates "the right
              prerequisites" from "more text in the prompt", which matters because
              plain local context is known to hurt on MathAtlas.

The Mathlib grounding of the item under test is never shown -- only its
dependencies' groundings -- mirroring the redaction in agentic/mathatlas_mcp.py.
Stricter than the MCP: a dependency whose grounding *equals* the item's own is
shown without it (4/698 MA-Hard items), since that would hand over the answer.

Edges follow the data layer in atlas.mcp.mathatlas.data._iter_edges:
`object_links` is a list of candidate lists, `entity_links` a flat list, and
self-edges are dropped.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger("benchmarks.dependency_context")

MODES = ("none", "informal", "mathlib", "both", "random")
DEFAULT_DATA = Path(
    os.environ.get("MATHATLAS_DATA", "~/src/mathatlas-formalization/data/MathAtlas.json")
).expanduser()

HEADER = (
    "## Background: prerequisites of this item\n"
    "The item below comes from a textbook. It relies on the following concepts, taken from "
    "the same source. Use them to pick the right definitions and hypotheses. Where a Mathlib "
    "declaration is listed, prefer it over redefining the concept.\n"
)
MATHLIB_HEADER = (
    "## Background: Mathlib declarations for this item's prerequisites\n"
    "The item below relies on concepts that already exist in Mathlib under these names. "
    "Prefer them over redefining the concepts.\n"
)


def iter_edges(rec: dict[str, Any]) -> Iterator[tuple[Optional[str], str]]:
    """(reference_name, target_uuid) per dependency edge; see module docstring."""
    source = rec.get("uuid")
    obj_refs = rec.get("object_references") or []
    for i, candidates in enumerate(rec.get("object_links") or []):
        ref = obj_refs[i] if i < len(obj_refs) else None
        for target in candidates or []:
            if target and target != source:
                yield ref, target
    ent_refs = rec.get("entity_references") or []
    for i, target in enumerate(rec.get("entity_links") or []):
        ref = ent_refs[i] if i < len(ent_refs) else None
        if target and target != source:
            yield ref, target


def mathlib_suggestion(rec: dict[str, Any]) -> Optional[dict[str, Any]]:
    """First grounded Mathlib declaration of a record (same logic as the MCP data layer)."""
    for link in rec.get("mathlib_links") or []:
        gm = link.get("grounded_match") if isinstance(link, dict) else None
        if not gm:
            continue
        name, module = gm.get("name"), gm.get("module_name")
        return {
            "decl": ".".join(name) if isinstance(name, list) else name,
            "module": ".".join(module) if isinstance(module, list) else module,
        }
    return None


class DependencyContext:
    def __init__(
        self,
        mode: str = "none",
        data_path: Path | str = DEFAULT_DATA,
        max_deps: int = 12,
        dep_chars: int = 800,
        seed: int = 1337,
    ):
        if mode not in MODES:
            raise ValueError(f"dependency context mode must be one of {MODES}, got `{mode}`")
        self.mode, self.max_deps, self.dep_chars, self.seed = mode, max_deps, dep_chars, seed
        self.records: dict[str, dict[str, Any]] = {}
        self.defs_by_book: dict[str, list[str]] = {}
        if mode == "none":
            return
        path = Path(data_path).expanduser()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("uuid"):
                    self.records[rec["uuid"]] = rec
        for uid, rec in self.records.items():
            if rec.get("type") == "definition":
                self.defs_by_book.setdefault(rec.get("file_id"), []).append(uid)
        logger.info("Loaded %d MathAtlas records from %s (mode=%s)", len(self.records), path, mode)

    # ------------------------------------------------------------------ #
    def dependencies(self, uuid: str) -> list[tuple[Optional[str], str]]:
        """Direct, in-dataset, de-duplicated dependencies, in reference order, capped."""
        rec = self.records.get(uuid)
        if rec is None:
            return []
        out, seen = [], set()
        for ref, target in iter_edges(rec):
            if target in seen or target not in self.records:
                continue
            seen.add(target)
            out.append((ref, target))
        return out[: self.max_deps]

    def random_matched(self, uuid: str, k: int) -> list[tuple[Optional[str], str]]:
        """k random definitions from the item's textbook, excluding it and its real deps."""
        rec = self.records.get(uuid)
        if rec is None or k == 0:
            return []
        exclude = {uuid} | {t for _, t in iter_edges(rec)}
        pool = [u for u in self.defs_by_book.get(rec.get("file_id"), []) if u not in exclude]
        if len(pool) < k:  # tiny book: top up from all definitions
            have = set(pool)
            pool = pool + [u for us in self.defs_by_book.values() for u in us if u not in exclude and u not in have]
        rng = random.Random(f"{self.seed}:{uuid}")
        picks = rng.sample(pool, min(k, len(pool)))
        return [((self.records[u].get("names") or [None])[0], u) for u in picks]

    def _truncate(self, text: str) -> str:
        text = (text or "").strip()
        return text if len(text) <= self.dep_chars else text[: self.dep_chars].rstrip() + " …"

    def _grounding(self, target: str, own: Optional[str]) -> Optional[dict[str, Any]]:
        ml = mathlib_suggestion(self.records[target])
        if not ml or not ml.get("decl") or ml["decl"] == own:
            return None
        return ml

    def _render_full(self, deps: list[tuple[Optional[str], str]], own: Optional[str] = None) -> str:
        parts = [HEADER.rstrip()]
        for i, (ref, target) in enumerate(deps, 1):
            dep = self.records[target]
            label = dep.get("identifier") or ", ".join(dep.get("names") or []) or "unlabelled"
            title = f'### {i}. "{ref}" ({label}; {dep.get("type")})' if ref else f"### {i}. {label} ({dep.get('type')})"
            body = [title, self._truncate(dep.get("text", ""))]
            ml = self._grounding(target, own)
            if ml:
                body.append(f"Mathlib: `{ml['decl']}`" + (f" (in `{ml['module']}`)" if ml.get("module") else ""))
            parts.append("\n".join(body))
        return "\n\n".join(parts)

    def _render_mathlib(self, deps: list[tuple[Optional[str], str]], own: Optional[str] = None) -> str:
        lines = []
        for ref, target in deps:
            ml = self._grounding(target, own)
            if ml:
                name = ref or ", ".join(self.records[target].get("names") or []) or "concept"
                lines.append(f"- {name}: `{ml['decl']}`" + (f" (in `{ml['module']}`)" if ml.get("module") else ""))
        return MATHLIB_HEADER + "\n".join(lines) if lines else ""

    def _render_informal(self, deps: list[tuple[Optional[str], str]]) -> str:
        parts = [HEADER.replace(" Where a Mathlib declaration is listed, prefer it over redefining the concept.", "").rstrip()]
        for i, (ref, target) in enumerate(deps, 1):
            dep = self.records[target]
            label = dep.get("identifier") or ", ".join(dep.get("names") or []) or "unlabelled"
            title = f'### {i}. "{ref}" ({label}; {dep.get("type")})' if ref else f"### {i}. {label} ({dep.get('type')})"
            parts.append(title + "\n" + self._truncate(dep.get("text", "")))
        return "\n\n".join(parts)

    def block(self, uuid: str) -> tuple[str, dict[str, Any]]:
        """Prompt block for an item plus metadata recorded in the output rows."""
        if self.mode == "none":
            return "", {"dep_mode": "none", "dep_ids": [], "dep_block_chars": 0}
        deps = self.dependencies(uuid)
        own_ml = mathlib_suggestion(self.records[uuid]) if uuid in self.records else None
        own = own_ml.get("decl") if own_ml else None
        if self.mode == "random":
            deps = self.random_matched(uuid, len(deps))
            text = self._render_full(deps, own) if deps else ""
        elif self.mode == "both":
            text = self._render_full(deps, own) if deps else ""
        elif self.mode == "informal":
            text = self._render_informal(deps) if deps else ""
        else:  # mathlib
            text = self._render_mathlib(deps, own) if deps else ""
        return text, {"dep_mode": self.mode, "dep_ids": [t for _, t in deps], "dep_block_chars": len(text)}
