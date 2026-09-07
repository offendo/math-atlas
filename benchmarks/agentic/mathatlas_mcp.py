#!/usr/bin/env python3
"""Benchmark-facing MathAtlas MCP server.

Thin wrapper over the read-only data layer in the formalization project
(`atlas.mcp.mathatlas.data`), exposing the same six tools with one change: the
**mathlib grounding of the item currently under test is withheld**.

Why: `get_item` normally returns `mathlib_suggestion` -- the Mathlib declaration
the dataset already grounded that concept to. Handing that to an agent that is
being scored on formalizing the very same item is answer leakage; compile rate
would rise for a reason unrelated to the model, and A1 would stop being
comparable to the single-pass control. Dependencies keep their grounding, so
reuse of already-formalized prerequisites still works.

The item under test is named by `$MA_HARD_ITEM_UUID`, which the A1 runner bakes
into a per-item MCP config. With no such variable this behaves exactly like the
upstream server.

Run it with the formalization project's environment:

    uv run --project ~/src/mathatlas-formalization python benchmarks/agentic/mathatlas_mcp.py
"""

from __future__ import annotations

import os
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from atlas.mcp.mathatlas import data

# Set per item by the A1 runner. Empty -> redact nothing.
ITEM_UNDER_TEST = os.environ.get("MA_HARD_ITEM_UUID", "").strip()
# Only `get_item` carries mathlib grounding; the other views do not.
REDACTED_FIELDS = ("mathlib_suggestion", "has_mathlib_link")

mcp = FastMCP("mathatlas")


def _redact(item: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Strip mathlib grounding from the item under test, and only that item."""
    if not item or not ITEM_UNDER_TEST:
        return item
    if item.get("entity_id") != ITEM_UNDER_TEST:
        return item
    return {k: v for k, v in item.items() if k not in REDACTED_FIELDS}


@mcp.tool()
def mathatlas_get_item(entity_id: str) -> Optional[dict[str, Any]]:
    """Item view: name, item_type, informal statement, textbook, source location,
    proof link, and any mathlib grounding suggestion. Null if the id is unknown."""
    return _redact(data.get_item(entity_id))


@mcp.tool()
def mathatlas_get_context(
    entity_id: str, max_chars: int = data.CONTEXT_MAX_CHARS
) -> Optional[dict[str, Any]]:
    """Priming context: the textbook prose immediately preceding the item (sliced
    from its .mmd source, trimmed to the enclosing section) plus the informal
    statements of its direct dependencies. `max_chars` bounds the prose look-back.
    Null if the id is unknown."""
    return data.get_context(entity_id, max_chars=max_chars)


@mcp.tool()
def mathatlas_get_dependencies(entity_id: str) -> Optional[list[dict[str, Any]]]:
    """Resolved dependency list (referenced concepts). Each entry has `reference`
    (concept name), `entity_id`, and resolved identifier/item_type/textbook.
    Null if the item id is unknown."""
    return data.get_dependencies(entity_id)


@mcp.tool()
def mathatlas_get_proofs(entity_id: str) -> Optional[list[dict[str, Any]]]:
    """The informal proof(s) of a theorem, in textbook order; each is an item view
    whose `statement` is the proof text. A theorem may have 0 proofs or several.
    `[]` = theorem present but unproved in the source; null = unknown id."""
    proofs = data.get_proofs(entity_id)
    return None if proofs is None else [_redact(p) for p in proofs]


@mcp.tool()
def mathatlas_list_items(
    textbook: Optional[str] = None,
    item_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List items (id + light metadata), filtered by textbook and/or item_type."""
    return data.list_items(textbook=textbook, item_type=item_type, limit=limit, offset=offset)


@mcp.tool()
def mathatlas_stats() -> dict[str, Any]:
    """Dataset dashboard: total item count, per-type counts, and textbook count."""
    return data.stats()


if __name__ == "__main__":
    mcp.run()  # stdio transport
