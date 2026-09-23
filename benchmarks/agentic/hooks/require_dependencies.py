#!/usr/bin/env python3
"""PreToolUse hook for the dependency-aware A1 arm (E2b `dep`).

Blocks Write/Edit of the item file until the session has called
`mcp__mathatlas__mathatlas_get_dependencies` -- in the main transcript or in a
subagent's (agents sometimes delegate the lookup). In the smoke test, the
prompt-only protocol was skipped outright by 1 of 3 agents, so the arm would
otherwise measure "told to" rather than "did".

Exit 2 blocks the tool call and shows stderr to the agent; exit 0 allows it.
Any failure inside the hook allows the call (never wedge a run on a hook bug).
"""

import json
import sys
from pathlib import Path

TOOL = "mcp__mathatlas__mathatlas_get_dependencies"


def called(transcript: Path) -> bool:
    files = [transcript] + sorted((transcript.parent / transcript.stem / "subagents").glob("*.jsonl"))
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if TOOL not in line:
                continue
            try:
                content = (json.loads(line).get("message") or {}).get("content")
            except json.JSONDecodeError:
                continue
            if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == TOOL for b in content
            ):
                return True
    return False


def main() -> int:
    try:
        event = json.load(sys.stdin)
        path = str((event.get("tool_input") or {}).get("file_path", ""))
        if "/MAHard/Item_" not in path:
            return 0
        if called(Path(event["transcript_path"])):
            return 0
        print(
            "Blocked by the benchmark protocol: before writing the formalization, call "
            f"`{TOOL}` (and `mcp__mathatlas__mathatlas_get_context`) on this item's MathAtlas id, "
            "read the prerequisites, and ground or formalize them first (step 1 of the protocol).",
            file=sys.stderr,
        )
        return 2
    except Exception:
        return 0


if __name__ == "__main__":
    sys.exit(main())
