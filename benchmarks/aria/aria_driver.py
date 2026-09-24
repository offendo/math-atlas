#!/usr/bin/env python3
"""Run the Aria autoformalizer over a JSONL of items, resumably.

This runs *inside Aria's own venv* (it imports Aria's `src` package), so it
only uses the standard library plus what Aria already depends on. It is driven
by `run_aria.py`, which handles item selection, verification and judging.

Aria's stock `run_batch` launches every item at once and only writes at the
end; here items are bounded by a semaphore, each has a wall-clock timeout, and
every finished item is appended to `--partial` so a crash loses nothing.

Input lines: {"uuid": ..., "text": ...}
Output lines: {"uuid", "code", "aria_success", "aria_error",
               "generated_context", "stats", "duration_s", "error"}
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def load_partial(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["uuid"])
    return done


async def main(args: argparse.Namespace) -> None:
    aria_root = Path(args.aria_root).resolve()
    partial = Path(args.partial).resolve()
    items = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    log = open(Path(args.log).resolve(), "a", buffering=1)

    # Aria reads `configs/*.yaml` relative to the cwd at import time.
    os.chdir(aria_root)
    sys.path.insert(0, str(aria_root))
    with contextlib.redirect_stdout(log):
        from src.Agents.Autoformalizer import nodes
        from src.Agents.Flow import run_batch

    # run_batch and nodes each load their own copy of the config; patch both.
    for cfg in (run_batch.CONFIG, nodes.CONFIG):
        cfg["enable_rag"] = args.rag
        for role in ("formalizer", "main_model"):
            if args.model:
                cfg["llm"][role]["model"] = args.model
            if args.base_url:
                cfg["llm"][role]["base_url"] = args.base_url
    if args.verify_url:
        os.environ["ARIA_VERIFY_URL"] = args.verify_url

    done = load_partial(partial)
    todo = [it for it in items if it["uuid"] not in done]
    print(f"[aria] {len(items)} items, {len(done)} already done, {len(todo)} to run "
          f"(concurrency={args.concurrency}, rag={args.rag}, model={run_batch.CONFIG['llm']['main_model']['model']})",
          file=sys.stderr, flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    finished = 0

    async def one(item: dict) -> None:
        nonlocal finished
        async with sem:
            t0 = time.time()
            rec = {"uuid": item["uuid"], "code": "", "aria_success": False, "aria_error": None,
                   "generated_context": "", "stats": {}, "error": None}
            try:
                res = await asyncio.wait_for(
                    run_batch.run_formalization_pipeline(informal_statement=item["text"]),
                    timeout=args.timeout,
                )
                rec["code"] = res.get("final_formal_statement") or ""
                rec["aria_success"] = bool(res.get("final_compilation_success"))
                rec["aria_error"] = res.get("final_compilation_error")
                rec["generated_context"] = res.get("newly_defined_context") or ""
                rec["stats"] = res.get("stats", {})
            except asyncio.TimeoutError:
                rec["error"] = f"timeout after {args.timeout}s"
            except Exception as e:  # noqa: BLE001 - record and move on
                rec["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
            rec["duration_s"] = round(time.time() - t0, 1)
            async with lock:
                with open(partial, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                finished += 1
                print(f"[aria] {finished}/{len(todo)} {item['uuid']} success={rec['aria_success']} "
                      f"err={bool(rec['error'])} t={rec['duration_s']}s", file=sys.stderr, flush=True)

    # Aria prints very verbosely; keep that out of the progress stream.
    with contextlib.redirect_stdout(log):
        await asyncio.gather(*(one(it) for it in todo))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aria-root", required=True, help="Aria-autoformalizer project dir (holds configs/ and src/).")
    p.add_argument("--input", required=True)
    p.add_argument("--partial", required=True, help="Append-only JSONL checkpoint.")
    p.add_argument("--log", required=True, help="Where Aria's own stdout goes.")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--timeout", type=float, default=3600, help="Per-item wall clock, seconds.")
    p.add_argument("--model", default=None, help="Override the LLM for both Aria roles.")
    p.add_argument("--base-url", default=None, help="Override the LLM base URL for both roles.")
    p.add_argument("--verify-url", default=None, help="Sets ARIA_VERIFY_URL.")
    p.add_argument("--rag", action=argparse.BooleanOptionalAction, default=False,
                   help="LeanSearch grounding (needs configs/leansearch.yaml endpoint).")
    asyncio.run(main(p.parse_args()))
