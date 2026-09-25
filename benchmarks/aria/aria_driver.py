#!/usr/bin/env python3
"""Run the Aria autoformalizer over a JSONL of items, resumably.

This runs *inside Aria's own venv* (it imports Aria's `src` package), so it
only uses the standard library plus what Aria already depends on. It is driven
by `run_aria.py`, which handles item selection, verification and judging.

Aria's stock `run_batch` launches every item at once and only writes at the
end; here items are bounded by a semaphore, each has a wall-clock timeout, and
every finished item is appended to `--partial` so a crash loses nothing.

Usage limits: behind the Claude Code proxy a limited account surfaces as an
HTTP 502 carrying the CLI's "hit your limit · resets 2:40am (Tz)" text. Aria
does not retry that, so every in-flight item would fail and be checkpointed as
a model failure. Instead, both of Aria's LLM entry points are wrapped: a limit
error closes a shared gate until the reset time, every call waits on the gate,
paused time does not count against the item timeout, and an item that still
fails on a limit is not checkpointed (a `--resume` rerun picks it up).

Input lines: {"uuid": ..., "text": ...}
Output lines: {"uuid", "code", "aria_success", "aria_error",
               "generated_context", "stats", "duration_s", "error"}
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from zoneinfo import ZoneInfo

# Same patterns as benchmarks/common.py (not importable from Aria's venv).
LIMIT_PATTERN = re.compile(r"(session|usage|weekly|rate) limit|hit your limit|limit reached", re.IGNORECASE)
RESET_PATTERN = re.compile(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)\s*\(([^)]+)\)", re.IGNORECASE)


def log(msg: str) -> None:
    print(f"[aria] {msg}", file=sys.stderr, flush=True)


def is_limit(text: str) -> bool:
    return bool(text) and bool(LIMIT_PATTERN.search(text))


def seconds_until_reset(text: str, default: int = 600, slack: int = 90) -> int:
    m = RESET_PATTERN.search(text or "")
    if not m:
        return default
    hour, minute, ampm, tz = int(m.group(1)) % 12, int(m.group(2) or 0), m.group(3).lower(), m.group(4)
    hour += 12 if ampm == "pm" else 0
    try:
        zone = ZoneInfo(tz.strip())
    except Exception:
        return default
    now = dt.datetime.now(zone)
    reset = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset <= now:
        reset += dt.timedelta(days=1)
    return int((reset - now).total_seconds()) + slack


class LimitGate:
    """Shared pause: closed from a limit error until the advertised reset."""

    def __init__(self) -> None:
        self._open = asyncio.Event()
        self._open.set()
        self._paused_total = 0.0
        self._paused_since: float | None = None

    def paused_seconds(self) -> float:
        extra = time.time() - self._paused_since if self._paused_since is not None else 0.0
        return self._paused_total + extra

    async def wait(self) -> None:
        await self._open.wait()

    async def trip(self, text: str) -> None:
        if not self._open.is_set():  # someone else is already sleeping it off
            await self._open.wait()
            return
        wait = seconds_until_reset(text)
        self._open.clear()
        self._paused_since = time.time()
        log(f"LIMIT hit, pausing all workers {wait}s (until ~{time.strftime('%H:%M', time.localtime(time.time() + wait))}): "
            f"{text.strip()[:160]}")
        try:
            await asyncio.sleep(wait)
        finally:
            self._paused_total += time.time() - self._paused_since
            self._paused_since = None
            self._open.set()
            log("LIMIT pause over, resuming")


def install_limit_guard(tools, gate: LimitGate) -> None:
    """Route Aria's LLM calls through `gate`; retry transient errors a few times."""
    from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

    async def llm(msgs, config, max_retries: int = 5):
        # Replaces tools.llm, whose own loop swallows the final failure and
        # returns None (which then crashes the caller on `.choices`).
        attempt = 0
        while True:
            await gate.wait()
            try:
                client = AsyncOpenAI(base_url=config["base_url"], api_key=config["api_key"], timeout=600.0)
                return await client.chat.completions.create(
                    model=config["model"],
                    messages=msgs,
                    stream=False,
                    temperature=config.get("temperature", 1),
                    max_completion_tokens=config.get("max_completion_tokens", 20000),
                )
            except Exception as e:  # noqa: BLE001
                if is_limit(str(e)):
                    await gate.trip(str(e))
                    continue
                transient = isinstance(e, (APIConnectionError, APITimeoutError)) or (
                    isinstance(e, APIStatusError) and e.status_code in (429, 500, 502, 503, 504)
                )
                attempt += 1
                if not transient or attempt >= max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)

    orig_strict = tools.llm_with_strict_format

    async def llm_with_strict_format(*a, **kw):
        while True:
            await gate.wait()
            try:
                return await orig_strict(*a, **kw)
            except Exception as e:  # noqa: BLE001
                if is_limit(str(e)):
                    await gate.trip(str(e))
                    continue
                raise

    # Callers reach both as module attributes (`tools.x(...)` or a bare
    # global lookup inside tools.py), so rebinding the module globals suffices.
    tools.llm = llm
    tools.llm_with_strict_format = llm_with_strict_format


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
    aria_log = open(Path(args.log).resolve(), "a", buffering=1)

    # Aria reads `configs/*.yaml` relative to the cwd at import time.
    os.chdir(aria_root)
    sys.path.insert(0, str(aria_root))
    with contextlib.redirect_stdout(aria_log):
        from src import tools
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

    gate = LimitGate()
    install_limit_guard(tools, gate)

    done = load_partial(partial)
    todo = [it for it in items if it["uuid"] not in done]
    log(f"{len(items)} items, {len(done)} already done, {len(todo)} to run "
        f"(concurrency={args.concurrency}, rag={args.rag}, model={run_batch.CONFIG['llm']['main_model']['model']})")

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    finished = skipped = 0

    async def run_with_budget(item: dict) -> dict:
        """`args.timeout` seconds of *unpaused* wall clock for one item."""
        t0, p0 = time.time(), gate.paused_seconds()
        task = asyncio.ensure_future(run_batch.run_formalization_pipeline(informal_statement=item["text"]))
        try:
            while True:
                remaining = args.timeout - ((time.time() - t0) - (gate.paused_seconds() - p0))
                if remaining <= 0:
                    raise asyncio.TimeoutError
                done_, _ = await asyncio.wait({task}, timeout=min(remaining, 30))
                if done_:
                    return task.result()
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task

    async def one(item: dict) -> None:
        nonlocal finished, skipped
        async with sem:
            await gate.wait()
            t0, p0 = time.time(), gate.paused_seconds()
            rec = {"uuid": item["uuid"], "code": "", "aria_success": False, "aria_error": None,
                   "generated_context": "", "stats": {}, "error": None}
            try:
                res = await run_with_budget(item)
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
            rec["paused_s"] = round(gate.paused_seconds() - p0, 1)
            async with lock:
                if is_limit(rec["error"] or "") or is_limit(rec["aria_error"] or ""):
                    skipped += 1
                    log(f"NOT checkpointing {item['uuid']} (failed on a usage limit; rerun with --resume)")
                    return
                with open(partial, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                finished += 1
                log(f"{finished}/{len(todo)} {item['uuid']} success={rec['aria_success']} "
                    f"err={bool(rec['error'])} t={rec['duration_s']}s paused={rec['paused_s']}s")

    # Aria prints very verbosely; keep that out of the progress stream.
    with contextlib.redirect_stdout(aria_log):
        await asyncio.gather(*(one(it) for it in todo))
    if skipped:
        log(f"{skipped} items hit a usage limit and were not checkpointed; rerun with --resume")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aria-root", required=True, help="Aria-autoformalizer project dir (holds configs/ and src/).")
    p.add_argument("--input", required=True)
    p.add_argument("--partial", required=True, help="Append-only JSONL checkpoint.")
    p.add_argument("--log", required=True, help="Where Aria's own stdout goes.")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--timeout", type=float, default=3600, help="Per-item unpaused wall clock, seconds.")
    p.add_argument("--model", default=None, help="Override the LLM for both Aria roles.")
    p.add_argument("--base-url", default=None, help="Override the LLM base URL for both roles.")
    p.add_argument("--verify-url", default=None, help="Sets ARIA_VERIFY_URL.")
    p.add_argument("--rag", action=argparse.BooleanOptionalAction, default=False,
                   help="LeanSearch grounding (needs configs/leansearch.yaml endpoint).")
    asyncio.run(main(p.parse_args()))
