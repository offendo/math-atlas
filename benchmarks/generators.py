"""Chat generation backends shared by the iterative baselines.

Three backends, same interface:
  * `OpenAIGenerator`    -- any OpenAI-compatible endpoint (vLLM server, OpenAI, ...)
  * `VLLMGenerator`      -- offline in-process vLLM
  * `ClaudeCLIGenerator` -- Claude models through the logged-in `claude` CLI

All take a list of conversations and return one completion per conversation,
in order, which is what a multi-round repair loop needs.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Sequence

logger = logging.getLogger("benchmarks.generators")

Conversation = list[dict[str, str]]


class BaseGenerator:
    def chat(self, conversations: Sequence[Conversation], temperature: float, seed: int | None = None) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - backend specific
        pass


class OpenAIGenerator(BaseGenerator):
    """OpenAI-compatible endpoint (vLLM `--api-server`, OpenAI, Anthropic-compatible proxies...)."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str = "EMPTY",
        api_style: str = "chat",
        max_tokens: int = 8192,
        top_p: float = 1.0,
        concurrency: int = 20,
        reasoning_effort: str | None = None,
    ):
        from openai import AsyncOpenAI

        self.model = model
        self.api_style = api_style
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.concurrency = concurrency
        self.reasoning_effort = reasoning_effort
        import os

        # An empty $OPENAI_BASE_URL (set on this machine) breaks the SDK default.
        default_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key) if base_url else AsyncOpenAI(base_url=default_url)
        # Some hosted models (e.g. GPT-5 family) reject sampling params; we drop
        # them permanently after the first rejection instead of failing the run.
        self._drop_sampling = False

    async def _complete(self, conversation: Conversation, temperature: float, seed: int | None) -> str:
        for attempt in range(2):
            kwargs: dict[str, Any] = {}
            if not self._drop_sampling:
                kwargs.update(temperature=temperature, top_p=self.top_p)
                if seed is not None:
                    kwargs["seed"] = seed
            if self.reasoning_effort:
                kwargs["reasoning_effort" if self.api_style == "chat" else "reasoning"] = (
                    self.reasoning_effort if self.api_style == "chat" else {"effort": self.reasoning_effort}
                )
            try:
                if self.api_style == "responses":
                    kwargs.pop("seed", None)
                    resp = await self.client.responses.create(
                        model=self.model, input=conversation, max_output_tokens=self.max_tokens, **kwargs
                    )
                    return resp.output_text or ""
                resp = await self.client.chat.completions.create(
                    model=self.model, messages=conversation, max_completion_tokens=self.max_tokens, **kwargs
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                msg = str(e)
                if attempt == 0 and any(p in msg for p in ("temperature", "top_p", "seed", "unsupported_value")):
                    logger.warning("Endpoint rejected sampling params; retrying without them (%s)", msg[:200])
                    self._drop_sampling = True
                    continue
                logger.warning("Generation failed: %s", msg[:300])
                return ""
        return ""

    def chat(self, conversations: Sequence[Conversation], temperature: float, seed: int | None = None) -> list[str]:
        from tqdm.asyncio import tqdm

        semaphore = asyncio.Semaphore(self.concurrency)

        async def bounded(conv):
            async with semaphore:
                return await self._complete(conv, temperature, seed)

        async def _run():
            return await tqdm.gather(*[bounded(c) for c in conversations], desc="Generating")

        return asyncio.run(_run())


class VLLMGenerator(BaseGenerator):
    """Offline vLLM. Loads the model once and reuses it across repair rounds."""

    def __init__(
        self,
        model: str,
        max_tokens: int = 8192,
        top_p: float = 1.0,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.90,
    ):
        from vllm import LLM  # type: ignore

        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        logger.info("Loaded vLLM model `%s`", model)

    def chat(self, conversations: Sequence[Conversation], temperature: float, seed: int | None = None) -> list[str]:
        from vllm import SamplingParams  # type: ignore

        params = SamplingParams(
            max_tokens=self.max_tokens, temperature=temperature, top_p=self.top_p, seed=seed, skip_special_tokens=False
        )
        outputs = self.llm.chat(list(conversations), params, use_tqdm=True)
        return [o.outputs[0].text for o in outputs]


class ClaudeCLIGenerator(BaseGenerator):
    """Claude models through the logged-in `claude` CLI, as a plain model.

    Each call is `claude -p` with every tool disabled (`--tools ""`), the system
    prompt replaced, no settings/MCP/session persistence, run from an empty scratch
    directory -- so the model sees only our conversation. The CLI takes a single
    prompt, so repair rounds are serialized into one transcript (previous answer +
    compiler feedback), which carries the same information as the chat history the
    OpenAI backend sends. The CLI exposes no sampling controls, so `temperature`
    and `seed` are ignored (recorded as such in the run config).
    """

    def __init__(
        self,
        model: str,
        claude_bin: str = "claude",
        concurrency: int = 16,
        timeout: int = 1200,
        max_retries: int = 3,
        workdir: str | None = None,
    ):
        import tempfile

        self.model = model
        self.claude_bin = claude_bin
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self.workdir = workdir or tempfile.mkdtemp(prefix="claude-cli-gen-")
        self.total_cost_usd = 0.0
        self.models_seen: set[str] = set()
        self.n_failed = 0                       # calls that returned "" after all retries
        self.max_limit_wait = 16 * 3600         # total seconds to wait out account limits per call

    @staticmethod
    def serialize(conversation: Conversation) -> tuple[str, str]:
        system = "\n\n".join(m["content"] for m in conversation if m["role"] == "system")
        turns = [m for m in conversation if m["role"] != "system"]
        prompt = turns[0]["content"] if turns else ""
        for msg in turns[1:]:
            if msg["role"] == "assistant":
                prompt += f"\n\n---\nYour previous answer was:\n\n{msg['content']}"
            else:
                prompt += f"\n\n---\n{msg['content']}"
        return system, prompt

    def _one(self, conversation: Conversation) -> str:
        import json
        import subprocess
        import time

        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import common

        system, prompt = self.serialize(conversation)
        cmd = [
            self.claude_bin, "-p", "--model", self.model, "--output-format", "json",
            "--tools", "", "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "",
        ]
        if system:
            cmd += ["--system-prompt", system]
        attempt, waited = 0, 0
        while attempt < self.max_retries:
            try:
                proc = subprocess.run(
                    cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout, cwd=self.workdir
                )
                blob = (proc.stdout or "") + (proc.stderr or "")
                payload = json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else {}
                errored = (not payload) or bool(payload.get("is_error"))
                if errored and common.is_claude_limit(str(payload.get("result", "")) if payload else blob):
                    # Account limit, not a model failure: wait for the reset, don't burn an attempt.
                    wait = common.seconds_until_reset(str(payload.get("result", "")) or blob)
                    if waited + wait > self.max_limit_wait:
                        raise RuntimeError(f"gave up after waiting {waited}s for Claude limits to reset")
                    logger.warning("Claude limit hit; sleeping %ds until reset", wait)
                    time.sleep(wait)
                    waited += wait
                    continue
                if not payload:
                    raise RuntimeError(f"unparseable CLI output (rc={proc.returncode}): {blob[-300:]}")
                if payload.get("is_error"):
                    raise RuntimeError(f"{payload.get('subtype')}: {str(payload.get('result'))[:200]}")
                self.total_cost_usd += float(payload.get("total_cost_usd") or 0.0)
                self.models_seen.update((payload.get("modelUsage") or {}).keys())
                return payload.get("result") or ""
            except Exception as e:  # timeouts, malformed output, API errors
                attempt += 1
                msg = re.sub(r"Command '\[.*?\]'", "claude command", str(e), flags=re.DOTALL)
                logger.warning("claude CLI call failed (attempt %d/%d): %s", attempt, self.max_retries, msg[:300])
                time.sleep(20 * attempt)
        self.n_failed += 1
        return ""

    def chat(self, conversations: Sequence[Conversation], temperature: float, seed: int | None = None) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        from tqdm import tqdm

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(tqdm(pool.map(self._one, conversations), total=len(conversations), desc="Generating (claude)"))


def build_generator(
    model: str,
    model_url: str | None,
    api_key: str,
    api_style: str,
    max_tokens: int,
    top_p: float,
    concurrency: int,
    reasoning_effort: str | None,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    backend: str = "auto",
) -> BaseGenerator:
    if backend == "claude-cli":
        return ClaudeCLIGenerator(model=model, concurrency=concurrency)
    if model_url:
        return OpenAIGenerator(
            model=model,
            base_url=model_url,
            api_key=api_key,
            api_style=api_style,
            max_tokens=max_tokens,
            top_p=top_p,
            concurrency=concurrency,
            reasoning_effort=reasoning_effort,
        )
    return VLLMGenerator(
        model=model,
        max_tokens=max_tokens,
        top_p=top_p,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
