"""Chat generation backends shared by the iterative baselines.

Two backends, same interface:
  * `OpenAIGenerator` -- any OpenAI-compatible endpoint (vLLM server, OpenAI, ...)
  * `VLLMGenerator`   -- offline in-process vLLM

Both take a list of conversations and return one completion per conversation,
in order, which is what a multi-round repair loop needs.
"""

from __future__ import annotations

import asyncio
import logging
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
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key) if base_url else AsyncOpenAI()
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
) -> BaseGenerator:
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
