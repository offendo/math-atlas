"""Shared utilities for MA-Hard baselines (iterative + agentic).

Provides dataset selection, Lean code extraction, `blv` compile verification,
and CriticLean-style alignment judging so that every baseline reports numbers
computed the exact same way as the single-pass runs.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

logger = logging.getLogger("benchmarks.common")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEADER = ("import Mathlib", "import Aesop")
DEFINITION_TYPES = {"definition"}

# Mirrors scripts/run_alignment_benchmark.py so judgements are comparable.
ALIGNMENT_SCHEMA = {
    "name": "reasoning_alignment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "The explanation or justification for the result."},
            "result": {
                "type": "string",
                "description": "Whether the object is aligned or misaligned.",
                "enum": ["aligned", "misaligned"],
            },
        },
        "required": ["reasoning", "result"],
        "additionalProperties": False,
    },
}


# --------------------------------------------------------------------------- #
# Dataset selection
# --------------------------------------------------------------------------- #
def parse_filters(filters: Sequence[str] | None) -> dict[str, str]:
    """Parse repeatable `--filter col=value` options into a dict."""
    out: dict[str, str] = {}
    for f in filters or []:
        if "=" not in f:
            raise ValueError(f"--filter must look like `column=value`, got `{f}`")
        col, _, val = f.partition("=")
        out[col.strip()] = val.strip()
    return out


def load_items(
    dataset: str,
    split: str = "train",
    item_types: Sequence[str] | None = None,
    filters: Sequence[str] | None = None,
    subset_file: Path | None = None,
    n_examples: int | None = None,
    seed: int = 1337,
) -> pd.DataFrame:
    """Load MA-Hard (or any MathAtlas slice) as a DataFrame.

    MA-Hard can be addressed three ways, all composable:
      * a dedicated dataset/split       (`--dataset ... --split hard`)
      * a column filter                 (`--filter difficulty=hard`)
      * an explicit uuid list           (`--subset-file ma_hard_uuids.json`)
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    df = ds.to_pandas()
    logger.info("Loaded `%s` split=`%s` (%d rows)", dataset, split, len(df))

    if item_types and "all" not in item_types:
        df = df[df["type"].isin(list(item_types))]
        logger.info("Filtered to types=%s (%d rows)", list(item_types), len(df))

    for col, val in parse_filters(filters).items():
        if col not in df.columns:
            raise KeyError(f"--filter column `{col}` not in dataset columns: {sorted(df.columns)}")
        df = df[df[col].astype(str) == val]
        logger.info("Filtered %s==%s (%d rows)", col, val, len(df))

    if subset_file is not None:
        uuids = load_uuid_list(subset_file)
        df = df[df["uuid"].isin(uuids)]
        logger.info("Filtered to %d uuids from `%s` (%d rows)", len(uuids), subset_file, len(df))

    if n_examples is not None and n_examples < len(df):
        df = df.sample(n_examples, random_state=seed)
        logger.info("Subsampled to %d rows (seed=%d)", len(df), seed)

    if len(df) == 0:
        raise ValueError("Selection produced 0 items; check --split/--filter/--subset-file.")
    return df.reset_index(drop=True)


def load_uuid_list(path: Path) -> set[str]:
    """Read a uuid subset file: JSON list, JSONL/records with a `uuid` key, or plain text."""
    text = Path(path).read_text().strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return {x["uuid"] if isinstance(x, dict) else str(x) for x in obj}
        if isinstance(obj, dict):
            return set(map(str, obj.get("uuids", obj.keys())))
    except json.JSONDecodeError:
        pass
    return {line.strip() for line in text.splitlines() if line.strip()}


# --------------------------------------------------------------------------- #
# Lean extraction
# --------------------------------------------------------------------------- #
THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
HARMONY_PATTERN = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)", flags=re.DOTALL)
FENCE_PATTERN = re.compile(r"```(?:lean4?|Lean4?)?\s*\n(.*?)```", flags=re.DOTALL)


def extract_lean(output: str) -> str:
    """Pull the Lean snippet out of a model response.

    Prefers the last fenced block (models often show a broken draft first), and
    falls back to the raw text so that a malformed response still gets compiled
    and counted as a failure rather than silently dropped.
    """
    if not output:
        return ""
    text = output
    harmony = HARMONY_PATTERN.findall(text)
    if harmony:
        text = harmony[-1]
    text = THINK_PATTERN.sub("", text)
    if "</think>" in text:  # unterminated thinking block
        text = text.split("</think>")[-1]

    blocks = [b.strip() for b in FENCE_PATTERN.findall(text) if b.strip()]
    if blocks:
        return blocks[-1]
    return text.strip()


def strip_imports(code: str) -> str:
    """Drop import/`set_option` header lines; `blv` re-adds a forced header."""
    return "\n".join(l for l in code.splitlines() if not l.strip().startswith("import ")).strip()


# --------------------------------------------------------------------------- #
# Compile verification (blv)
# --------------------------------------------------------------------------- #
def resolve_blv_verify():
    """Return the `blv` verification entry point.

    The API was renamed between releases (`blv.verify` -> `blv.verify_theorems`)
    and both are in use on this machine, so resolve it at call time.
    """
    import blv

    for name in ("verify_theorems", "verify"):
        fn = getattr(blv, name, None)
        if callable(fn) and not inspect.ismodule(fn):
            return fn
    raise RuntimeError("Could not find a callable verification function in `blv`.")


def verify_batch(
    codes: Sequence[str],
    timeout: int = 60,
    force_header: tuple[str, ...] | None = DEFAULT_HEADER,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    flush_db_after: bool = True,
) -> list[dict[str, Any]]:
    """Verify Lean snippets with `blv`, preserving input order.

    Empty snippets are short-circuited to a failure rather than sent to the REPL.
    """
    fn = resolve_blv_verify()

    codes = list(codes)
    todo = [(i, c) for i, c in enumerate(codes) if c and c.strip()]
    results: list[dict[str, Any]] = [
        {"verified": False, "errors": [{"data": "empty generation"}], "response": None} for _ in codes
    ]
    if not todo:
        return results

    kwargs: dict[str, Any] = {
        "timeout": timeout,
        "force_header": force_header,
        "redis_host": redis_host,
        "redis_port": redis_port,
        "redis_db": redis_db,
    }
    if "flush_db_after" in inspect.signature(fn).parameters:
        kwargs["flush_db_after"] = flush_db_after
    outputs = fn([c for _, c in todo], **kwargs)
    for (idx, _), out in zip(todo, outputs):
        results[idx] = out
    return results


def error_text(compiler_output: dict[str, Any] | None, max_chars: int = 4000) -> str:
    """Render `blv` errors as the compiler feedback string shown to the model."""
    if not compiler_output:
        return "Unknown verification failure."
    errors = compiler_output.get("errors") or []
    parts: list[str] = []
    for err in errors:
        if isinstance(err, dict):
            data = err.get("data", "")
            pos = err.get("pos") or {}
            line, col = pos.get("line"), pos.get("column")
            loc = f"line {line}, column {col}: " if line is not None else ""
            parts.append(f"error: {loc}{data}")
        else:
            parts.append(f"error: {err}")
    if not parts:
        parts = ["error: verification failed with no message"]
    out = "\n".join(parts)
    return out[:max_chars] + ("\n...[truncated]" if len(out) > max_chars else "")


def sorries(compiler_output: dict[str, Any] | None) -> int:
    """Count `sorry`s the REPL reported (statement tasks expect exactly one)."""
    resp = (compiler_output or {}).get("response") or {}
    return len(resp.get("sorries", []) or [])


# --------------------------------------------------------------------------- #
# Alignment judging (CriticLean-32B or any OpenAI-compatible judge)
# --------------------------------------------------------------------------- #
def make_judge_prompt(informal: str, formal: str, template: str) -> list[dict[str, str]]:
    """Build the judge conversation; supports both `{informal}/{formal}` and system-prompt templates."""
    if "{formal}" in template:
        return [{"role": "user", "content": template.format(informal=informal, formal=formal)}]
    return [
        {"role": "system", "content": template},
        {"role": "user", "content": f"Informal:\n{informal}\n\nFormal:\n{formal}"},
    ]


def parse_judgement(raw: str) -> dict[str, Any]:
    """Parse judge output; mirrors scripts/run_alignment_benchmark.py:try_parse."""
    try:
        text, thinking = raw, None
        harmony = HARMONY_PATTERN.findall(text)  # gpt-oss without a reasoning parser
        if harmony:
            text = harmony[-1]
        if "</think>" in text:
            thinking, text = text.split("</think>", 1)
        if "<consistency>" in text:
            verdict = text.split("<consistency>")[1].split("</consistency>")[0].strip()
            return {"result": "aligned" if verdict == "Correct" else "misaligned", "reasoning": text, "error": None}
        if "```" in text:
            text = re.sub(r"```(?:json)?", "", text)
        text = text.strip()
        if not text.startswith("{") and "{" in text:
            text = text[text.index("{"): text.rindex("}") + 1]
        out = json.loads(text)
        result = str(out.get("result", "misaligned"))
        out["result"] = "aligned" if result in {"aligned", "Correct", "correct", "True", "true"} else "misaligned"
        out.setdefault("reasoning", "")
        out["error"] = None
        return out
    except Exception as e:  # unparseable judge output counts as misaligned
        return {"result": "misaligned", "reasoning": raw, "error": str(e)}


def default_openai_url() -> str:
    """$OPENAI_BASE_URL, unless unset *or empty* -- an empty value (set on this
    machine) makes the SDK post to "" and fail with a bare "Connection error"."""
    import os

    return os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"


def judge_alignment(
    pairs: Sequence[tuple[str, str]],
    model: str,
    model_url: str | None,
    prompt_file: Path,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    top_p: float = 1.0,
    concurrency: int = 20,
    api_key: str = "EMPTY",
    structured: bool = True,
) -> list[dict[str, Any]]:
    """Score (informal, formal) pairs with the CriticLean judge."""
    from openai import AsyncOpenAI
    from tqdm.asyncio import tqdm

    if not pairs:
        return []
    template = Path(prompt_file).read_text()
    prompts = [make_judge_prompt(i, f, template) for i, f in pairs]
    client = AsyncOpenAI(base_url=model_url, api_key=api_key) if model_url else AsyncOpenAI(base_url=default_openai_url())
    semaphore = asyncio.Semaphore(concurrency)

    kwargs: dict[str, Any] = {}
    if structured:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": ALIGNMENT_SCHEMA}

    sampling = {"temperature": temperature, "top_p": top_p}

    async def complete(prompt):
        async with semaphore:
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=prompt,
                        max_completion_tokens=max_tokens,
                        **sampling,
                        **kwargs,
                    )
                    return resp.choices[0].message.content or ""
                except Exception as e:
                    msg = str(e)
                    # Hosted reasoning models (gpt-5.x) reject sampling params: drop them for the run.
                    if attempt < 2 and any(p in msg for p in ("temperature", "top_p", "unsupported_value")):
                        logger.warning("Judge endpoint rejected sampling params; retrying without them")
                        sampling.clear()
                        continue
                    if attempt < 2 and any(p in msg.lower() for p in ("rate limit", "429", "timeout", "overloaded", "502", "503")):
                        await asyncio.sleep(10 * (attempt + 1))
                        continue
                    logger.warning("Judge call failed: %s", e)
                    return f"JUDGE_ERROR: {e}"
            return "JUDGE_ERROR: retries exhausted"

    async def _run():
        return await tqdm.gather(*[complete(p) for p in prompts], desc="Judging")

    raw = asyncio.run(_run())
    return [parse_judgement(r) for r in raw]


def judge_and_attach(
    df: pd.DataFrame,
    judge_model: str,
    judge_model_url: str | None,
    judge_prompt_file: Path,
    judge_definition_prompt_file: Path,
    judge_max_tokens: int = 8192,
    judge_concurrency: int = 20,
    api_key: str = "EMPTY",
    structured: bool = True,
) -> None:
    """Judge compiling rows in-place; definitions and statements get their own prompt.

    Only compiling rows are judged (same convention as scripts/run_alignment_score.py);
    non-compiling rows stay `aligned=False` so `joint_rate` is over all items.
    """
    df["alignment_output"] = None
    df["aligned"] = False
    compiling = df[df["verified"].fillna(False).astype(bool)]
    if compiling.empty:
        logger.warning("Nothing compiled; skipping judge.")
        return

    for is_def, prompt_file in [(True, judge_definition_prompt_file), (False, judge_prompt_file)]:
        rows = compiling[compiling["type"].isin(DEFINITION_TYPES) == is_def]
        if rows.empty:
            continue
        logger.info("Judging %d %s with `%s`", len(rows), "definitions" if is_def else "statements", Path(prompt_file).name)
        judgements = judge_alignment(
            list(zip(rows["text"], rows["code"])),
            model=judge_model,
            model_url=judge_model_url,
            prompt_file=prompt_file,
            max_tokens=judge_max_tokens,
            concurrency=judge_concurrency,
            api_key=api_key,
            structured=structured,
        )
        for idx, judgement in zip(rows.index, judgements):
            df.at[idx, "alignment_output"] = judgement
            df.at[idx, "aligned"] = judgement["result"] == "aligned"


# --------------------------------------------------------------------------- #
# Metrics / IO
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    """Everything needed to reproduce a run; dumped alongside the metrics."""

    name: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"baseline": self.name, **self.extra}


def summarize(df: pd.DataFrame, max_rounds: int | None = None) -> dict[str, Any]:
    """Compute compile / alignment / joint rates.

    The headline number is `joint_rate` (compiles AND judged aligned): compile
    rate alone is trivially gamed by emitting degenerate statements.
    """
    n = len(df)
    verified = df["verified"].fillna(False).astype(bool)
    metrics: dict[str, Any] = {"n_items": int(n), "compile_rate": float(verified.mean()) if n else 0.0}
    # Without the judge there is no honest joint number, so report null rather than 0.
    if "aligned" in df.columns:
        aligned = df["aligned"].fillna(False).astype(bool)
        metrics["aligned_rate_of_compiling"] = float(aligned[verified].mean()) if verified.any() else 0.0
        metrics["joint_rate"] = float((verified & aligned).mean()) if n else 0.0
    else:
        metrics["aligned_rate_of_compiling"] = None
        metrics["joint_rate"] = None
    if "n_rounds" in df.columns:
        metrics["mean_rounds"] = float(df["n_rounds"].mean())
        # compile@k: fraction solved using at most k rounds -- the saturation curve.
        limit = max_rounds or int(df["n_rounds"].max())
        metrics["compile_at_k"] = {
            str(k): float(((df["n_rounds"] <= k) & verified).mean()) for k in range(1, limit + 1)
        }
    for col, key in [
        ("degenerate", "degenerate_rate"),
        ("cost_usd", "total_cost_usd"),
        ("duration_s", "total_duration_s"),
        ("num_turns", "mean_turns"),
    ]:
        if col in df.columns and df[col].notna().any():
            metrics[key] = float(df[col].sum() if key.startswith("total") else df[col].mean())
    return metrics


DEGENERATE_PATTERN = re.compile(r":\s*True\s*:?=|:=\s*trivial\b|:\s*True\b", flags=re.MULTILINE)


def is_degenerate(code: str) -> bool:
    """Cheap guard for compiling-but-vacuous statements (`theorem foo : True := trivial`).

    Reported as a rate, never used to silently drop rows -- iterative and agentic
    systems both drift toward satisfying the compiler rather than the math.
    """
    if not code or not code.strip():
        return True
    body = strip_imports(code)
    if len(body.split()) < 4:
        return True
    return bool(DEGENERATE_PATTERN.search(body))


def save_results(df: pd.DataFrame, metrics: dict[str, Any], output: Path, config: dict[str, Any]) -> None:
    """Write predictions + metrics, matching the single-pass output layout."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output, orient="records", indent=2)
    metrics_path = output.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"config": config, "metrics": metrics}, f, indent=2)
    logger.info("Saved results to `%s` and metrics to `%s`", output, metrics_path)


def print_metrics(metrics: dict[str, Any]) -> None:
    print("\n=== metrics ===")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"  k={kk}: {100 * vv:.2f}%")
        elif v is None:
            print(f"{k}: n/a (judge skipped)")
        elif isinstance(v, float) and "rate" in k:
            print(f"{k}: {100 * v:.2f}%")
        else:
            print(f"{k}: {v}")
    print()


def load_done_uuids(output: Path) -> set[str]:
    """uuids already present in an output file, for `--resume`."""
    if not Path(output).exists():
        return set()
    try:
        prev = pd.read_json(output)
        return set(prev["uuid"].astype(str))
    except Exception:
        return set()
