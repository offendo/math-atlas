#!/usr/bin/env python3
"""A1 -- Claude Code agentic baseline on MA-Hard.

Each item gets its own module inside a real Lake project so the agent gets true
Lean feedback (via the lean-lsp MCP) and can import items it formalized earlier.
Final files are then scored the same way as every other baseline: `blv` compile
check + CriticLean alignment judging.

Example:

    python benchmarks/agentic/run_claude_code.py \
        --dataset offendo/math-atlas --filter split=hard \
        --project ~/src/mathatlas-formalization --model sonnet \
        --mcp-config ./math-atlas-mcp.json \
        --max-budget-usd 0.75 --timeout 900 \
        --judge-model m-a-p/CriticLeanGPT-Qwen3-32B-RL --judge-model-url http://localhost:8001/v1 \
        --output outputs/agentic/claude-code-sonnet.ma-hard.json
"""

from __future__ import annotations

import copy
import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402

app = typer.Typer(pretty_exceptions_show_locals=False)
logger = logging.getLogger("run_claude_code")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logger.setLevel(logging.INFO)

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
ITEMS_SUBDIR = "MAHard"
INDEX_FILE = "ma_hard_index.json"


# --------------------------------------------------------------------------- #
# Lake project management
# --------------------------------------------------------------------------- #
def run_cmd(cmd: list[str], cwd: Path, timeout: int = 3600) -> subprocess.CompletedProcess:
    logger.info("$ %s (cwd=%s)", " ".join(cmd), cwd)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def ensure_project(project: Path, lib_name: str | None, build: bool) -> tuple[Path, str]:
    """Return (project_dir, lib_name), creating a Mathlib-backed Lake project if needed."""
    project = project.expanduser().resolve()
    has_lakefile = (project / "lakefile.toml").exists() or (project / "lakefile.lean").exists()

    if not has_lakefile:
        if project.exists() and any(project.iterdir()):
            raise typer.BadParameter(f"`{project}` exists, is non-empty, and has no lakefile.")
        project.mkdir(parents=True, exist_ok=True)
        name = lib_name or "".join(p.capitalize() for p in re.split(r"[^A-Za-z0-9]+", project.name) if p)
        if shutil.which("lake") is None:
            raise RuntimeError("`lake` not found on PATH; install elan or pass an existing --project.")
        res = run_cmd(["lake", "init", name, "math"], cwd=project)
        if res.returncode != 0:
            raise RuntimeError(f"`lake init` failed:\n{res.stdout}\n{res.stderr}")
        logger.info("Initialized Lake project `%s` at %s", name, project)
        lib_name = name

    if lib_name is None:
        lib_name = detect_lib_name(project)

    if build:
        cache = run_cmd(["lake", "exe", "cache", "get"], cwd=project)
        if cache.returncode != 0:
            logger.warning("`lake exe cache get` failed (continuing):\n%s", cache.stderr[-800:])
        built = run_cmd(["lake", "build"], cwd=project)
        if built.returncode != 0:
            raise RuntimeError(f"`lake build` failed; fix the project before benchmarking:\n{built.stderr[-2000:]}")
        logger.info("Project builds cleanly.")

    (project / lib_name / ITEMS_SUBDIR).mkdir(parents=True, exist_ok=True)
    return project, lib_name


def detect_lib_name(project: Path) -> str:
    """Infer the library module name from the lakefile, falling back to the layout.

    The `[[lean_lib]]` name (capitalized module) is what we need, not the
    package `name` at the top of lakefile.toml, which is usually lowercased.
    """
    toml = project / "lakefile.toml"
    if toml.exists():
        text = toml.read_text()
        m = re.search(r"\[\[lean_lib\]\](?:.|\n)*?name\s*=\s*\"([^\"]+)\"", text)
        if m:
            return m.group(1)
    lean = project / "lakefile.lean"
    if lean.exists():
        m = re.search(r"lean_lib\s+«?([A-Za-z0-9_]+)»?", lean.read_text())
        if m:
            return m.group(1)
    for path in project.glob("*.lean"):
        if (project / path.stem).is_dir():
            return path.stem
    raise RuntimeError(f"Could not infer library name in `{project}`; pass --lib-name.")


def read_partial(path: Path) -> list[dict[str, Any]]:
    """Read the append-as-you-go checkpoint, skipping any half-written last line."""
    if not Path(path).exists():
        return []
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed checkpoint line in %s", path)
    return records


def load_partial_uuids(path: Path) -> set[str]:
    return {str(r["uuid"]) for r in read_partial(path) if "uuid" in r}


def module_name_for(uuid: str) -> str:
    """Lean module component for an item; uuids can start with a digit, so prefix."""
    return "Item_" + re.sub(r"[^A-Za-z0-9]", "_", str(uuid))


def write_index(project: Path, lib_name: str, rows: pd.DataFrame) -> Path:
    """Catalogue of item modules so the agent can find things to reuse."""
    index_path = project / INDEX_FILE
    existing = {}
    if index_path.exists():
        try:
            existing = {e["uuid"]: e for e in json.loads(index_path.read_text())}
        except Exception:
            existing = {}
    for _, row in rows.iterrows():
        mod = module_name_for(row["uuid"])
        existing[row["uuid"]] = {
            "uuid": row["uuid"],
            "type": row["type"],
            "names": list(row["names"]) if row.get("names") is not None else [],
            "module": f"{lib_name}.{ITEMS_SUBDIR}.{mod}",
            "source": row["file_id"],
            "text": row["text"][:400],
        }
    index_path.write_text(json.dumps(list(existing.values()), indent=2))
    return index_path


IMPORT_PATTERN = re.compile(r"^\s*import\s+([A-Za-z0-9_.«»]+)\s*$", flags=re.MULTILINE)


def inline_local_imports(project: Path, lib_name: str, path: Path, seen: set[Path] | None = None) -> str:
    """Splice in project-local modules the agent reused, so `blv` can verify standalone.

    Without this, an item that correctly reuses an earlier definition would be
    scored as a compile failure purely because `blv` runs outside this project.
    """
    seen = seen if seen is not None else set()
    path = path.resolve()
    if path in seen or not path.exists():
        return ""
    seen.add(path)
    text = path.read_text()
    parts: list[str] = []
    for mod in IMPORT_PATTERN.findall(text):
        if not mod.startswith(lib_name + "."):
            continue
        dep = project.joinpath(*mod.split(".")).with_suffix(".lean")
        dep_src = inline_local_imports(project, lib_name, dep, seen)
        if dep_src.strip():
            parts.append(f"-- inlined from {mod}\n{dep_src.strip()}")
    parts.append(common.strip_imports(text))
    return "\n\n".join(p for p in parts if p.strip())


# --------------------------------------------------------------------------- #
# Claude Code invocation
# --------------------------------------------------------------------------- #
MATHATLAS_SERVER = Path(__file__).resolve().parent / "mathatlas_mcp.py"

MATHATLAS_HINT = """- The `mathatlas` MCP tools cover this item's source dataset: `mathatlas_get_context`
  (the textbook prose preceding it, plus its prior definitions), `mathatlas_get_dependencies`
  (the concepts it references), `mathatlas_get_item` / `mathatlas_get_proofs` for any of those
  by id. This item's id is `{entity_id}`. Reach for them when the informal text alone is
  ambiguous about a symbol, a hypothesis, or a referenced concept."""


def build_mcp_servers(
    project: Path,
    lean_lsp_command: str,
    extra_configs: list[Path],
    mathatlas_project: Path | None = None,
    mathatlas_data: Path | None = None,
    mathatlas_textbooks: Path | None = None,
) -> dict[str, Any]:
    """Assemble the MCP server table: lean-lsp, MathAtlas, and anything supplied."""
    argv = lean_lsp_command.split()
    servers: dict[str, Any] = {
        "lean-lsp": {
            "command": argv[0],
            "args": argv[1:],
            "env": {"LEAN_PROJECT_PATH": str(project)},
        }
    }
    if mathatlas_project is not None:
        # Runs the benchmark wrapper (mathatlas_mcp.py) inside the formalization
        # project's environment, where the `atlas` data layer is installed.
        env: dict[str, str] = {}
        if mathatlas_data is not None:
            env["MATHATLAS_DATA"] = str(Path(mathatlas_data).expanduser().resolve())
        if mathatlas_textbooks is not None:
            env["MATHATLAS_TEXTBOOKS"] = str(Path(mathatlas_textbooks).expanduser().resolve())
        servers["mathatlas"] = {
            "command": "uv",
            "args": ["run", "--project", str(Path(mathatlas_project).expanduser().resolve()),
                     "python", str(MATHATLAS_SERVER)],
            "env": env,
        }
    for cfg in extra_configs:
        data = json.loads(Path(cfg).read_text())
        servers.update(data.get("mcpServers", data))
    return servers


def write_mcp_config(servers: dict[str, Any], path: Path, item_uuid: str | None = None) -> Path:
    """Write a per-item MCP config.

    The item under test is baked into the MathAtlas server's `env` rather than
    inherited from this process, so the redaction of that item's mathlib
    grounding cannot silently no-op, and stays correct when agents run in
    parallel.
    """
    servers = copy.deepcopy(servers)
    if item_uuid and "mathatlas" in servers:
        servers["mathatlas"].setdefault("env", {})["MA_HARD_ITEM_UUID"] = str(item_uuid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}, indent=2))
    return path


def run_agent(
    prompt: str,
    project: Path,
    claude_bin: str,
    model: str,
    mcp_config: Path,
    allowed_tools: str,
    max_budget_usd: float | None,
    timeout: int,
    effort: str | None,
    strict_mcp: bool,
    extra_args: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    """One headless Claude Code session, scoped to the Lake project."""
    cmd = [
        claude_bin,
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "json",
        "--permission-mode",
        "acceptEdits",
        "--permission-prompts",
        "none",
        "--mcp-config",
        str(mcp_config),
        "--allowed-tools",
        allowed_tools,
        "--add-dir",
        str(project),
    ]
    if strict_mcp:
        cmd.append("--strict-mcp-config")
    if max_budget_usd is not None:
        cmd += ["--max-budget-usd", str(max_budget_usd)]
    if effort:
        cmd += ["--effort", effort]
    cmd += extra_args

    if dry_run:
        print(" ".join(cmd))
        return {"dry_run": True}

    tik = time.time()
    try:
        proc = subprocess.run(cmd, cwd=project, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"agent_error": "timeout", "duration_s": time.time() - tik}
    elapsed = time.time() - tik

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "agent_error": f"unparseable output (rc={proc.returncode})",
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
            "duration_s": elapsed,
        }
    return {
        "agent_error": payload.get("subtype") if payload.get("is_error") else None,
        "result": payload.get("result"),
        "cost_usd": payload.get("total_cost_usd"),
        "num_turns": payload.get("num_turns"),
        "session_id": payload.get("session_id"),
        "duration_s": payload.get("duration_ms", elapsed * 1000) / 1000,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@app.command()
def run(
    # --- data selection -----------------------------------------------------
    dataset: str = typer.Option("offendo/math-atlas", help="HuggingFace dataset name/path."),
    split: str = typer.Option("train", help="Dataset split (use this if MA-Hard is its own split)."),
    item_type: list[str] = typer.Option(["all"], help="Item type(s) to run; repeatable or 'all'."),
    filter: list[str] = typer.Option([], help="Column filter `col=value`; repeatable (e.g. --filter split=hard)."),
    subset_file: Path | None = typer.Option(None, help="File of MA-Hard uuids (JSON list / JSONL / plain text)."),
    n_examples: int | None = typer.Option(None, help="Subsample this many items (for debugging)."),
    seed: int = typer.Option(1337, help="Seed for subsampling."),
    # --- lean project -------------------------------------------------------
    project: Path = typer.Option(..., help="Lake project to work in; created with `lake init <name> math` if absent."),
    lib_name: str | None = typer.Option(None, help="Library module name; inferred from the lakefile if omitted."),
    build_project: bool = typer.Option(True, help="Run `lake exe cache get` + `lake build` before starting."),
    reset_items: bool = typer.Option(False, help="Delete existing item modules before running (fresh, no reuse)."),
    # --- agent --------------------------------------------------------------
    claude_bin: str = typer.Option("claude", help="Path to the claude CLI."),
    model: str = typer.Option("sonnet", help="Model alias or full name (e.g. sonnet, opus, claude-opus-5)."),
    effort: str | None = typer.Option(None, help="Effort level (low, medium, high, xhigh, max)."),
    mcp_config: list[Path] = typer.Option([], help="Extra MCP config JSON files, merged in."),
    mathatlas_project: Path | None = typer.Option(
        None, help="Formalization project providing the MathAtlas MCP (e.g. ~/src/mathatlas-formalization)."
    ),
    mathatlas_data: Path | None = typer.Option(None, help="MathAtlas dataset JSON; defaults to the project's data/."),
    mathatlas_textbooks: Path | None = typer.Option(None, help="Textbook .mmd dir; defaults to the project's data/."),
    lean_lsp_command: str = typer.Option("uvx lean-lsp-mcp", help="Command that starts the lean-lsp MCP server."),
    allowed_tools: str = typer.Option(
        "Read,Write,Edit,Glob,Grep,Bash,mcp__lean-lsp,mcp__mathatlas",
        help="Tools the agent may use without prompting.",
    ),
    strict_mcp: bool = typer.Option(True, help="Ignore MCP servers outside --mcp-config (reproducibility)."),
    max_budget_usd: float | None = typer.Option(0.75, help="Per-item spend cap. Report this with your numbers."),
    timeout: int = typer.Option(900, help="Per-item wall-clock cap (seconds)."),
    concurrency: int = typer.Option(1, help="Parallel agents. >1 risks lake lock contention and cross-item races."),
    prompt_file: Path = typer.Option(PROMPT_DIR / "agent_task.txt", help="Agent task template."),
    extra_arg: list[str] = typer.Option([], help="Extra raw args passed to the claude CLI; repeatable."),
    dry_run: bool = typer.Option(False, help="Print the claude command for the first item and exit."),
    # --- verification -------------------------------------------------------
    verify_timeout: int = typer.Option(60, help="Per-theorem REPL timeout (seconds)."),
    force_header: bool = typer.Option(True, help="Force `import Mathlib` / `import Aesop` when verifying."),
    inline_imports: bool = typer.Option(True, help="Inline reused project-local modules before verifying."),
    redis_host: str = typer.Option("localhost", help="Redis host for blv workers."),
    redis_port: int = typer.Option(6379, help="Redis port for blv workers."),
    redis_db: int = typer.Option(0, help="Redis DB for blv workers."),
    # --- judging ------------------------------------------------------------
    skip_judge: bool = typer.Option(False, help="Skip alignment judging (compile rate only)."),
    judge_model: str = typer.Option("m-a-p/CriticLeanGPT-Qwen3-32B-RL", help="Alignment judge model name."),
    judge_model_url: str | None = typer.Option(None, help="OpenAI-compatible base URL for the judge."),
    judge_api_key: str = typer.Option("EMPTY", help="API key for the judge endpoint."),
    judge_prompt_file: Path = typer.Option(
        common.REPO_ROOT / "prompts" / "critic_lean_prompt.txt", help="Judge prompt for statements."
    ),
    judge_definition_prompt_file: Path = typer.Option(
        common.REPO_ROOT / "prompts" / "definition_alignment.txt", help="Judge prompt for definitions."
    ),
    judge_max_tokens: int = typer.Option(8192, help="Max judge output tokens."),
    judge_concurrency: int = typer.Option(20, help="Concurrent judge requests."),
    judge_structured: bool = typer.Option(True, help="Request JSON-schema structured judge output."),
    # --- output -------------------------------------------------------------
    output: Path = typer.Option(..., dir_okay=False, help="Output JSON path."),
    resume: bool = typer.Option(False, help="Skip items already present in --output and merge results."),
):
    """Run Claude Code over MA-Hard inside a real Lake project."""
    df = common.load_items(dataset, split, item_type, filter, subset_file, n_examples, seed)

    proj, lib = ensure_project(project, lib_name, build_project)
    items_dir = proj / lib / ITEMS_SUBDIR
    if reset_items:
        shutil.rmtree(items_dir, ignore_errors=True)
        items_dir.mkdir(parents=True, exist_ok=True)
        (proj / INDEX_FILE).unlink(missing_ok=True)

    # Agent runs take hours; each finished item is appended here immediately so a
    # crash costs one item rather than the whole run.
    partial_path = Path(str(output).replace(".json", "")).with_suffix(".partial.jsonl")
    partial_lock = threading.Lock()

    previous = pd.DataFrame()
    if resume:
        done = common.load_done_uuids(output) | load_partial_uuids(partial_path)
        if done:
            if Path(output).exists():
                previous = pd.read_json(output)
            df = df[~df["uuid"].astype(str).isin(done)].reset_index(drop=True)
            logger.info("Resuming: %d already done, %d remaining", len(done), len(df))
            if len(df) == 0:
                logger.info("Nothing left to run.")
                return
    elif partial_path.exists():
        partial_path.unlink()

    index_path = write_index(proj, lib, df)
    scratch = proj / ".ma-hard"
    scratch.mkdir(exist_ok=True)
    mcp_servers = build_mcp_servers(
        proj, lean_lsp_command, list(mcp_config), mathatlas_project, mathatlas_data, mathatlas_textbooks
    )
    logger.info("MCP servers: %s", ", ".join(sorted(mcp_servers)))
    template = prompt_file.read_text()

    if concurrency > 1:
        logger.warning("concurrency=%d: agents share one Lake project; expect lock contention.", concurrency)

    def process(row) -> dict[str, Any]:
        mod = module_name_for(row["uuid"])
        item_file = items_dir / f"{mod}.lean"
        if not item_file.exists():
            item_file.write_text(f"import Mathlib\n\n-- {row['type']}: {row['uuid']}\n-- TODO: replace with the formalization\n")
        prompt = template.format(
            mathatlas_hint=MATHATLAS_HINT.format(entity_id=row["uuid"]) if "mathatlas" in mcp_servers else "",
            text=row["text"],
            item_type=row["type"],
            names=", ".join(row["names"]) if row.get("names") is not None else "",
            item_file=str(item_file),
            module_name=f"{lib}.{ITEMS_SUBDIR}.{mod}",
            items_dir=str(items_dir),
            index_file=str(index_path),
            lib_name=lib,
        )
        item_mcp = write_mcp_config(mcp_servers, scratch / "mcp" / f"{mod}.json", row["uuid"])
        agent = run_agent(
            prompt=prompt,
            project=proj,
            claude_bin=claude_bin,
            model=model,
            mcp_config=item_mcp,
            allowed_tools=allowed_tools,
            max_budget_usd=max_budget_usd,
            timeout=timeout,
            effort=effort,
            strict_mcp=strict_mcp,
            extra_args=list(extra_arg),
            dry_run=dry_run,
        )
        code = inline_local_imports(proj, lib, item_file) if inline_imports else common.strip_imports(item_file.read_text())
        record = {
            "uuid": row["uuid"],
            "file_id": row["file_id"],
            "type": row["type"],
            "text": row["text"],
            "module": f"{lib}.{ITEMS_SUBDIR}.{mod}",
            "item_file": str(item_file),
            "code": code,
            "parsed_output": {"text": code},
            **{k: agent.get(k) for k in ("agent_error", "result", "cost_usd", "num_turns", "session_id", "duration_s")},
        }
        with partial_lock:
            with open(partial_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        return record

    rows = [row for _, row in df.iterrows()]
    if dry_run:
        process(rows[0])
        return

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            records = list(tqdm(pool.map(process, rows), total=len(rows), desc="Agents"))
    else:
        records = [process(r) for r in tqdm(rows, desc="Agents")]

    del records  # the checkpoint file is the source of truth from here on
    out = pd.DataFrame.from_records(read_partial(partial_path))

    results = common.verify_batch(
        out["code"].tolist(),
        timeout=verify_timeout,
        force_header=common.DEFAULT_HEADER if force_header else None,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
    )
    out["compiler_output"] = results
    out["verified"] = [bool(r.get("verified")) for r in results]
    out["degenerate"] = [common.is_degenerate(c) for c in out["code"]]

    if not skip_judge:
        common.judge_and_attach(
            out,
            judge_model=judge_model,
            judge_model_url=judge_model_url,
            judge_prompt_file=judge_prompt_file,
            judge_definition_prompt_file=judge_definition_prompt_file,
            judge_max_tokens=judge_max_tokens,
            judge_concurrency=judge_concurrency,
            api_key=judge_api_key,
            structured=judge_structured,
        )

    if not previous.empty:
        fresh = set(out["uuid"].astype(str)) if not out.empty else set()
        keep = previous[~previous["uuid"].astype(str).isin(fresh)]
        out = pd.concat([keep, out], ignore_index=True) if not keep.empty else out

    metrics = common.summarize(out)
    common.print_metrics(metrics)
    config = {
        "baseline": "A1-claude-code",
        "model": model,
        "effort": effort,
        "dataset": dataset,
        "split": split,
        "item_type": list(item_type),
        "filter": list(filter),
        "subset_file": str(subset_file) if subset_file else None,
        "project": str(proj),
        "lib_name": lib,
        "allowed_tools": allowed_tools,
        "mcp_servers": sorted(mcp_servers),
        "mcp_config": [str(p) for p in mcp_config],
        "mathatlas_project": str(mathatlas_project) if mathatlas_project else None,
        "lean_lsp_command": lean_lsp_command,
        "max_budget_usd": max_budget_usd,
        "timeout_s": timeout,
        "concurrency": concurrency,
        "inline_imports": inline_imports,
        "force_header": force_header,
        "judge_model": None if skip_judge else judge_model,
        "seed": seed,
    }
    common.save_results(out, metrics, output, config)


if __name__ == "__main__":
    app()
