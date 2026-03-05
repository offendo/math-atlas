import json
import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import typer
from datasets import load_dataset
from tqdm import tqdm

from proofflow import ProofFlow, LLMManager, LeanServer

app = typer.Typer(pretty_exceptions_show_locals=False)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger("run_proof_flow")
logger.setLevel(logging.INFO)


def load_proofs(dataset: str) -> list[str]:
    """
    Load informal proofs with theorems from a HuggingFace dataset.
    """
    ds = load_dataset(dataset, split="train")

    def _format(ex):
        proof = ex["text"]
        thm = ex["text"]
        return {"example": f"Theorem: {thm}\nProof: {proof}"}

    ds = ds.map(_format)
    return list(ds["example"])


@app.command()
def run(
    graph_model: str = typer.Option(..., help="Graph model name for ProofFlow."),
    formalize_model: str = typer.Option(..., help="Formalizer model name for ProofFlow."),
    solver_model: str = typer.Option(..., help="Solver model name for ProofFlow."),
    dataset: str = typer.Option(..., help="HF dataset name."),
    output: Path | None = typer.Option(
        None,
        help="Output JSON file. Defaults to '<model>.proof.json'.",
    ),
    graph_url: str = typer.Option(
        "http://localhost:8001/v1", help="vLLM url which hosts graph/formalize/solver models."
    ),
    solver_url: str = typer.Option(None, help="vLLM url which hosts graph/formalize/solver models."),
    formalize_url: str = typer.Option(None, help="vLLM url which hosts graph/formalize/solver models."),
    prompt_dir: Path = typer.Option(".", help="Path to directory containing prompts"),
    lean_server_url: str = typer.Option("http://localhost:1337", help="URL for remote Kimina Lean Server"),
    limit: int | None = typer.Option(None, help="Optional limit for debugging."),
):
    """
    Benchmark ProofFlow on a dataset of informal proofs.
    """

    if solver_model is None and formalize_model is None:
        raise ValueError("Need at least one of --formalize-model or --solver-model")

    if output is None:
        default_name = model.replace("/", ".").lower() + ".proof.json"
        output = Path(default_name)

    logger.info("Loading proofs...")
    proofs = load_proofs(dataset)

    if limit is not None:
        proofs = proofs[:limit]

    logger.info(f"Loaded {len(proofs)} proofs.")

    # Set up Lean server (local or remote)
    lean_server = LeanServer(api_url=lean_server_url)  # Remote server

    # Configure LLM models
    graph_model = LLMManager(
        model_info={
            "base_url": graph_url,
            "model": graph_model,
        },
        system_prompt_path=Path(prompt_dir, "proof_graph.md"),
    )

    formalize_model = LLMManager(
        model_info={
            "base_url": formalize_url or solver_url,
            "model": formalize_model,
        },
        system_prompt_path=Path(prompt_dir, "lemma_formalizer.md"),
    )

    solver_model = LLMManager(
        model_info={
            "base_url": solver_url or formalize_url,
            "model": solver_model,
        },
        system_prompt_path=Path(prompt_dir, "lemma_prover.md"),
    )
    proof_flow = ProofFlow(
        lean_server=lean_server,
        graph_model_manager=graph_model,
        formalize_model_manager=formalize_model,
        solver_model_manager=solver_model,
        verbose=True,
    )

    results = []
    for idx, proof in enumerate(tqdm(proofs, desc="Running ProofFlow")):
        logger.debug(f"Processing example {idx}")

        proof_flow.autoformalize_series(proof)

        summary = proof_flow.summary()
        lean_code = proof_flow.get_lean_code()

        results.append(
            {
                "example_id": idx,
                "informal_proof": proof,
                "summary": summary,
                "lean_code": lean_code,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(results)
    df.to_json(output)
    logger.info(f"Saved results to {output}")


if __name__ == "__main__":
    app()
