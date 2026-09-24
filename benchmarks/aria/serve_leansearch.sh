#!/usr/bin/env bash
# Serve LeanSearch v2 (standard mode) locally for Aria's RAG grounding.
#
# Aria's configs/leansearch.yaml points at http://127.0.0.1:8003/search.
# The pipeline puts the embedder on the first visible GPU and the reranker on
# the rest, so it needs two: GPUS=1,2 by default.
#
#   benchmarks/aria/serve_leansearch.sh            # foreground
#   curl -s localhost:8003/health
set -euo pipefail

LS_ROOT="${LS_ROOT:-$HOME/src/Aria-autoformalizer/Aria-autoformalizer/LeanSearch-v2}"
GPUS="${GPUS:-1,2}"
PORT="${PORT:-8003}"
EMBEDDER="${EMBEDDER:-Qwen/Qwen3-Embedding-8B}"   # must match the index
RERANKER="${RERANKER:-Qwen/Qwen3-Reranker-8B}"    # the paper's Table 1 config

cd "$LS_ROOT"
HF=.venv/bin/hf

# Model names would be resolved as repo-relative paths; hand over the snapshots.
export EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-$($HF download "$EMBEDDER" --quiet)}"
export RERANKER_MODEL_PATH="${RERANKER_MODEL_PATH:-$($HF download "$RERANKER" --quiet)}"
export VECTORDB_DIR="${VECTORDB_DIR:-$LS_ROOT/data/cuvs/mathlib-v4.28.0-rc1}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export NUM_GPUS="${NUM_GPUS:-$(tr ',' '\n' <<< "$GPUS" | wc -l)}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
export PYTHONPATH="$LS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec .venv/bin/python -m uvicorn leansearchv2.server:app --host 127.0.0.1 --port "$PORT"
