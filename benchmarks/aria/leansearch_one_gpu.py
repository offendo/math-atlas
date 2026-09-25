"""Serve LeanSearch v2 with embedder and reranker sharing a single GPU.

LeanSearch puts the embedder on cuda:0 and reranker replicas on cuda:1..n-1,
so with one visible GPU it loads no reranker at all and `_rerank` silently
falls back to retriever order. Here the reranker is placed on the embedder's
GPU instead (two 8B fp16 models, ~32 GB). Run with LeanSearch's src on the
path; `serve_leansearch.sh` does this when given a single GPU.
"""

import sys

import uvicorn
from leansearchv2.pipeline import RetrievalPipeline

_orig_load_models = RetrievalPipeline._load_models


def _load_models(self) -> None:
    if not self.reranker_devices:
        self.reranker_devices = [self.embedding_device]
        print(f"Single GPU: reranker shares {self.embedding_device} with the embedder")
    _orig_load_models(self)


RetrievalPipeline._load_models = _load_models

if __name__ == "__main__":
    uvicorn.run("leansearchv2.server:app", host="127.0.0.1", port=int(sys.argv[1]))
