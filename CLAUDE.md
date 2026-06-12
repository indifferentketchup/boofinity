# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

indifferentketchup's fork of michaelfeil/infinity (MIT), an embedding/reranking REST
server. Work happens on branch `ik-main`, cut from upstream release tag `0.0.77` (main
tracks upstream). The fork is being modernized in batches; deployment target is
serving `BAAI/bge-m3` plus a reranker as a llama-swap child process on CPU machines.
This box is CPU-only: never require CUDA for anything.

Fork state lives in these documents, read them before planning work:
- `RECON.md`: dependency staleness, backend inventory, import-time cost, SIGTERM
  trace, dead code, modernization risks (all with file:line citations)
- `BASELINE_NOTES.md`: how `.venv-baseline` was built and why it deviates from
  a plain install
- `openspec/changes/`: planned/active change folders (OpenSpec workflow:
  `openspec validate <id>`, `openspec list`)

## Operator law

- Never commit, push, or stage unless explicitly instructed; prove edits with
  `git diff --stat`. When staging is requested, add files by explicit path, never
  `git add -A`.
- Git identity is repo-local `indifferentketchup <sam@indifferentketchup.com>`. Never
  commit with any other email.
- `.venv-baseline/` and `baseline-freeze.txt` are the frozen batch-1 reference. Never
  modify them; build new venvs (`.venv-batch2` etc.) for new dependency resolutions.
- No em dashes (U+2014) in output or files.

## Commands

The Python package is `libs/infinity_emb/` (source in `libs/infinity_emb/infinity_emb/`,
poetry-core build backend, installable with pip).

```bash
# Venv build recipe (CPU box; torch must come from the CPU wheel index first)
python3 -m venv .venv-NAME && .venv-NAME/bin/pip install -U pip
.venv-NAME/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-NAME/bin/pip install -e "libs/infinity_emb[torch,server,logging]"
.venv-NAME/bin/pip install pytest pytest-mock httpx asgi_lifespan anyio trio  # test tooling

# Tests (from libs/infinity_emb/; use engine="debugengine" patterns to avoid model downloads)
../../.venv-baseline/bin/python -m pytest tests/unit_test -x -q
../../.venv-baseline/bin/python -m pytest tests/unit_test/test_engine.py -x -q   # single file
../../.venv-baseline/bin/python -m pytest tests/end_to_end/test_api_with_dummymodel.py -x -q

# Lint (upstream uses ruff + mypy; line-length 100 per pyproject)
ruff check libs/infinity_emb/infinity_emb

# Embedding parity gate: REQUIRED after any dependency or model-path change.
# Must run from tests/parity/ (it imports its sibling common.py).
cd tests/parity && ../../.venv-baseline/bin/python check_parity.py
# PASS means every input >= 0.9999 cosine vs fixtures/baseline_bge-m3_cpu.npz.
# capture_baseline.py regenerates the baseline; only do that deliberately.

# Run the server (v2 CLI)
.venv-baseline/bin/infinity_emb v2 --model-id BAAI/bge-m3 --device cpu

# Import-time profile (cold-start target; baseline ~4.4-5.5 s, log in importtime.log)
.venv-baseline/bin/python -X importtime -c "import infinity_emb" 2> importtime.log
```

## Architecture

Request path, async to threads and back (the part that takes multiple files to see):

1. `engine.py`: `AsyncEngineArray` holds one `AsyncEmbeddingEngine` per model.
   `astart()`/`astop()` manage lifecycle; `engine.running` is the readiness flag
   (array method `is_running()` at engine.py:343).
2. `inference/batch_handler.py`: the core. The asyncio side (`embed`/`rerank` ->
   `_schedule`) creates futures, queues `PrioritizedQueueItem`s, and awaits an
   `asyncio.gather` on the results. A `ModelWorker` runs three daemon threads
   (`_preprocess_batch`, `_core_batch`, `_postprocess_batch`) connected by
   `queue.Queue`s, all polling with `QUEUE_TIMEOUT = 0.5` and checking a shared
   `threading.Event` `_shutdown` between (not during) operations. Results come back
   via `item.complete()` resolving the caller's future.
3. Backend dispatch: `primitives.py` defines the `InferenceEngine`, `Device`, `Dtype`
   enums; `transformer/utils.py` maps enum -> class (torch path is
   `SentenceTransformerPatched` in `transformer/embedder/sentence_transformer.py`);
   `inference/select_model.py` picks embedder/reranker/classifier from the HF config
   and runs warmup. Optional backends (optimum/ONNX, ctranslate2, neuron) are guarded
   by `_optional_imports.py` `CHECK_*` objects; `debugengine` (dummytransformer) is
   the test backend.
4. Server: `infinity_server.py` `create_server()` builds the FastAPI app; the ASGI
   lifespan creates and starts the engine array (so uvicorn accepts connections only
   after models load) and calls `astop()` on shutdown. `cli.py` (typer) `v2` command
   parses env-var-backed args (`env.py` `MANAGER`) and calls `uvicorn.run`.

Multi-model serving, caching (`inference/caching_layer.py`), and embedding dtype
quantization (`transformer/quantization/`) hang off the same BatchHandler pipeline.

## Known traps at tag 0.0.77 (verified, see RECON.md for citations)

- Installing optimum 2.x alongside crashes the torch path at import:
  `transformer/acceleration.py` imports `optimum.bettertransformer` whenever optimum
  is present. Batch 2 (openspec change `modernize-deps-and-lifecycle`) removes it.
- huggingface_hub >= 1.0 breaks `transformer/utils_optimum.py` (`HfFolder` removed);
  upstream fix is commit `2ecb218` (after the tag).
- `numpy < 2` is pinned for ONNX; transformers resolves 4.x regardless of caps because
  sentence-transformers 3.x requires `transformers<5`.
- `/health` returns 200 unconditionally; readiness currently works only because
  uvicorn defers serving until lifespan completes.
- The pyproject `[all]` extra and Makefile assume poetry; this fork develops with
  plain pip venvs instead.
