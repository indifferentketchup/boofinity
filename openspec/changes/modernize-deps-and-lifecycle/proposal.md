# Modernize dependencies and harden server lifecycle (batch 2)

## Why

RECON.md (batch 1) and the llama.cpp transfer analysis identified four defects that block
the fork's goals (current deps, clean SIGTERM as a llama-swap child):

1. `optimum.bettertransformer` is imported whenever optimum is installed
   (`libs/infinity_emb/infinity_emb/transformer/acceleration.py:10-14`); optimum 2.x deleted
   that module, so a modern optimum install crashes every torch model class at import.
   The flag defaults to true (`env.py:147`).
2. Tag 0.0.77 still uses `HfFolder`, removed in huggingface_hub 1.0. Upstream fixed it
   after the tag in commit 2ecb218 (2 lines in `transformer/utils_optimum.py`).
3. Pins block modernization: `uvicorn ^0.32.0` (pyproject line 25, latest 0.49) and
   `transformers >=4.47.0,<=5.0` (line 33, latest 5.12).
4. SIGTERM is graceful only between batches: `_core_batch` never checks `_shutdown`
   before the forward pass (`inference/batch_handler.py:576`), the result gather waits
   forever if the pipeline dies (`batch_handler.py:329-331`), and `/health` returns 200
   unconditionally (`infinity_server.py:162-170`), so readiness for llama-swap depends
   on an undocumented uvicorn ordering detail.

## What Changes

- Apply the upstream hf_hub>=1 fix (2ecb218) to the working tree, uncommitted.
- Remove BetterTransformer end to end: trim `acceleration.py` to its TF32 block, delete
  call sites in the five torch model modules, flip the env default to false, keep the
  CLI/EngineArgs flag as a warn-and-ignore for deployed configs.
- Lift the uvicorn and transformers pins in `libs/infinity_emb/pyproject.toml`.
- Add a shutdown check before the forward pass, an opt-in bounded wait on the result
  gather, and a readiness-gated `/health` (503 until `engine_array.is_running()`).
- Prove every step with the batch-1 parity harness (`tests/parity/check_parity.py`,
  min cosine >= 0.9999 against the frozen baseline) from a new `.venv-batch2`.

## Impact

- Affected specs: `dependency-stack` (new), `server-lifecycle` (new)
- Affected code: `transformer/acceleration.py`, `transformer/abstract.py`,
  `transformer/embedder/sentence_transformer.py`, `transformer/crossencoder/torch.py`,
  `transformer/classifier/torch.py`, `transformer/audio/torch.py`,
  `transformer/vision/torch_vision.py`, `transformer/utils_optimum.py`, `args.py`,
  `env.py`, `inference/batch_handler.py`, `infinity_server.py`,
  `libs/infinity_emb/pyproject.toml`, new tests
- Not affected: baseline venv and fixtures (stay frozen), ONNX/CT2/neuron backends
  (deleted in a later batch), public API surface (flag kept as no-op)

## Deferred (YAGNI)

- Double-signal force exit: uvicorn force-exits on a second SIGINT
  (`.venv-baseline/.../uvicorn/server.py:334-339`) and llama-swap escalates SIGTERM to
  kill after its stop timeout. Reopen if the server ever runs without a supervisor.
- Per-stage worker crash flags (only `_preprocess_batch` clears `_ready`,
  `batch_handler.py:560`). Reopen on the first observed silent pipeline stall.
- numpy >= 2 (pinned `<2` "for onnx", pyproject line 17). Reopen when the ONNX backend
  is deleted or onnxruntime confirms numpy 2 support for our resolved version.
- sentence-transformers major bump (^3.0.1, latest 5.x); `quantize_embeddings` API
  compatibility unverified. Reopen when a 5.x feature or fix is needed.
- Cold-start lazy imports and QUEUE_TIMEOUT latency floor (batch 3, with importtime.log
  and a latency benchmark as baselines).
- Explicit weight release in `astop()` (CPU deployment; GC at process exit suffices).
  Reopen when a CUDA deployment shares GPUs across processes.
