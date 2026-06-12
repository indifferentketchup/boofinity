# Design

## Context

Branch `ik-main` at tag 0.0.77. Baseline venv `.venv-baseline` is the frozen reference
(BASELINE_NOTES.md); all new resolution happens in a fresh `.venv-batch2`. The parity
harness from batch 1 is the regression gate for every behavioral claim.

## D1: hf_hub >= 1.0 fix by applying the upstream diff

`git show 2ecb218 | git apply` from the repo root. The commit touches only
`libs/infinity_emb/infinity_emb/transformer/utils_optimum.py` (import `get_token`
instead of `HfFolder` at line 8; `get_token()` instead of `HfFolder().get_token()` at
line 212). Applying the diff (not cherry-picking) keeps the tree uncommitted per
operator law. Acceptance: `grep -rn HfFolder libs/infinity_emb/infinity_emb/` is empty.

## D2: BetterTransformer removal

Evidence: `acceleration.py:10-14` imports `optimum.bettertransformer` under
`CHECK_OPTIMUM.is_available`; optimum 2.x has no such module, so the import raises at
module load of every file that imports acceleration. Call-site inventory (grep plus
adversarial re-verification, V1): active sites are
`embedder/sentence_transformer.py:20-21,62-66,95-101`,
`crossencoder/torch.py:32-33,48-49,73-74`, `classifier/torch.py:9-10,29-30,53-54`.
`audio/torch.py:34` and `vision/torch_vision.py:97` are commented-out references only
(cosmetic cleanup, no imports). `embedder/ct2.py:76` inherits the fix via
`super().__init__`.

Approach:
- Trim `acceleration.py` to only the TF32 enablement block (current lines 16-22),
  which must survive: it is a module-level side effect the torch path relies on for
  CUDA throughput on the remote machines. Full deletion scope (V8): lines 10-14
  (BT import), 23-24 (now-dead `AutoConfig` import), 35-97 (both BT functions), and
  the then-unused `CHECK_OPTIMUM`, `CHECK_TRANSFORMERS`, `Device` imports; the
  surviving module imports only `CHECK_TORCH` and `torch`.
- Import `acceleration` for side effect from `transformer/abstract.py` (already
  imported by every model class), so TF32 stays enabled with one import site.
- Delete the `check_if_bettertransformer_possible`/`to_bettertransformer` call sites
  in the five model modules. The `attn_implementation="eager"` kwarg set at
  `sentence_transformer.py:64` exists only to support BT and goes with it; transformers
  defaults to SDPA, which is the modern replacement for BT.
- Keep the `bettertransformer` EngineArgs field (`args.py:65`) and CLI flag for
  backward compatibility with deployed systemd units, but flip the env default from
  `["true"]` to `["false"]` (`env.py:147`) and log one warning from
  `EngineArgs.__post_init__` when the flag is explicitly true: deprecated, ignored.
  Warning placement in args (not per-model-class) fires once per engine.

## D3: Pin lifts

`libs/infinity_emb/pyproject.toml`: line 25 `uvicorn ^0.32.0` becomes `>=0.32.0`;
line 33 `transformers >=4.47.0,<=5.0` becomes `>=4.47.0,<6`. The `<6` cap is
deliberate: transformers 5.x is allowed (current 5.12), an unknown future 6.x is not.
numpy stays `<2` (deferred, see proposal). torch needs no change (`>=2.2.1` already
resolves 2.12).

Risk note (corrected by V3): the `torch_dtype` risk is dormant this batch.
sentence-transformers 3.4.1 transitively requires `transformers<5.0.0,>=4.41.0`
(verified in `.venv-baseline` dist metadata), so pip resolves transformers 4.x
regardless of our `<6` cap until the deferred sentence-transformers bump happens.
The kwarg uses at `sentence_transformer.py:70`, `classifier/torch.py:36`,
`crossencoder/torch.py:56`, `vision/torch_vision.py:56,58` remain forward guidance
for that deferred change, not work in this one.

## D4: Validation venv

`.venv-batch2` built with the BASELINE_NOTES.md recipe (CPU torch wheel index, extras
`[torch,server,logging]` plus the pytest test tooling, JD-001/JD-002), freeze to
`batch2-freeze.txt`. Acceptance for D2 is the co-install test: `pip install optimum
onnxruntime` (current 2.x) into `.venv-batch2`, then `python -c "import infinity_emb"`
and a debugengine startup must succeed. The claim that tag 0.0.77 crashes under this
combination is code-analysis-based (acceleration.py import guard), not yet
demonstrated on this box (JD-004); the implementer may optionally demonstrate it in a
scratch venv first to anchor the before state.

## D5: Shutdown check before the forward pass

`batch_handler.py:567-576`: `_core_batch` dequeues then calls
`self._model.encode_core(feat)` with no shutdown check between them. Add
`if self._shutdown.is_set(): break` after a successful dequeue, before encode.
Apply the same guard to `_postprocess_batch` (verified V10: identical
dequeue-then-work shape at lines 590-621, get at 595, work at 600-611, put at 617).
This bounds shutdown latency to one in-flight forward pass instead of the full queue
backlog.

Strand semantics (V2): when shutdown fires between dequeue and encode, the dequeued
item's futures never complete. This matches existing behavior (the current inner
while-loops already exit without `put`/`task_done` on shutdown) and is acceptable
during shutdown; nothing calls `_feature_queue.join()` (grep verified zero matches),
so the skipped `task_done()` is harmless. The fix saves wasted compute; it does not
introduce a new strand case.

## D6: Bounded result wait, opt-in

`batch_handler.py:329-331` gathers per-item futures with no timeout; a dead pipeline
hangs callers forever (confirmed by the architecture analysis, finding B7). Wrap with
`asyncio.wait_for` using `INFINITY_REQUEST_TIMEOUT_S` read via the `env.py` MANAGER
pattern, default `0` meaning disabled, preserving current behavior exactly unless the
operator opts in.

Error surface (V4): the fallback branch of `openai_exception_handler`
(`fastapi_schemas/errors.py:43-56`) turns any non-`OpenAIException` into a generic
500 "Internal Server Error". So on timeout, catch the `TimeoutError` that
`asyncio.wait_for` raises in the caller (Python 3.12 semantics) and re-raise an
`OpenAIException` with a descriptive request-timeout message and a 5xx code.

Orphan semantics (V5): timed-out items are already in `_queue_prio` and will be fully
processed by the worker pipeline with results discarded (futures completed, nobody
awaiting). No memory or state leak (the result store does not track in-flight
futures); wasted model cycles are acceptable for an opt-in safety mechanism.

Cancellation scope (JD-005, corrected): the gather at `batch_handler.py:329-331` spans
the sentences of a single API call (`_schedule` is invoked per request), so a timeout
cancels exactly that one request, not unrelated callers. Per-batch vs per-future
granularity is therefore a non-issue. `batch_handler.py:16` already imports `MANAGER`
from `env.py`, so no new import or circularity (JD-008).

## D7: Readiness-gated /health

`infinity_server.py:162-170` returns `{"unix": time.time()}` unconditionally. Gate:
resolve `engine_array = getattr(app, "engine_array", None)` (the attribute is only
set at `infinity_server.py:88` inside lifespan startup, JD-006) and return 503 with
`{"status": "loading"}` when it is None, has no engines (`all([]) is True`, JD-012),
or `not engine_array.is_running()` (method, `engine.py:343-344`); otherwise the
current payload. Today uvicorn refuses
connections until lifespan startup (model load) completes, so llama-swap works by
accident; the gate makes the contract explicit and survives any future change that
serves requests during load. llama-swap polls its `checkEndpoint` (default `/health`)
for HTTP 200 before routing.

Verified flow states (V6, V11): `preload_only` runs AFTER `astart()`
(`infinity_server.py:88-106`), so a preloaded server correctly reports 200; after
`astop()` sets `running=False` (`engine.py:105`) the gate reports 503 while in-flight
requests drain, which is correct shutdown signaling. The existing health assertion at
`tests/end_to_end/test_api_with_dummymodel.py:62-64` runs after `LifespanManager`
startup and doubles as the 200-when-running regression test (V7).

## Test strategy

Existing patterns: `libs/infinity_emb/tests/unit_test/test_engine.py` and
`tests/end_to_end/test_api_with_dummymodel.py` exercise `engine="debugengine"` without
model downloads. New tests follow them: shutdown responsiveness (astop returns promptly
with items queued), gather timeout (monkeypatched never-completing future), health gate
(503 before astart, 200 after, 503 after astop). Final gate is always
`tests/parity/check_parity.py` from `.venv-batch2` against the frozen baseline.

## Alternatives considered

- Cherry-pick 2ecb218 as a commit: rejected, operator commits manually.
- Delete the `bettertransformer` flag outright: rejected, breaks deployed CLI configs;
  warn-and-ignore is reversible and grep-discoverable for later removal.
- Default the gather timeout on (e.g. 1800 s): rejected for this batch; changing
  failure semantics silently is riskier than opt-in, revisit with llama-swap configs.
