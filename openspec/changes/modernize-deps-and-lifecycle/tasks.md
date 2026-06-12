# Tasks

## 1. Dependency modernization

- [ ] 1.1 Apply the upstream hf_hub fix: from the repo root run `git show 2ecb218 | git apply`.
      Verify: `grep -rn HfFolder libs/infinity_emb/infinity_emb/` is empty and
      `git diff --stat` shows only `transformer/utils_optimum.py` (2 changed lines).
- [ ] 1.2 Trim `transformer/acceleration.py` to the TF32 block only (current lines 16-22):
      delete lines 10-14 (BT import), 23-24 (dead AutoConfig import), 35-97 (both BT
      functions), and the then-unused `CHECK_OPTIMUM`, `CHECK_TRANSFORMERS`, `Device`
      imports; survivor imports only `CHECK_TORCH` and `torch` (V8).
      Verify: `grep -n "bettertransformer\|AutoConfig\|CHECK_OPTIMUM" transformer/acceleration.py`
      is empty; `.venv-baseline/bin/python -c "import infinity_emb.transformer.acceleration"` exits 0.
- [ ] 1.3 Add `from infinity_emb.transformer import acceleration  # noqa: F401` (TF32 side
      effect) to `transformer/abstract.py`. Verify: `.venv-baseline/bin/python -c "import
      infinity_emb.transformer.abstract; import torch; assert torch.backends.cuda.matmul.allow_tf32"`.
- [ ] 1.4 Remove BT call sites in `transformer/embedder/sentence_transformer.py`
      (imports at 20-21, `attempt_bt` blocks at 62-66 and 95-101, including the
      `attn_implementation="eager"` kwarg). Verify: file has no `bettertransformer`
      matches; module imports cleanly.
- [ ] 1.5 Remove BT call sites in `transformer/crossencoder/torch.py` (32-33, 48-49, 73-74)
      and `transformer/classifier/torch.py` (9-10, 29-30, 53-54), reading each block
      before deleting. Verify: no `bettertransformer` matches in either file; both import.
- [ ] 1.6 Clean up commented-out BT references in `transformer/audio/torch.py` (line 34)
      and `transformer/vision/torch_vision.py` (line 97); these are comments, not active
      call sites (V1), so this is cosmetic.
      Verify: `grep -rn bettertransformer libs/infinity_emb/infinity_emb/transformer/` is empty.
- [ ] 1.7 Flip the env default: `env.py:147` `default=["true"]` to `default=["false"]`.
      Add a warning in `EngineArgs.__post_init__` (`args.py`) when `bettertransformer`
      is true: deprecated and ignored. Verify: starting a debugengine with
      `--bettertransformer true` logs the warning; default start does not.
- [ ] 1.8 Lift pins in `libs/infinity_emb/pyproject.toml`: line 25 uvicorn `^0.32.0` to
      `>=0.32.0`; line 33 transformers `>=4.47.0,<=5.0` to `>=4.47.0,<6`. Verify:
      `python3 -c "import tomllib; tomllib.load(open('libs/infinity_emb/pyproject.toml','rb'))"`.
- [ ] 1.9 Build `.venv-batch2` with the BASELINE_NOTES.md recipe (venv, pip upgrade, CPU
      torch index, editable install `[torch,server,logging]`), then install the test
      tooling the suite needs: `pip install pytest pytest-mock httpx asgi_lifespan
      anyio trio` (JD-001/JD-002; matches the pyproject test group). Write
      `pip freeze > batch2-freeze.txt`. Verify: `import infinity_emb` reports 0.0.77,
      `pytest --version` works from the venv, and
      resolved uvicorn is >= 0.49. Note: transformers will resolve 4.x (sentence-
      transformers 3.4.1 pins `<5.0.0`, V3), so the `torch_dtype` rename in design D3
      will not trigger this batch. Allow 20-40 min wall clock for downloads (V9).
- [ ] 1.10 Co-install acceptance for BT removal: `.venv-batch2/bin/pip install optimum
      onnxruntime` (current 2.x), then `.venv-batch2/bin/python -c "import infinity_emb"`
      and a debugengine startup. Verify: both exit 0 (they crash on tag 0.0.77).
- [ ] 1.11 Parity gate: `cd tests/parity && ../../.venv-batch2/bin/python check_parity.py`
      (the harness imports `common.py` from its own directory, JD-013).
      Verify: PASS, min cosine >= 0.9999, exit 0. The bge-m3 weights
      are already in the HF cache from batch 1; CPU inference takes a few minutes (V9).

## 2. Lifecycle hardening

Prerequisite: `.venv-batch2` from task 1.9 (with test tooling) must exist before any
section-2 verification step runs (JD-003). The code edits themselves are independent
of section 1 and may be developed in parallel once the venv exists.

- [ ] 2.1 In `inference/batch_handler.py`: add `if self._shutdown.is_set(): break` after a
      successful dequeue in BOTH `_core_batch` (after `_feature_queue.get`, lines
      567-576) and `_postprocess_batch` (after `_postprocess_queue.get` at line 595;
      same shape, verified V10). Verify: existing unit tests pass in `.venv-batch2`
      (`pytest libs/infinity_emb/tests/unit_test/test_engine.py -x -q`).
- [ ] 2.2 Add `INFINITY_REQUEST_TIMEOUT_S` to `env.py` via the MANAGER pattern (default
      `0`, disabled). Verify: `MANAGER.request_timeout_s` resolves with default and
      with the env var set.
- [ ] 2.3 Wrap the result gather (`batch_handler.py:329-331`) in `asyncio.wait_for` when
      the timeout is nonzero; catch the `TimeoutError` raised in the caller and re-raise
      an `OpenAIException` with a descriptive request-timeout message and 5xx code,
      since the handler's fallback branch (`fastapi_schemas/errors.py:43-56`) would
      otherwise emit a generic 500 (V4). Verify: new unit test from 2.5.
- [ ] 2.4 Gate `/health` (`infinity_server.py:162-170`): resolve the array defensively,
      `engine_array = getattr(app, "engine_array", None)`, and return 503
      `{"status": "loading"}` when it is None, empty, or `not engine_array.is_running()`
      (JD-006; also covers `all([]) is True` on an empty engines_dict, JD-012), else the
      current unix-timestamp payload. Verify: new test from 2.6.
- [ ] 2.5 New unit test: gather timeout. With debugengine and a monkeypatched
      never-completing future and `INFINITY_REQUEST_TIMEOUT_S=1`, an embed call raises
      within ~1 s instead of hanging. Verify: test passes; with timeout 0 the old
      behavior holds.
- [ ] 2.6 New tests: shutdown responsiveness and health gate. (a) debugengine with items
      queued: `astop()` returns in < 2 s; (b) ASGI test (pattern:
      `tests/end_to_end/test_api_with_dummymodel.py` with `asgi_lifespan`): `/health`
      is 200 after startup; calling the handler with a stopped array yields 503.
      The existing assertion at `test_api_with_dummymodel.py:62-64` doubles as the
      200-when-running regression test and must stay green (V7). Verify: pytest green.
- [ ] 2.7 Final gate: re-run `cd tests/parity && ../../.venv-batch2/bin/python
      check_parity.py`; capture
      `git diff --stat` for the operator. Verify: parity PASS and the diff touches only
      the files listed in proposal Impact.
