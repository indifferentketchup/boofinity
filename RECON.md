# Recon Report: infinity-emb 0.0.77

Repo: `michaelfeil/infinity` at tag `0.0.77`, branch `ik-main`.
Package root: `libs/infinity_emb/` (editable install).
Box: CPU-only, Python 3.12, venv at `.venv-baseline/`.

---

## 1. Dependency Graph

All pins from `libs/infinity_emb/pyproject.toml:14-57`. Resolved versions from `baseline-freeze.txt`. Latest from PyPI as of 2026-06-12.

| Package | Pinned (pyproject) | Resolved (freeze) | Latest (PyPI) | Majors behind |
|---|---|---|---|---|
| numpy | `>=1.20.0,<2` | 1.26.4 | 2.4.6 | 1 |
| huggingface_hub | `>=0.32.0` | 0.36.2 | 1.19.0 | 1 |
| torch | `>=2.2.1` | 2.12.0+cpu | 2.12.0 | 0 |
| transformers | `>=4.47.0,<=5.0` | 4.57.6 | 5.12.0 | 1 |
| sentence-transformers | `^3.0.1` | 3.4.1 | 5.5.1 | 2 |
| fastapi | `>=0.103.2` | 0.136.3 | 0.136.3 | 0 |
| pydantic | `>=2.4.0,<3` | 2.13.4 | 2.13.4 | 0 |
| uvicorn | `^0.32.0` | 0.32.1 | 0.49.0 | 0 (semver-minor) |
| orjson | `>=3.9.8,!=3.10.0` | 3.11.9 | 3.11.9 | 0 |
| ctranslate2 | `>=4.0.0` | (not installed) | 4.8.0 | 0 |
| optimum | `>=1.24.0` | (not installed) | 2.2.0 | 1 |
| einops | `*` | (not installed) | 0.8.2 | 0 |
| pillow | `*` | 12.2.0 | 12.2.0 | 0 |
| timm | `*` | (not installed) | 1.0.27 | 0 |
| diskcache | `*` | (not installed) | 5.6.3 | 0 |
| rich | `^13` | 13.9.4 | 15.0.0 | 2 |
| typer | `^0.12.5` | 0.12.5 | 0.26.7 | 0 (semver-minor) |
| posthog | `*` | 7.18.3 | 7.18.3 | 0 |
| prometheus-fastapi-instrumentator | `>=6.1.0` | 8.0.0 | 8.0.0 | 0 |
| colpali-engine | `^0.3.8` | (not installed) | 0.3.17 | 0 |
| soundfile | `^0.12.1` | (not installed) | 0.14.0 | 0 |
| torchvision | `*` | (not installed) | 0.27.0 | 0 |
| tensorrt | `^10.6.0` | (not installed) | (N/A) | N/A |

Notes:
- `ctranslate2`, `optimum`, `einops`, `timm`, `diskcache`, `soundfile`, `torchvision`, `colpali-engine` are optional extras not installed in the baseline venv. Their PyPI versions are shown for awareness.
- `numpy<2` pin is declared "for onnx" at `pyproject.toml:17`. The baseline resolves numpy 1.26.4 while latest is 2.4.6 (1 major behind).
- `huggingface_hub>=0.32.0` has no upper pin. Resolved 0.36.2, latest 1.19.0 (1 major behind). The upstream fix commit 2ecb218 on `ik-main` after tag 0.0.77 touches `HfFolder().get_token()` -> `get_token()` in `transformer/utils_optimum.py:212`.
- `uvicorn ^0.32.0` caps at <0.33. Latest 0.49.0 means the pin prevents upgrading.

---

## 2. Backend Inventory

### 2a. Engine/Device Enums

`infinity_emb/primitives.py:95-105`: `InferenceEngine` enum defines: `torch`, `ctranslate2`, `optimum`, `neuron`, `debugengine`.

`infinity_emb/primitives.py:107-122`: `Device` enum defines: `cpu`, `cuda`, `mps`, `tensorrt`, `auto`.

### 2b. Torch Backend (primary)

- **Entry point**: `infinity_emb/transformer/utils.py:31` maps `EmbedderEngine.torch = SentenceTransformerPatched`.
- **SentenceTransformerPatched** (`transformer/embedder/sentence_transformer.py:54-169`): 169 LOC. Loads via `sentence_transformers.SentenceTransformer`, applies BetterTransformer optionally, handles `torch.compile`, int8/fp8 quantization via `quant_interface`.
- **CrossEncoderTorch** (`transformer/crossencoder/torch.py`): 123 LOC. Loaded via `RerankEngine.torch` at `utils.py:54`.
- **SentenceClassifier** (`transformer/classifier/torch.py`): 91 LOC. Loaded via `PredictEngine.torch` at `utils.py:90`.
- **TIMM** (`transformer/vision/torch_vision.py`): 244 LOC. Loaded via `ImageEmbedEngine.torch` at `utils.py:68`. Supports `torchvision` and `colpali-engine`.
- **TorchAudioModel** (`transformer/audio/torch.py`): 145 LOC. Loaded via `AudioEmbedEngine.torch` at `utils.py:79`.
- **Loading strategy**: `inference/loading_strategy.py:28-100` (`get_loading_strategy_torch`) resolves device (cuda/npu/mps/cpu), dtype, and quantization dtype. CPU path at line 58-60. MPS path at lines 54-57.
- **Total torch path LOC**: ~843 lines (sentence_transformer + crossencoder/torch + classifier/torch + vision/torch_vision + audio/torch).

### 2c. Optimum/ONNX Backend

- **Entry point**: `transformer/utils.py:34` maps `EmbedderEngine.optimum = OptimumEmbedder`.
- **OptimumEmbedder** (`transformer/embedder/optimum.py:36-115`): 115 LOC. Uses `ORTModelForFeatureExtraction`.
- **OptimumCrossEncoder** (`transformer/crossencoder/optimum.py:28-91`): 91 LOC. Uses `ORTModelForSequenceClassification`.
- **OptimumClassifier** (`transformer/classifier/optimum.py:29-90`): 90 LOC. Uses `ORTModelForSequenceClassification` via `pipeline`.
- **Utils** (`transformer/utils_optimum.py:1-253`): 253 LOC. ONNX file discovery, model optimization, ROCm/OpenVINO/CUDA/TensorRT execution provider selection (`device_to_onnx` at lines 50-84).
- **Total optimum path LOC**: ~549 lines.
- **Removability**: These files are cleanly isolated behind `CHECK_ONNXRUNTIME`/`CHECK_OPTIMUM` guards. They can be removed without touching the torch code path, though the `utils.py` enum wiring would need updates.

### 2d. CTranslate2 Backend

- **Entry point**: `transformer/utils.py:32` maps `EmbedderEngine.ctranslate2 = CT2SentenceTransformer`.
- **CT2SentenceTransformer** (`transformer/embedder/ct2.py:38-89`): extends `SentenceTransformerPatched`, replaces the transformer module with `CT2Transformer`.
- **CT2Transformer** (`transformer/embedder/ct2.py:91-181`): wraps `ctranslate2.Encoder`. Deprecated with a warning at line 113.
- **Total CT2 path LOC**: 181 lines.
- **Removability**: Removable without touching torch code. `ct2.py` depends on `sentence_transformer.py` via inheritance.

### 2e. AWS INF2/Neuron Backend

- **Entry point**: `transformer/utils.py:35` maps `EmbedderEngine.neuron = NeuronOptimumEmbedder`.
- **NeuronOptimumEmbedder** (`transformer/embedder/neuron.py:80-161`): 161 LOC. Uses `optimum.neuron.NeuronModelForFeatureExtraction`. Requires `neuron-ls` CLI for core count detection (line 37-46).
- **Loading strategy**: `inference/loading_strategy.py:9` imports `is_torch_npu_available` (NPU is the Neuron device path).
- **Removability**: Removable without touching torch code. Guarded by `CHECK_OPTIMUM_NEURON`.

### 2f. ROCm Backend

- **No separate code path.** ROCm is handled as a CUDA execution provider in the ONNX path (`transformer/utils_optimum.py:59-60`: `ROCMExecutionProvider`) and as a pytorch wheel source in `pyproject.toml:159-162` (`pytorch_rocm` source for `download.pytorch.org/whl/rocm6.1`).
- **In the torch path**, ROCm uses the standard CUDA device path (`loading_strategy.py:44-47`).
- **Removability**: The explicit ROCm source in pyproject.toml can be removed without code changes. The ONNX ROCm provider handling is in `utils_optimum.py`.

### 2g. MPS Backend

- **Device enum**: `primitives.py:110` defines `Device.mps`.
- **Loading strategy**: `loading_strategy.py:54-57` handles MPS device placement.
- **BetterTransformer**: `acceleration.py:53-58` explicitly skips BetterTransformer on MPS.
- **ONNX**: `utils_optimum.py:65` maps MPS to `CoreMLExecutionProvider`.
- **Removability**: MPS is woven through the torch path. Not removable without touching torch code.

### 2h. Plain CPU Backend

- **Loading strategy**: `loading_strategy.py:58-60` returns `["cpu"] * max(len(args.device_id), 1)`.
- **ONNX**: `utils_optimum.py:54-57` maps CPU to `CPUExecutionProvider` (or `OpenVINOExecutionProvider` if available).
- **Quantization on CPU**: `quantization/interface.py:38-45` uses `torch.quantization.quantize_dynamic` for CPU int8.
- **Removability**: CPU is the default fallback. Not removable.

### 2i. BetterTransformer (acceleration)

- **File**: `transformer/acceleration.py:1-97` (97 LOC).
- **Imports** `optimum.bettertransformer.BetterTransformer` and `BetterTransformerManager` at lines 11-14.
- **check_if_bettertransformer_possible** at line 35: checks `config.model_type in BetterTransformerManager.MODEL_MAPPING`.
- **to_bettertransformer** at line 49: applies `BetterTransformer.transform(model)`.
- **Called from** `sentence_transformer.py:62-63` and `sentence_transformer.py:96-100`.

---

## 3. Import-Time Cost

Command: `/opt/forks/infinity-emb/.venv-baseline/bin/python -X importtime -c "import infinity_emb" 2>importtime.log`

Top 10 cumulative import offenders (microseconds, approximate; other processes may briefly share CPU):

| Cumulative (us) | Package |
|---|---|
| 4,443,301 | `infinity_emb` (total) |
| 3,938,754 | `infinity_emb.engine` |
| 3,938,269 | `infinity_emb.inference` |
| 3,937,419 | `infinity_emb.inference.batch_handler` |
| 3,932,714 | `infinity_emb.transformer.utils` |
| 3,876,641 | `infinity_emb.transformer.audio.torch` |
| 3,869,995 | `infinity_emb.transformer.abstract` |
| 3,869,499 | `infinity_emb.transformer.quantization.interface` |
| 2,671,994 | `sentence_transformers.quantization` |
| 2,671,970 | `sentence_transformers` |

The dominant cost is `sentence_transformers` and its transitive imports (`transformers`, `torch`, `huggingface_hub`). The import chain is: `infinity_emb` -> `engine` -> `inference` -> `batch_handler` -> `transformer.utils` -> `transformer.abstract` -> `quantization.interface` -> `sentence_transformers.quantization`. Nearly all 4.4s is spent in `sentence_transformers` and its dependencies.

---

## 4. SIGTERM Handling

Trace for `infinity_emb v2` entrypoint:

1. **CLI entry**: `cli.py:372` calls `uvicorn.run(app, ...)` with `http="httptools"` and `loop=loopname`.

2. **uvicorn lifecycle**: uvicorn installs its own signal handlers. On SIGTERM, uvicorn initiates its shutdown sequence, which triggers the ASGI lifespan shutdown.

3. **Lifespan handler**: `infinity_server.py:81-117` defines the `lifespan` async context manager:
   - **Startup** (lines 83-104): creates `AsyncEngineArray`, starts telemetry thread, calls `app.engine_array.astart()`.
   - **Shutdown** (line 116): `await app.engine_array.astop()`.

4. **Engine stop**: `engine.py:99-106` (`AsyncEmbeddingEngine.astop`): sets `self.running = False`, calls `await self._batch_handler.shutdown()`.

5. **BatchHandler shutdown**: `batch_handler.py:471-480` (`BatchHandler.shutdown`):
   - Sets `self._shutdown` threading event (line 477).
   - Calls `await asyncio.to_thread(self._threadpool.shutdown)` (line 478), which waits for all model worker threads to finish.
   - Cancels the `_push_task` async task (line 480).

6. **ModelWorker threads**: Each worker thread (`_preprocess_batch`, `_core_batch`, `_postprocess_batch`) checks `self._shutdown.is_set()` in tight loops with `QUEUE_TIMEOUT=0.5s` sleeps. Once the shutdown event fires, threads exit within one timeout cycle.

7. **AsyncEngineArray stop**: `engine.py:316-319` iterates over all engines and calls `astop()` on each.

**Verdict**:
- **Graceful shutdown**: Yes. uvicorn's SIGTERM triggers lifespan shutdown, which calls `astop()` -> `shutdown()`, which signals all threads via the shutdown event and waits for threadpool completion.
- **Model weights release**: The model objects (`SentenceTransformer`, etc.) are held as instance attributes on `ModelWorker._model`. They are released when the `AsyncEngineArray` and its engines are garbage collected after the lifespan context exits. There is no explicit `del model` or `torch.cuda.empty_cache()` call.
- **CUDA contexts**: No explicit CUDA context cleanup. On CUDA, the process exit will release CUDA memory. On CPU (this box), N/A.
- **Potential issue**: If a model worker thread is stuck in `encode_core` (e.g., a very long forward pass), the shutdown waits up to 0.5s per queue operation but the thread itself may block indefinitely in torch inference. The `_shutdown` event is only checked between queue operations, not during the forward pass itself (`batch_handler.py:576` calls `self._model.encode_core(feat)` without checking shutdown).

---

## 5. Dead Code Candidates

Ranked safest first:

### 5a. Commented-out OpenVINO dependencies (SAFEST)

`pyproject.toml:41-45`: Four commented-out openvino deps:
```toml
# optimum-intel = {version=">=1.20.0", optional=true, extras=["openvino"]}
# onnxruntime-openvino = {version=">=1.19.0", optional=true}
# openvino = {version="2024.4.0", optional=true}
# openvino-tokenizers = {version="2024.4.0.0", optional=true}
```
Also `pyproject.toml:111`: `# openvino=["onnxruntime-openvino","openvino","openvino-tokenizers"]`.
These are fully commented out and have no code references beyond `utils_optimum.py` checking for `OpenVINOExecutionProvider` at runtime (lines 55-56, 79-80). Safe to delete the comments.

### 5b. GPTQ quantization code (SAFE)

`transformer/quantization/quant.py:40-41` has commented-out imports:
```python
# from infinity_emb.transformer.quantization.GPTQ import GenericGPTQRunner, InputRecorder
# from infinity_emb.transformer.quantization.eval import get_task_dict, evaluate, lm_eval
```
The `GPTQQuantHandler` class (lines 201-350) references `GenericGPTQRunner` and `InputRecorder` which are never imported. The `int4-gptq` mode in `quantize()` (lines 697-722) calls these undefined names and would crash at runtime. This is dead code. The `GPTQQuantHandler` class and the `int4-gptq` branch (734 LOC total file, ~200 LOC GPTQ-specific) are unreachable.

### 5c. DummyTransformer / debugengine (SAFE for production)

`transformer/embedder/dummytransformer.py`: 32 LOC. Referenced only by `InferenceEngine.debugengine` at `utils.py:33,43-44` and `select_model.py:27-28`. Used exclusively in tests (`tests/end_to_end/test_api_with_dummymodel.py`, `tests/unit_test/test_engine.py`, `tests/unit_test/test_sync_engine.py`). Safe to remove from production code if tests are updated.

### 5d. INFINITY_DISABLE_OPTIMUM env var check (SAFE)

`transformer/acceleration.py:61-66`: Checks `os.environ.get("INFINITY_DISABLE_OPTIMUM", False)` and logs a deprecation error. The comment at line 62 says "TODO: remove this code path". No functional effect.

### 5e. sync_engine.py (MODERATE)

`sync_engine.py`: 238 LOC. Provides `SyncEngineArray` for synchronous usage. Referenced from `__init__.py:13,25` (exported) and tests (`test_sync_engine.py`). This is public API (exported in `__init__.py`). Not dead, but potentially unused in production server deployments.

### 5f. huggingface_hub HfFolder import (SAFE to fix)

`transformer/utils_optimum.py:8`: imports `HfFolder` from `huggingface_hub`. Used at line 212: `HfFolder().get_token()`. This is deprecated in huggingface_hub >= 1.0 (the fix commit 2ecb218 replaces it with `get_token()`). This is the known breakage evidence.

---

## 6. Risk List for Dependency Modernization

### 6a. torch >= 2.5: BetterTransformer is dead

`transformer/acceleration.py:11-14` imports `optimum.bettertransformer.BetterTransformer` and `BetterTransformerManager`. In transformers >= 4.49, BetterTransformer was removed/deprecated in favor of `torch.nn.functional.scaled_dot_product_attention`. The `optimum` package also removed BetterTransformer support in newer versions.

- `acceleration.py:46`: `config.model_type in BetterTransformerManager.MODEL_MAPPING` will fail with `AttributeError`.
- `acceleration.py:83`: `BetterTransformer.transform(model)` will fail.
- **Impact**: The `bettertransformer` CLI flag (default True per `env.py:147`) will cause startup failures.
- **Fix**: Remove or gate the BetterTransformer code path.

### 6b. transformers >= 4.57: torch_dtype deprecation

At runtime, `sentence-transformers` / `transformers` emits a `FutureWarning` about `torch_dtype` parameter. Used at:
- `transformer/embedder/sentence_transformer.py:70`: `model_kwargs["torch_dtype"] = ls.loading_dtype`
- `transformer/classifier/torch.py:36`: `model_kwargs["torch_dtype"] = ls.loading_dtype`
- `transformer/crossencoder/torch.py:56`: `model_kwargs["torch_dtype"] = ls.loading_dtype`
- `transformer/vision/torch_vision.py:56,58`: `extra_model_args["torch_dtype"] = ...`

In transformers >= 5.x, `torch_dtype` may be removed. The replacement is `dtype` parameter.

### 6c. numpy < 2 pin

`pyproject.toml:17`: `numpy = ">=1.20.0,<2"`. Comment says "pin numpy <2 for onnx". The baseline resolves numpy 1.26.4. numpy 2.x broke backwards compatibility for C extensions. If onnxruntime/ctranslate2/optimum are not used, this pin can be relaxed. But if any ONNX code path is needed, this pin must stay until onnxruntime ships numpy 2 wheels.

### 6d. uvicorn ^0.32 cap

`pyproject.toml:25`: `uvicorn = {version = "^0.32.0", ...}`. The `^` means >=0.32.0, <0.33.0. Latest uvicorn is 0.49.0. The pin prevents upgrading to any newer uvicorn. This cap should be relaxed (e.g., `>=0.32.0`).

### 6e. pydantic v1/v2 compat

`fastapi_schemas/pydantic_v2.py`: Imports `pydantic.AnyUrl`, `pydantic.HttpUrl`, `pydantic.StringConstraints` (all pydantic v2 APIs). The `args.py:115-125` uses `pydantic.dataclasses.dataclass` with `ConfigDict` (v2 only). The project targets `pydantic >=2.4.0,<3`. No v1 compat shims exist in the codebase. This is clean for pydantic v2. Risk: if pydantic v3 appears, the `ConfigDict` and `StringConstraints` APIs may change.

### 6f. posthog / telemetry API drift

`telemetry.py:237-241` creates `Posthog(project_api_key=..., host=..., disabled=...)`. The `posthog` package API is generally stable. The baseline resolves posthog 7.18.3 which matches latest. Low risk.

### 6g. huggingface_hub >= 1.0 breaking changes

**Known breakage**: commit 2ecb218 (on `ik-main` after tag 0.0.77) fixes exactly this:
- `transformer/utils_optimum.py:8`: `from huggingface_hub import HfApi, HfFolder` -> `from huggingface_hub import HfApi, get_token`
- `transformer/utils_optimum.py:212`: `HfFolder().get_token()` -> `get_token()`

`HfFolder` was removed in huggingface_hub 1.0. The fix commit touches only `transformer/utils_optimum.py` (2 lines changed).

Additional risk areas for huggingface_hub >= 1.0:
- `transformer/embedder/ct2.py:11-13`: imports `HUGGINGFACE_HUB_CACHE` from `huggingface_hub.constants`. This constant may change.
- `transformer/utils_optimum.py:9`: imports `HUGGINGFACE_HUB_CACHE`. Same risk.
- `inference/select_model.py:34`: imports `hf_hub_download`. API likely stable but should verify.

### 6h. sentence-transformers ^3.0.1 -> 5.5.1

`pyproject.toml:32`: `sentence-transformers = {version = "^3.0.1"}`. Latest is 5.5.1 (2 majors behind). The `sentence_transformers.quantization` module is imported at `quantization/interface.py:25`. If the quantization API changed between v3 and v5, this will break. The `sentence_transformers.SentenceTransformer` and `sentence_transformers.util.batch_to_device` APIs used in `sentence_transformer.py:34,123` should be verified.

### 6i. optimum >= 1.24.0 -> 2.2.0

`pyproject.toml:35`: `optimum = {version = ">=1.24.0"}`. Optimum 2.x had breaking changes. The code uses `optimum.bettertransformer` (removed in 2.x), `optimum.onnxruntime.ORTModel*` classes, and `optimum.onnxruntime.configuration.OptimizationConfig`. These should be verified against optimum 2.x APIs.

### 6j. transformers <= 5.0 -> 5.12.0

`pyproject.toml:33`: `transformers = {version = ">=4.47.0,<=5.0"}`. The upper bound `<=5.0` prevents upgrading to 5.12.0. This cap must be relaxed. Key risk: `torch_dtype` parameter deprecation (see 6b), and `AutoConfig.from_pretrained` API stability.

---

## Claims I did not verify

1. Whether `uvicorn.run()` with `http="httptools"` installs SIGTERM handlers that conflict with the lifespan handler. I assumed standard uvicorn behavior.
2. Whether `torch.cuda.empty_cache()` is needed for CUDA memory release on shutdown. I noted its absence but did not verify if Python GC is sufficient.
3. Whether the `ctranslate2`, `optimum`, `timm`, `colpali-engine`, `soundfile`, or `torchvision` packages have numpy 2 compatibility.
4. Whether `sentence_transformers` v5.x changed the `quantize_embeddings` API used at `quantization/interface.py:146`.
5. Whether `pydantic` v3 (if it ships) will break the `ConfigDict` or `StringConstraints` usage.
6. Whether the `prometheus-fastapi-instrumentator` v8 API changed from v6.
7. Whether `onnxruntime-gpu` 1.19.x (pinned at `pyproject.toml:55`) has any remaining compatibility with the optimum 2.x API.
8. Whether the `ReRankReturnType` dataclass at `primitives.py:54-57` conflicts with the `ReRankReturnType = float` alias at line 65 (both names coexist; the dataclass is used in `batch_handler.py:213`).
9. Whether `openvino` execution provider paths in `utils_optimum.py` are tested or bitrotted.
10. Whether `float8_experimental` (referenced at `quantization/interface.py:54-65`) is compatible with torch >= 2.5 (float8 support was added to torch core).
