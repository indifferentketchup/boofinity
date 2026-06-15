# Deploying boofinity (bge-m3 + a reranker)

Validated on CPU on 2026-06-15: `boofinity v2` boots `BAAI/bge-m3` plus a reranker
and serves `/embeddings` and `/rerank` end to end. CUDA is supported but never
required; the CPU path is fully functional with no CUDA present.

## What you serve

- Embedder: `BAAI/bge-m3` (1024-dim).
- Reranker (pick one; both auto-detected):
  - Classic cross-encoder, e.g. `mixedbread-ai/mxbai-rerank-xsmall-v1`.
  - CausalLM reranker, e.g. `Qwen/Qwen3-Reranker-0.6B` (detected via its
    `1_LogitScore` module and routed to the `causal_lm` backend).

## 1. Install

The package lives in `libs/boofinity`. Install torch FIRST from the index that
matches the box, then the package with the `[torch,server,logging]` extras.

### Linux, CPU only

```bash
cd ~/boofinity
python3 -m venv .venv && .venv/bin/pip install -U pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e "libs/boofinity[torch,server,logging]"
```

### Linux, NVIDIA P104 (Pascal, compute capability 6.1) - 100.90.172.55

Use a CUDA wheel that still ships sm_61 kernels (cu124 does as of torch 2.6.x;
the very newest cu128 builds may have dropped Pascal):

```bash
cd ~/boofinity
python3 -m venv .venv && .venv/bin/pip install -U pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -e "libs/boofinity[torch,server,logging]"
# Verify the GPU is usable BEFORE serving:
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(torch.zeros(8, device='cuda').sum())"
```

If `cuda.is_available()` is False, or the forward pass raises `CUDA error: no
kernel image is available`, that torch build dropped sm_61. Install an older
Pascal-capable build (e.g. `pip install "torch==2.4.*" --index-url
https://download.pytorch.org/whl/cu121`) or fall back to `--device cpu`.

On Pascal also serve with `--dtype float32`. The default `auto` dtype resolves to
fp16 on any GPU without bf16 (`loading_strategy.py`), and fp16 is pathologically
slow on Pascal, so fp32 is both faster and matches the validated CPU parity. (A
compute-capability-aware default is planned in the `gpu-multistack-acceleration`
change.)

### Windows, RTX 5090 (Blackwell, compute capability 12.0) - 100.101.41.16 (D:\boofinity)

Blackwell sm_120 needs torch >= 2.7 built against CUDA 12.8 (cu128):

```powershell
cd D:\boofinity
py -m venv .venv
.venv\Scripts\pip install -U pip
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -e "libs/boofinity[torch,server,logging]"
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(torch.zeros(8, device='cuda').sum())"
```

(`uvloop` is skipped automatically on Windows; that is expected and harmless.)

## 2. Serve

`boofinity v2` accepts repeated `--model-id` flags (one process, multiple models):

```bash
# Linux (CPU shown; use --device cuda on a GPU box)
.venv/bin/boofinity v2 \
  --model-id BAAI/bge-m3 \
  --model-id Qwen/Qwen3-Reranker-0.6B \
  --device cpu \
  --port 7997
```

```powershell
# Windows (5090)
.venv\Scripts\boofinity v2 --model-id BAAI/bge-m3 --model-id Qwen/Qwen3-Reranker-0.6B --device cuda --port 7997
```

Notes:

- The server accepts connections only after every model finishes loading (uvicorn
  defers serving until the lifespan completes), so a 200 on `/health` means ready.
- Per-model boolean flags (`--trust-remote-code`, `--model-warmup`, ...) take the
  first configured value across all models in one process. Run one model per
  process (the llama-swap child pattern) if you need them to differ per model.
- To force the CausalLM reranker for a repo that lacks an auto-detect marker, set
  `INFINITY_RERANK_MODE=causal_lm`.

## 3. Smoke test

```bash
curl -s localhost:7997/health        # 200 when ready
curl -s localhost:7997/models        # lists both model ids

curl -s localhost:7997/embeddings -H 'content-type: application/json' \
  -d '{"model":"BAAI/bge-m3","input":["hello world"]}'

curl -s localhost:7997/rerank -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-Reranker-0.6B","query":"capital of France?","documents":["Paris is the capital of France.","Bananas are yellow."]}'
```

Expected: embeddings of length 1024; the rerank ranks the relevant document first
(on CPU the Qwen3 reranker scored the Paris document 0.997 vs ~0.0 for the
distractor during validation).
