# Bench: boofinity performance measurement

## What it measures

In-process engine performance using the `AsyncEngineArray` / `AsyncEmbeddingEngine`
Python API. This measures the core pipeline (tokenizer, model forward, batching)
without HTTP, uvicorn, or serialization overhead. The parity harness uses the same
in-process pattern, so results are comparable.

HTTP-layer benchmarks (uvicorn throughput, TLS, connection pooling) are a separate
concern and should be measured with a dedicated load-testing tool (e.g., wrk, k6).

## Metrics reported

- **p50/p95 single-call latency**: time to embed all 50 test inputs in one call.
  Iteration counts are in the JSON output (`latency_iters`, `warmup_iters`).
- **Batch throughput**: sentences/sec at batch sizes [1, 8, 32].
  Iteration counts are in the JSON output (`batch_iterations`, `batch_warmup_iters`).
- **Cold import time**: subprocess measurement of `import boofinity` with
  `python -X importtime`. Cleared of bytecode cache before measurement.

## How to run

Requires `.venv-batch2` at the repo root (or any venv with boofinity installed).

```bash
cd tests/bench

# CPU baseline
../../.venv-batch2/bin/python run_bench.py --device cpu

# CPU baseline with JSON output
../../.venv-batch2/bin/python run_bench.py --device cpu --json-out baseline_cpu.json

# Custom model
../../.venv-batch2/bin/python run_bench.py --device cpu --model-id BAAI/bge-m3
```

## CUDA (manual step on a CUDA host)

This dev box has no GPU. On a CUDA host, run:

```bash
cd tests/bench

# Capture CUDA baseline (auto dtype: bf16 or fp16)
../../.venv-batch2/bin/python run_bench.py --device cuda --json-out baseline_cuda.json

# Also capture the CUDA parity fixture:
cd tests/parity
../../.venv-batch2/bin/python capture_baseline.py --device cuda
```

The parity fixture will be written to
`tests/parity/fixtures/baseline_bge-m3_cuda_<dtype>.npz` where `<dtype>` is
resolved automatically (bf16 if `torch.cuda.is_bf16_supported()`, else fp16).

## Output format

JSON output includes: `device`, `dtype`, `model_id`, `python`, `venv`,
`timestamp_utc`, `inputs_count`, iteration counts, `cold_import_us`,
`cold_import_seconds`, `latency_p50_s`, `latency_p95_s`, `latency_n`,
and `batch_throughput` (dict of batch_size -> sentences/sec).
