"""Performance benchmark for boofinity.

Benchmarks the engine IN-PROCESS using AsyncEngineArray/AsyncEmbeddingEngine
Python API rather than a running v2 HTTP server. Rationale: measuring in-process
avoids HTTP serialization, network, and uvicorn overhead, giving a direct reading
of the engine pipeline (tokenizer, model forward, batching) that is the component
being optimized. The parity harness already uses this pattern, so results are
comparable and setup is simpler. HTTP-layer benchmarks (uvicorn throughput, TLS,
connection pooling) are a separate concern and should be measured with a dedicated
HTTP load-testing tool.

Reports:
  - p50/p95 single-call embed latency (all inputs as one call)
  - Batch throughput at sizes [1, 8, 32] (sentences/sec)
  - Cold import time (subprocess running python -X importtime)

Usage:
  python run_bench.py --device cpu
  python run_bench.py --device cpu --json-out baseline_cpu.json
  python run_bench.py --device cuda  # SKIPPED on CPU-only box
"""

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("INFINITY_ANONYMOUS_USAGE_STATS", "0")

PARITY_DIR = Path(__file__).resolve().parent.parent / "parity"
FIXTURES_DIR = PARITY_DIR / "fixtures"
INPUTS_PATH = FIXTURES_DIR / "inputs.json"

WARMUP_ITERS = 1
LATENCY_ITERS = 3
BATCH_WARMUP_ITERS = 1
BATCH_ITERATIONS = 2
COLD_IMPORT_REPEATS = 3

MODEL_ID_DEFAULT = "BAAI/bge-m3"
BATCH_SIZES = [1, 8, 32]


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _cold_import_time_us() -> float:
    """Measure cold import time via subprocess. Returns microseconds.

    Clears __pycache__ directories before measuring so the result reflects a
    true cold start (no bytecode cache). Parses the importtime format:
        import time:    <self_us> | <cumulative_us> | <module>
    We take the cumulative_us for the top-level ``import boofinity`` line.
    """
    import shutil

    # Find and remove __pycache__ dirs under the boofinity package to
    # ensure a cold import (no bytecode cache).
    pkg_dir = Path(__file__).resolve().parent.parent.parent / "libs" / "boofinity" / "boofinity"
    if pkg_dir.is_dir():
        for pycache in pkg_dir.rglob("__pycache__"):
            shutil.rmtree(pycache, ignore_errors=True)

    cmd = [
        sys.executable, "-B", "-X", "importtime", "-c", "import boofinity",
    ]
    times = []
    for _ in range(COLD_IMPORT_REPEATS):
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        for line in result.stderr.splitlines():
            stripped = line.strip()
            if "import time:" in stripped and "boofinity" in stripped:
                parts = stripped.split("|")
                if len(parts) >= 2:
                    cumul_part = parts[1].strip()
                    try:
                        times.append(int(cumul_part))
                    except ValueError:
                        pass
                break
    if not times:
        return -1.0
    return statistics.median(times)


def _load_inputs() -> list[str]:
    with open(INPUTS_PATH, encoding="utf-8") as f:
        texts = json.load(f)
    return texts


def _make_engine_args(device: str, dtype: str, model_id: str, batch_size: int = 32):
    from boofinity import EngineArgs
    return EngineArgs(
        model_name_or_path=model_id,
        engine="torch",
        device=device,
        dtype=dtype,
        bettertransformer=False,
        compile=False,
        model_warmup=False,
        batch_size=batch_size,
    )


def _run_bench(args) -> dict:
    """Run all benchmarks with a single engine lifecycle. Returns result dict."""
    from boofinity import AsyncEngineArray

    device = args.device
    model_id = args.model_id

    if device == "cuda":
        from boofinity.hardware.provider_policy import provider_plan
        from boofinity.hardware.capability import HardwareCapability

        # Build a minimal capability for the policy. Physical GPU detection
        # may not have run; the policy falls back to torch-view GPUs.
        cap = HardwareCapability(
            os_name="Linux",
            machine_arch="x86_64",
            torch_available=True,
            cuda_available=True,
            onnxruntime_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        plan = provider_plan(cap)
        dtype = plan.dtype
    else:
        dtype = "float32"

    texts = _load_inputs()
    engine_args = _make_engine_args(device, dtype, model_id)

    print(f"device: {device}")
    print(f"dtype:  {dtype}")
    print(f"model:  {model_id}")
    print(f"python: {sys.executable}")
    print(f"venv:   {sys.prefix}")
    print(f"inputs: {len(texts)} texts")

    # Cold import (subprocess, separate from engine)
    print(f"\n--- Cold import time (subprocess, {COLD_IMPORT_REPEATS} repeats) ---", flush=True)
    cold_us = _cold_import_time_us()
    if cold_us > 0:
        cold_s = cold_us / 1_000_000
        print(f"cold import: {cold_s:.3f}s (median of {COLD_IMPORT_REPEATS})")
    else:
        cold_s = None
        print("cold import: FAILED to parse")

    # Engine benchmarks (single lifecycle)
    print(f"\n--- Engine benchmarks ---", flush=True)

    async def _engine_bench():
        array = AsyncEngineArray.from_args([engine_args])
        engine = array[model_id]
        await engine.astart()
        try:
            # Warmup
            for _ in range(WARMUP_ITERS):
                await engine.embed(sentences=texts)
            print("warmup done", flush=True)

            # Latency
            print(
                f"\n--- Single-call latency "
                f"({LATENCY_ITERS} iters, {WARMUP_ITERS} warmup) ---",
                flush=True,
            )
            latencies = []
            for _ in range(LATENCY_ITERS):
                t0 = time.perf_counter()
                await engine.embed(sentences=texts)
                t1 = time.perf_counter()
                latencies.append(t1 - t0)
            p50 = statistics.median(latencies)
            p95 = (
                sorted(latencies)[int(len(latencies) * 0.95)]
                if len(latencies) >= 20
                else max(latencies)
            )
            print(
                f"p50: {p50*1000:.1f}ms  p95: {p95*1000:.1f}ms  "
                f"(n={len(latencies)}, inputs={len(texts)})"
            )

            # Batch throughput
            print(
                f"\n--- Batch throughput "
                f"({BATCH_ITERATIONS} iters, {BATCH_WARMUP_ITERS} warmup) ---",
                flush=True,
            )
            throughputs = {}
            for bs in BATCH_SIZES:
                batch = texts[:bs] if len(texts) >= bs else (texts * (bs // len(texts) + 1))[:bs]
                for _ in range(BATCH_WARMUP_ITERS):
                    await engine.embed(sentences=batch)
                bt_times = []
                for _ in range(BATCH_ITERATIONS):
                    t0 = time.perf_counter()
                    await engine.embed(sentences=batch)
                    t1 = time.perf_counter()
                    bt_times.append(t1 - t0)
                avg = statistics.mean(bt_times)
                tps = bs / avg if avg > 0 else 0.0
                throughputs[bs] = tps
                print(f"batch={bs:>3d}: {tps:.1f} sentences/sec")

            return {
                "cold_import_us": cold_us,
                "cold_import_seconds": cold_s,
                "latency_p50_s": p50,
                "latency_p95_s": p95,
                "latency_n": len(latencies),
                "batch_throughput": {str(k): v for k, v in throughputs.items()},
            }
        finally:
            await engine.astop()

    bench_data = asyncio.run(_engine_bench())

    return {
        "device": device,
        "dtype": dtype,
        "model_id": model_id,
        "python": sys.executable,
        "venv": sys.prefix,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs_count": len(texts),
        "warmup_iters": WARMUP_ITERS,
        "latency_iters": LATENCY_ITERS,
        "batch_warmup_iters": BATCH_WARMUP_ITERS,
        "batch_iterations": BATCH_ITERATIONS,
        "cold_import_repeats": COLD_IMPORT_REPEATS,
        **bench_data,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark boofinity engine.")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="Device to benchmark on (default: cpu)",
    )
    parser.add_argument(
        "--model-id", default=MODEL_ID_DEFAULT,
        help=f"Model to benchmark (default: {MODEL_ID_DEFAULT})",
    )
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="Path to write JSON output",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not _cuda_available():
        print("SKIPPED: --device cuda requested but CUDA is not available on this box")
        return 0

    print("=== boofinity bench ===")
    print(f"model:  {args.model_id}")

    output = _run_bench(args)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {out_path}")

    print("\n=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
