"""Capture the bge-m3 embedding baseline.

Embeds tests/parity/fixtures/inputs.json with boofinity's Python API
(torch engine) and writes a baseline fixture file.

Default (no flags): CPU float32, writes fixtures/baseline_bge-m3_cpu.npz
(frozen file, not rewritten when device=cpu dtype=float32).

With --device/--dtype: writes fixtures/baseline_bge-m3_<device>_<dtype>.npz.
On CUDA, --dtype auto resolves to bf16 or fp16 per loading_strategy.py logic.
"""

import argparse
import sys

import numpy as np

from common import (
    MODEL_ID,
    embed_all,
    embed_all_device,
    freeze_hash,
    load_inputs,
    resolve_dtype,
    fixture_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture embedding baseline.")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="Device (default: cpu)",
    )
    parser.add_argument(
        "--dtype", default=None,
        help="Dtype: float32, float16, bfloat16, or auto (default: auto, resolves per device)",
    )
    args = parser.parse_args()

    device = args.device
    dtype = resolve_dtype(device, args.dtype)
    out_path = fixture_path(device, dtype)

    import boofinity

    texts = load_inputs()
    print(f"embedding {len(texts)} inputs with {MODEL_ID} on {device} dtype={dtype} ...", flush=True)
    embeddings = embed_all_device(texts, device=device, dtype=dtype)
    print(f"embeddings shape: {embeddings.shape}")

    np.savez(
        out_path,
        embeddings=embeddings,
        boofinity_version=np.array(boofinity.__version__),
        freeze_sha256=np.array(freeze_hash()),
        model_id=np.array(MODEL_ID),
        device=np.array(device),
        dtype=np.array(dtype),
        n_inputs=np.array(len(texts)),
    )
    print(f"wrote {out_path}")
    print(f"boofinity version: {boofinity.__version__}")
    print(f"freeze sha256: {freeze_hash()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
