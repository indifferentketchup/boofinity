"""Check embedding parity against a captured baseline.

Re-embeds tests/parity/fixtures/inputs.json and compares each row against
the matching baseline fixture. Every input must reach cosine >= 0.9999.

Default (no flags): checks against the frozen cpu float32 fixture
(baseline_bge-m3_cpu.npz). Behavior is identical to the pre-batch-0 version.

With --device/--dtype: selects the matching fixture. If the fixture file does
not exist, prints SKIPPED and exits with code 3 (distinct from PASS=0, FAIL=1).

Exit codes:
  0  PASS
  1  FAIL (cosine below threshold)
  2  error (missing baseline, shape mismatch, etc.)
  3  SKIPPED (fixture file not found for the requested device/dtype)
"""

import argparse
import sys

import numpy as np

from common import (
    MODEL_ID,
    cosine_rows,
    embed_all,
    embed_all_device,
    freeze_hash,
    fixture_path,
    load_inputs,
    resolve_dtype,
)

THRESHOLD = 0.9999


def main() -> int:
    parser = argparse.ArgumentParser(description="Check embedding parity.")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Device to check (default: cpu, no flag preserves original behavior)",
    )
    parser.add_argument(
        "--dtype", default=None,
        help="Dtype to check (default: auto, resolves per device)",
    )
    args = parser.parse_args()

    # Preserve original behavior when no flags given: cpu float32, frozen fixture.
    device = args.device if args.device else "cpu"
    dtype = resolve_dtype(device, args.dtype)
    base_path = fixture_path(device, dtype)

    if not base_path.exists():
        capture_cmd = (
            f"cd tests/parity && python capture_baseline.py "
            f"--device {device} --dtype {dtype}"
        )
        print(
            f"SKIPPED: fixture not found at {base_path}\n"
            f"  To create it, run:\n"
            f"    {capture_cmd}"
        )
        return 3

    baseline = np.load(base_path)
    base_emb = baseline["embeddings"]
    base_version = str(baseline["boofinity_version"])
    base_freeze = str(baseline["freeze_sha256"])

    import boofinity

    if str(boofinity.__version__) != base_version:
        print(f"note: boofinity {boofinity.__version__} vs baseline {base_version}")
    if freeze_hash() != base_freeze:
        print(f"note: freeze hash differs from baseline ({freeze_hash()[:12]} vs {base_freeze[:12]})")

    texts = load_inputs()
    if base_emb.shape[0] != len(texts):
        print(f"FAIL: baseline has {base_emb.shape[0]} rows, inputs.json has {len(texts)}")
        return 2

    print(
        f"re-embedding {len(texts)} inputs with {MODEL_ID} "
        f"on {device} dtype={dtype} ...",
        flush=True,
    )
    new_emb = embed_all_device(texts, device=device, dtype=dtype)
    if new_emb.shape != base_emb.shape:
        print(f"FAIL: shape mismatch {new_emb.shape} vs baseline {base_emb.shape}")
        return 2

    cosines = cosine_rows(new_emb, base_emb)
    failures = np.where(cosines < THRESHOLD)[0]

    print(f"min cosine:  {cosines.min():.8f}")
    print(f"mean cosine: {cosines.mean():.8f}")
    if len(failures):
        for i in failures:
            preview = texts[i][:60].replace("\n", "\\n")
            print(f"FAIL input {i}: cosine {cosines[i]:.8f} < {THRESHOLD} | {preview!r}")
        print(f"FAIL: {len(failures)}/{len(texts)} inputs below threshold {THRESHOLD}")
        return 1

    print(f"PASS: all {len(texts)} inputs >= {THRESHOLD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
