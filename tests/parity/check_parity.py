"""Check embedding parity against the captured bge-m3 CPU baseline.

Re-embeds tests/parity/fixtures/inputs.json and compares each row against
fixtures/baseline_bge-m3_cpu.npz. Every input must reach cosine >= 0.9999.
Prints min and mean cosine; exits nonzero on any failure.
"""

import sys

import numpy as np

from common import BASELINE_PATH, MODEL_ID, cosine_rows, embed_all, freeze_hash, load_inputs

THRESHOLD = 0.9999


def main() -> int:
    if not BASELINE_PATH.exists():
        print(f"FAIL: baseline not found at {BASELINE_PATH}; run capture_baseline.py first")
        return 2

    baseline = np.load(BASELINE_PATH)
    base_emb = baseline["embeddings"]
    base_version = str(baseline["infinity_emb_version"])
    base_freeze = str(baseline["freeze_sha256"])

    import infinity_emb

    if str(infinity_emb.__version__) != base_version:
        print(f"note: infinity_emb {infinity_emb.__version__} vs baseline {base_version}")
    if freeze_hash() != base_freeze:
        print(f"note: freeze hash differs from baseline ({freeze_hash()[:12]} vs {base_freeze[:12]})")

    texts = load_inputs()
    if base_emb.shape[0] != len(texts):
        print(f"FAIL: baseline has {base_emb.shape[0]} rows, inputs.json has {len(texts)}")
        return 2

    print(f"re-embedding {len(texts)} inputs with {MODEL_ID} on cpu ...", flush=True)
    new_emb = embed_all(texts)
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
