"""Capture the bge-m3 CPU embedding baseline.

Embeds tests/parity/fixtures/inputs.json with infinity_emb's Python API
(torch engine, CPU, float32) and writes fixtures/baseline_bge-m3_cpu.npz
containing the embeddings, the infinity_emb version, and the sha256 of
baseline-freeze.txt at capture time.
"""

import sys

import numpy as np

from common import BASELINE_PATH, MODEL_ID, embed_all, freeze_hash, load_inputs


def main() -> int:
    import infinity_emb

    texts = load_inputs()
    print(f"embedding {len(texts)} inputs with {MODEL_ID} on cpu ...", flush=True)
    embeddings = embed_all(texts)
    print(f"embeddings shape: {embeddings.shape}")

    np.savez(
        BASELINE_PATH,
        embeddings=embeddings,
        infinity_emb_version=np.array(infinity_emb.__version__),
        freeze_sha256=np.array(freeze_hash()),
        model_id=np.array(MODEL_ID),
        n_inputs=np.array(len(texts)),
    )
    print(f"wrote {BASELINE_PATH}")
    print(f"infinity_emb version: {infinity_emb.__version__}")
    print(f"freeze sha256: {freeze_hash()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
