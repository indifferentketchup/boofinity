"""Shared helpers for the parity harness. CPU-only by design."""

import asyncio
import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("INFINITY_ANONYMOUS_USAGE_STATS", "0")

PARITY_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = PARITY_DIR / "fixtures"
INPUTS_PATH = FIXTURES_DIR / "inputs.json"
BASELINE_PATH = FIXTURES_DIR / "baseline_bge-m3_cpu.npz"
REPO_ROOT = PARITY_DIR.parent.parent
FREEZE_PATH = REPO_ROOT / "baseline-freeze.txt"

MODEL_ID = "BAAI/bge-m3"


def load_inputs() -> list[str]:
    with open(INPUTS_PATH, encoding="utf-8") as f:
        texts = json.load(f)
    assert isinstance(texts, list) and all(isinstance(t, str) for t in texts)
    return texts


def freeze_hash() -> str:
    if not FREEZE_PATH.exists():
        return "missing"
    return hashlib.sha256(FREEZE_PATH.read_bytes()).hexdigest()


async def _embed_async(texts: list[str]) -> np.ndarray:
    from infinity_emb import AsyncEngineArray, EngineArgs

    engine_args = EngineArgs(
        model_name_or_path=MODEL_ID,
        engine="torch",
        device="cpu",
        dtype="float32",
        bettertransformer=False,
        compile=False,
        model_warmup=False,
        batch_size=8,
    )
    array = AsyncEngineArray.from_args([engine_args])
    engine = array[MODEL_ID]
    await engine.astart()
    try:
        embeddings, _usage = await engine.embed(sentences=texts)
    finally:
        await engine.astop()
    return np.stack([np.asarray(e, dtype=np.float32) for e in embeddings])


def embed_all(texts: list[str]) -> np.ndarray:
    return asyncio.run(_embed_async(texts))


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-12)
