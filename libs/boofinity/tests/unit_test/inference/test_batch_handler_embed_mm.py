# SPDX-License-Identifier: MIT
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest


class _Item:
    """Minimal stand-in for an MMEmbeddingInput item (text-only avoids image IO)."""

    def __init__(self, text=None, image=None):
        self.text = text
        self.image = image


@pytest.fixture
def embed_handler():
    import threading

    from boofinity.inference.batch_handler import BatchHandler

    bh = BatchHandler.__new__(BatchHandler)
    bh.model_worker = [MagicMock()]
    bh.model_worker[0].capabilities = {"embed", "image_embed"}
    bh._threadpool = ThreadPoolExecutor(max_workers=1)
    bh._shutdown = threading.Event()

    async def _fake_schedule(singles):
        return [np.arange(1024, dtype=np.float32) for _ in singles], 7

    bh._schedule = _fake_schedule
    return bh


@pytest.mark.anyio
async def test_embed_mm_applies_matryoshka_dim(embed_handler):
    embs, usage = await embed_handler.embed_mm(items=[_Item(text="a")], matryoshka_dim=64)
    assert len(embs) == 1
    assert embs[0].shape[0] == 64
    assert usage == 7


@pytest.mark.anyio
async def test_embed_mm_none_returns_full_dim(embed_handler):
    embs, _ = await embed_handler.embed_mm(items=[_Item(text="a")], matryoshka_dim=None)
    assert embs[0].shape[0] == 1024


@pytest.mark.anyio
async def test_embed_mm_out_of_range_raises(embed_handler):
    from boofinity.primitives import MatryoshkaDimError

    with pytest.raises(MatryoshkaDimError):
        await embed_handler.embed_mm(items=[_Item(text="a")], matryoshka_dim=2048)
