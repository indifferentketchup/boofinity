# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest


class TestBatchHandlerMMRerank:
    @pytest.fixture
    def handler_stub(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import MagicMock

        from boofinity.args import EngineArgs
        from boofinity.inference.batch_handler import BatchHandler
        from boofinity.primitives import Device, Dtype, InferenceEngine

        model = MagicMock()
        model.capabilities = {"rerank"}
        model.encode_pre.return_value = [{"input_ids": MagicMock()}]
        model.encode_core.return_value = MagicMock()
        model.encode_post.return_value = [0.1, 0.9, 0.5]
        model.encode_post_raw.return_value = [1.2, -0.5, 0.1]

        bh = BatchHandler.__new__(BatchHandler)
        bh.model_worker = [MagicMock()]
        bh.model_worker[0]._model = model
        bh.model_worker[0].capabilities = {"rerank"}
        bh._threadpool = ThreadPoolExecutor(max_workers=1)
        bh._shutdown = threading.Event()

        return bh, model

    @pytest.mark.anyio
    async def test_mm_rerank_raw_scores_false(self, handler_stub):
        from boofinity.primitives import MMItem

        bh, model = handler_stub
        query = MMItem(text="Where is Munich?")
        documents = [
            MMItem(text="The sky is blue."),
            MMItem(text="Munich is in Germany."),
            MMItem(text="Paris is in France."),
        ]

        results, usage = await bh.mm_rerank(
            query=query, documents=documents, raw_scores=False,
        )

        assert len(results) == 3
        assert results[0].relevance_score == 0.9
        assert results[0].index == 1
        assert results[2].relevance_score == 0.1
        assert results[2].index == 0
        assert usage > 0

    @pytest.mark.anyio
    async def test_mm_rerank_raw_scores_true(self, handler_stub):
        from boofinity.primitives import MMItem

        bh, model = handler_stub
        model.encode_post_raw.return_value = [1.2, -0.5, 0.1]

        query = MMItem(text="Where is Munich?")
        documents = [
            MMItem(text="The sky is blue."),
            MMItem(text="Munich is in Germany."),
            MMItem(text="Paris is in France."),
        ]

        results, usage = await bh.mm_rerank(
            query=query, documents=documents, raw_scores=True,
        )

        assert len(results) == 3
        assert results[0].relevance_score == 1.2
        assert results[0].index == 0

    @pytest.mark.anyio
    async def test_mm_rerank_top_n(self, handler_stub):
        from boofinity.primitives import MMItem

        bh, model = handler_stub

        query = MMItem(text="q")
        documents = [MMItem(text=f"d{i}") for i in range(5)]

        results, usage = await bh.mm_rerank(
            query=query, documents=documents, top_n=2,
        )

        assert len(results) == 2

    @pytest.mark.anyio
    async def test_mm_rerank_empty_query_rejected(self, handler_stub):
        from boofinity.primitives import MMItem

        bh, model = handler_stub

        with pytest.raises(ValueError, match="query must have text or image"):
            await bh.mm_rerank(
                query=MMItem(),
                documents=[MMItem(text="d")],
            )

    @pytest.mark.anyio
    async def test_mm_rerank_uses_to_thread_not_event_loop(self, handler_stub):
        from boofinity.primitives import MMItem
        import threading

        bh, model = handler_stub

        main_thread = threading.current_thread()
        blocking_thread = None

        def _record_thread():
            nonlocal blocking_thread
            blocking_thread = threading.current_thread()
            model.encode_post.return_value = [0.5, 0.5]

        bh._sync_rerank_impl = _record_thread

        class TracingModel:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def encode_pre(self, pairs):
                blocking_thread_called = threading.current_thread()
                if blocking_thread_called is main_thread:
                    raise RuntimeError("encode_pre ran on event loop!")
                return self._inner.encode_pre(pairs)

            def encode_core(self, features_list):
                if threading.current_thread() is main_thread:
                    raise RuntimeError("encode_core ran on event loop!")
                return self._inner.encode_core(features_list)

            def encode_post(self, out):
                if threading.current_thread() is main_thread:
                    raise RuntimeError("encode_post ran on event loop!")
                return self._inner.encode_post(out)

            def encode_post_raw(self, out):
                if threading.current_thread() is main_thread:
                    raise RuntimeError("encode_post_raw ran on event loop!")
                return self._inner.encode_post_raw(out)

        original_model = bh.model_worker[0]._model
        bh.model_worker[0]._model = TracingModel(original_model)

        query = MMItem(text="q")
        documents = [MMItem(text="d")]

        results, usage = await bh.mm_rerank(
            query=query, documents=documents,
        )

        assert results is not None
        bh.model_worker[0]._model = original_model

    @pytest.mark.anyio
    async def test_mm_rerank_schedule_not_called(self, handler_stub):
        from boofinity.primitives import MMItem

        bh, model = handler_stub
        _schedule_called = False

        async def _fake_schedule(*args, **kwargs):
            nonlocal _schedule_called
            _schedule_called = True
            return [], 0

        bh._schedule = _fake_schedule

        await bh.mm_rerank(
            query=MMItem(text="q"),
            documents=[MMItem(text="d")],
        )

        assert not _schedule_called, "_schedule was called but should not have been"

    @pytest.mark.anyio
    async def test_mm_rerank_holds_lock_for_full_chain(self, handler_stub):
        import threading

        from boofinity.primitives import MMItem

        bh, model = handler_stub
        real_lock = threading.Lock()
        bh.model_worker[0]._model_lock = real_lock

        observed = []

        class TracingModel:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def encode_pre(self, pairs):
                observed.append(("pre", real_lock.locked(), threading.current_thread().name))
                return self._inner.encode_pre(pairs)

            def encode_core(self, features_list):
                observed.append(("core", real_lock.locked(), threading.current_thread().name))
                return self._inner.encode_core(features_list)

            def encode_post(self, out):
                observed.append(("post", real_lock.locked(), threading.current_thread().name))
                return self._inner.encode_post(out)

        bh.model_worker[0]._model = TracingModel(model)

        await bh.mm_rerank(query=MMItem(text="q"), documents=[MMItem(text="d")])

        # The lock was held for every encode call of the chain.
        assert [o[0] for o in observed] == ["pre", "core", "post"]
        assert all(held for _, held, _ in observed), observed
        # The whole chain ran on a single (worker) thread.
        assert len({o[2] for o in observed}) == 1
        # The lock is released after the chain completes.
        assert not real_lock.locked()

    @pytest.mark.anyio
    async def test_mm_rerank_serialises_concurrent_calls(self):
        import asyncio
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import MagicMock

        from boofinity.inference.batch_handler import BatchHandler
        from boofinity.primitives import MMItem

        lock = threading.Lock()
        intervals = []

        class SlowModel:
            def encode_pre(self, pairs):
                return [{"x": 1}]

            def encode_core(self, features_list):
                start = time.perf_counter()
                time.sleep(0.05)
                end = time.perf_counter()
                intervals.append((start, end))
                return MagicMock()

            def encode_post(self, out):
                return [0.5]

            def encode_post_raw(self, out):
                return [0.5]

        bh = BatchHandler.__new__(BatchHandler)
        bh.model_worker = [MagicMock()]
        bh.model_worker[0]._model = SlowModel()
        bh.model_worker[0]._model_lock = lock
        bh.model_worker[0].capabilities = {"rerank"}
        bh._threadpool = ThreadPoolExecutor(max_workers=4)
        bh._shutdown = threading.Event()

        async def _one():
            return await bh.mm_rerank(
                query=MMItem(text="q"), documents=[MMItem(text="d")]
            )

        await asyncio.gather(_one(), _one())

        assert len(intervals) == 2
        # The two encode_core windows must not overlap: the lock serialises them.
        first, second = sorted(intervals, key=lambda iv: iv[0])
        assert second[0] >= first[1] - 1e-3, f"encode chains overlapped: {intervals}"
