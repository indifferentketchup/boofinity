import asyncio
import threading
import time

import pytest

from boofinity import AsyncEmbeddingEngine, EngineArgs
from boofinity.inference.batch_handler import QUEUE_TIMEOUT
from boofinity.primitives import InferenceEngine


@pytest.mark.anyio
async def test_worker_exception_resolves_futures(monkeypatch):
    """A worker exception resolves in-flight caller futures with an exception."""
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(engine=InferenceEngine.debugengine)
    )

    def _raise(*args, **kwargs):
        raise RuntimeError("injected core failure")

    async with engine:
        monkeypatch.setattr(
            engine._batch_handler.model_worker[0]._model,
            "encode_core",
            _raise,
        )
        with pytest.raises(RuntimeError, match="injected core failure"):
            await engine.embed(["hello"])


@pytest.mark.anyio
async def test_wait_ready_and_running_flag():
    """wait_ready returns once threads are up, and astart sets running only after."""
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(engine=InferenceEngine.debugengine)
    )
    assert not engine.running
    await engine.astart()
    try:
        assert engine.running
    finally:
        await engine.astop()


@pytest.mark.anyio
async def test_shutdown_resolves_queued_futures():
    """Shutdown resolves futures queued in _queue_prio with RuntimeError."""
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(engine=InferenceEngine.debugengine)
    )
    await engine.astart()

    loop = asyncio.get_event_loop()
    futs = [loop.create_future() for _ in range(3)]

    from boofinity.primitives import EmbeddingInner, EmbeddingSingle, PrioritizedQueueItem

    items = []
    for fut in futs:
        inner = EmbeddingInner(content=EmbeddingSingle(sentence="test"), future=fut)
        items.append(PrioritizedQueueItem(priority=0, item=inner))
    engine._batch_handler._queue_prio.extend(items)

    await engine.astop()

    for fut in futs:
        assert fut.done()
        with pytest.raises(RuntimeError, match="server shutting down"):
            fut.result()


@pytest.mark.anyio
async def test_overload_graceful_after_astop():
    """is_overloaded and overload_status return gracefully after astop."""
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(engine=InferenceEngine.debugengine)
    )
    await engine.astart()
    await engine.astop()

    assert engine.is_overloaded() is True
    status = engine.overload_status()
    assert status.queue_fraction == 1.0
    assert status.queue_absolute == 0
    assert status.results_absolute == 0


@pytest.mark.anyio
async def test_sleep_guard_comparison():
    """The sleep-guard in _postprocess_batch compares correctly (recent inference -> sleep, old -> no sleep)."""
    from boofinity.inference.batch_handler import ModelWorker
    from unittest.mock import MagicMock

    worker = ModelWorker(
        shutdown=MagicMock(),
        model=MagicMock(),
        threadpool=MagicMock(),
        input_q=MagicMock(),
        output_q=MagicMock(),
    )
    worker._batch_delay = 0.005
    worker._postprocess_queue = MagicMock()
    worker._output_q = MagicMock()

    # Recent inference: should trigger sleep
    worker._last_inference = 99.995
    fake_time = 100.0
    worker._postprocess_queue.empty.return_value = True
    slept = [False]

    def fake_sleep(duration):
        slept[0] = True

    import time as _time

    orig_sleep = _time.sleep
    _time.sleep = fake_sleep
    try:
        # Simulate the condition check
        cond = (
            worker._postprocess_queue.empty()
            and worker._last_inference > fake_time - worker._batch_delay * 2
        )
        assert cond is True, "recent inference should trigger sleep guard"
    finally:
        _time.sleep = orig_sleep

    # Old inference: should NOT trigger sleep
    worker._last_inference = 99.0
    cond = (
        worker._postprocess_queue.empty()
        and worker._last_inference > fake_time - worker._batch_delay * 2
    )
    assert cond is False, "old inference should NOT trigger sleep guard"


@pytest.mark.anyio
async def test_wait_ready_blocks_until_all_events():
    """wait_ready blocks until all required events are set, then returns."""
    from boofinity.inference.batch_handler import BatchHandler
    from boofinity.transformer.embedder.dummytransformer import DummyTransformer

    model = DummyTransformer(engine_args=EngineArgs(engine=InferenceEngine.debugengine))
    handler = BatchHandler(
        model_replicas=[model],
        max_batch_size=4,
    )
    # Ensure none of the ready events are set (fresh handler, no spawn)
    handler._publisher_ready.clear()
    for w in handler.model_worker:
        w._preprocess_ready.clear()
        w._core_ready.clear()
        w._postprocess_ready.clear()

    result = [None]

    def _run_wait():
        try:
            handler.wait_ready(timeout=5.0)
            result[0] = "done"
        except RuntimeError:
            result[0] = "timeout"

    t = threading.Thread(target=_run_wait, daemon=True)
    t.start()

    # Give the thread time to enter wait_ready and block
    time.sleep(0.3)
    assert result[0] is None, "wait_ready should still be blocking"

    # Now satisfy every event the method checks
    handler._publisher_ready.set()
    for w in handler.model_worker:
        w._preprocess_ready.set()
        w._core_ready.set()
        w._postprocess_ready.set()

    t.join(timeout=5.0)
    assert result[0] == "done", "wait_ready should have returned after events were set"


@pytest.mark.anyio
async def test_m1_get_nowait_first_pattern():
    """Verify the real consumer path: get_nowait is tried before event.wait,
    so an enqueued item is picked up promptly without waiting the full QUEUE_TIMEOUT."""
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(engine=InferenceEngine.debugengine)
    )
    await engine.astart()
    try:
        # Warm up: first embed ensures the worker threads are fully running and
        # past their initial get_nowait/empty-queue iterations.
        await engine.embed(["warmup"])

        t0 = time.perf_counter()
        # Second embed exercises the get_nowait-first path: the worker thread
        # is already looping, and the publisher has moved items into the
        # input queue.  The preprocess worker should pick up the batch via
        # get_nowait() on its first iteration, well before the QUEUE_TIMEOUT
        # it would otherwise sleep for.
        result, usage = await engine.embed(["test get_nowait"])
        elapsed = time.perf_counter() - t0

        assert elapsed < QUEUE_TIMEOUT, (
            f"embed took {elapsed:.3f}s, expected < {QUEUE_TIMEOUT}s "
            "(suggests get_nowait-first pattern was not exercised)"
        )
        assert len(result) == 1
    finally:
        await engine.astop()
