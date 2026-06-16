import asyncio
import logging

import pytest

import boofinity.inference.swap_manager as swap_module
from boofinity.inference.swap_manager import SwapManager


class FakeEngineArgs:
    def __init__(self, served_model_name: str):
        self.served_model_name = served_model_name


class FakeEngine:
    """Stand-in for AsyncEmbeddingEngine with counting astart/astop."""

    def __init__(self, name: str):
        self.name = name
        self.running = False
        self.start_calls = 0
        self.stop_calls = 0

    async def astart(self):
        self.start_calls += 1
        self.running = True

    async def astop(self):
        self.stop_calls += 1
        self.running = False

    @property
    def is_running(self) -> bool:
        return self.running


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _logger():
    return logging.getLogger("test_swap_manager")


def _make_manager(monkeypatch, names, *, max_resident=1, ttl_s=0, slot_wait_s=30,
                  build=None, builds=None):
    """Build a SwapManager over fake engines with a monkeypatched from_args.

    `build` overrides the per-name factory; `builds` is a dict name -> callable.
    Returns (manager, factory_record) where factory_record["count"][name] tracks
    how many times from_args ran for that name.
    """
    args_list = [FakeEngineArgs(n) for n in names]
    record = {"count": {}, "engines": {}}

    def default_factory(engine_args):
        n = engine_args.served_model_name
        record["count"][n] = record["count"].get(n, 0) + 1
        eng = FakeEngine(n)
        record["engines"][n] = eng
        return eng

    def from_args(engine_args):
        n = engine_args.served_model_name
        record["count"][n] = record["count"].get(n, 0) + 1
        if builds is not None and n in builds:
            return builds[n](engine_args)
        if build is not None:
            return build(engine_args)
        eng = FakeEngine(n)
        record["engines"][n] = eng
        return eng

    monkeypatch.setattr(
        swap_module.AsyncEmbeddingEngine, "from_args", staticmethod(from_args)
    )
    mgr = SwapManager(args_list, max_resident, ttl_s, slot_wait_s, _logger())
    return mgr, record


# --- task 4.1: builds once, reuses resident ---
@pytest.mark.anyio
async def test_build_then_reuse(monkeypatch):
    mgr, rec = _make_manager(monkeypatch, ["A"], max_resident=1)
    async with mgr.using("A") as e1:
        assert e1.is_running
        assert e1.start_calls == 1
    assert rec["count"]["A"] == 1
    # second use: resident, not rebuilt
    async with mgr.using("A") as e2:
        assert e2 is e1
    assert rec["count"]["A"] == 1
    assert e1.start_calls == 1


# --- task 4.2: in-flight balance on normal exit and on exception ---
@pytest.mark.anyio
async def test_inflight_balance(monkeypatch):
    mgr, _ = _make_manager(monkeypatch, ["A"], max_resident=1)
    async with mgr.using("A"):
        assert mgr._inflight["A"] == 1
    assert mgr._inflight["A"] == 0

    with pytest.raises(RuntimeError):
        async with mgr.using("A"):
            assert mgr._inflight["A"] == 1
            raise RuntimeError("boom")
    assert mgr._inflight["A"] == 0


# --- task 4.3: single-flight build under concurrency ---
@pytest.mark.anyio
async def test_single_flight(monkeypatch):
    gate = asyncio.Event()

    def slow_build(engine_args):
        eng = FakeEngine(engine_args.served_model_name)

        async def slow_astart():
            await gate.wait()
            eng.start_calls += 1
            eng.running = True

        eng.astart = slow_astart  # type: ignore[method-assign]
        return eng

    mgr, rec = _make_manager(monkeypatch, ["A"], max_resident=1, build=slow_build)

    async def acquire_once():
        async with mgr.using("A") as e:
            return e

    tasks = [asyncio.ensure_future(acquire_once()) for _ in range(5)]
    await asyncio.sleep(0.05)  # let all 5 reach the in-flight build
    gate.set()
    results = await asyncio.gather(*tasks)
    assert rec["count"]["A"] == 1  # from_args ran once
    assert all(r is results[0] for r in results)
    assert results[0].start_calls == 1


# --- task 4.4: build failure propagates and is retryable ---
@pytest.mark.anyio
async def test_build_failure_then_retry(monkeypatch):
    state = {"fail": True}

    def maybe_failing(engine_args):
        if state["fail"]:
            raise ValueError("download failed")
        return FakeEngine(engine_args.served_model_name)

    mgr, _ = _make_manager(monkeypatch, ["A"], max_resident=1, build=maybe_failing)

    async def acquire_once():
        async with mgr.using("A"):
            pass

    tasks = [asyncio.ensure_future(acquire_once()) for _ in range(4)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, ValueError) for r in results)
    assert "A" not in mgr._engines
    assert "A" not in mgr._loads

    # subsequent build now succeeds
    state["fail"] = False
    async with mgr.using("A") as e:
        assert e.is_running
    assert mgr.is_resident("A")


# --- task 4.5: LRU eviction frees the engine and empties CUDA cache ---
@pytest.mark.anyio
async def test_lru_eviction(monkeypatch):
    mgr, rec = _make_manager(monkeypatch, ["A", "B"], max_resident=1)

    class FakeCuda:
        def __init__(self):
            self.empty_calls = 0

        def is_available(self):
            return True

        def empty_cache(self):
            self.empty_calls += 1

    fake_torch = type("FakeTorch", (), {})()
    fake_torch.cuda = FakeCuda()
    monkeypatch.setattr(swap_module, "torch", fake_torch)

    async with mgr.using("A"):
        pass
    assert mgr.is_resident("A")
    engine_a = rec["engines"]["A"]

    async with mgr.using("B"):
        pass
    assert engine_a.stop_calls == 1
    assert "A" not in mgr._engines
    assert mgr.is_resident("B")
    assert len([e for e in mgr._engines.values() if e.is_running]) == 1
    assert fake_torch.cuda.empty_calls >= 1


# --- task 4.6: busy engine soft over-cap + non-blocking concurrent acquire ---
@pytest.mark.anyio
async def test_busy_soft_overcap(monkeypatch, caplog):
    mgr, rec = _make_manager(monkeypatch, ["A", "B"], max_resident=1, slot_wait_s=0)

    release = asyncio.Event()

    async def hold_a():
        async with mgr.using("A"):
            await release.wait()

    holder = asyncio.ensure_future(hold_a())
    await asyncio.sleep(0.02)  # A resident with in-flight 1
    assert mgr._inflight["A"] == 1

    with caplog.at_level(logging.WARNING):
        async with mgr.using("B"):
            assert mgr.is_resident("B")
            # A was NOT stopped while busy
            assert rec["engines"]["A"].stop_calls == 0
    assert any("over cap" in r.message for r in caplog.records)

    release.set()
    await holder


@pytest.mark.anyio
async def test_concurrent_acquire_not_blocked_during_eviction(monkeypatch):
    # Eviction astop runs outside _cond: a concurrent acquire of a resident model
    # is not blocked while a (slow) astop runs for the victim.
    mgr, rec = _make_manager(monkeypatch, ["A", "B", "C"], max_resident=2)

    async with mgr.using("A"):
        pass
    async with mgr.using("B"):
        pass
    # A and B resident, both idle. Make B's astop slow.
    slow_done = asyncio.Event()
    orig_b_astop = rec["engines"]["B"].astop

    async def slow_astop():
        await asyncio.sleep(0.05)
        await orig_b_astop()
        slow_done.set()

    rec["engines"]["B"].astop = slow_astop  # type: ignore[method-assign]

    # Touch A so B becomes the LRU victim.
    async with mgr.using("A"):
        pass

    async def acquire_c():
        async with mgr.using("C"):
            pass

    async def acquire_a_again():
        async with mgr.using("A") as e:
            return e

    c_task = asyncio.ensure_future(acquire_c())
    await asyncio.sleep(0.01)  # let C start evicting B (slow astop in progress)
    # A is resident; this must resolve without waiting for B's astop to finish.
    a_engine = await asyncio.wait_for(acquire_a_again(), timeout=0.03)
    assert a_engine.is_running
    await c_task
    assert slow_done.is_set()


# --- task 4.7: wait resolves when the busy engine frees ---
@pytest.mark.anyio
async def test_wait_resolves(monkeypatch, caplog):
    mgr, rec = _make_manager(monkeypatch, ["A", "B"], max_resident=1, slot_wait_s=5)
    release = asyncio.Event()

    async def hold_a():
        async with mgr.using("A"):
            await release.wait()

    holder = asyncio.ensure_future(hold_a())
    await asyncio.sleep(0.02)
    assert mgr._inflight["A"] == 1

    async def acquire_b():
        async with mgr.using("B"):
            return True

    b_task = asyncio.ensure_future(acquire_b())
    await asyncio.sleep(0.02)  # B is waiting for a slot, not over-cap
    assert not b_task.done()
    with caplog.at_level(logging.WARNING):
        release.set()
        await holder
        assert await asyncio.wait_for(b_task, timeout=1.0)
    assert rec["engines"]["A"].stop_calls == 1  # A evicted after freeing
    assert mgr.is_resident("B")
    assert not any("over cap" in r.message for r in caplog.records)


# --- task 4.8: _resolve_name ---
@pytest.mark.anyio
async def test_resolve_name(monkeypatch):
    mgr2, _ = _make_manager(monkeypatch, ["A", "B"], max_resident=2)
    with pytest.raises(IndexError):
        mgr2._resolve_name("unknown")
    assert mgr2._resolve_name("A") == "A"

    mgr1, _ = _make_manager(monkeypatch, ["solo"], max_resident=1)
    assert mgr1._resolve_name("anything") == "solo"
    assert mgr1._resolve_name("solo") == "solo"


# --- task 4.9: reaper tick ---
@pytest.mark.anyio
async def test_reaper_tick(monkeypatch):
    mgr, rec = _make_manager(monkeypatch, ["A"], max_resident=1, ttl_s=10)
    async with mgr.using("A"):
        pass
    # stale last_used: used long ago, idle
    mgr._last_used["A"] = 100.0
    await mgr._reaper_tick(now=200.0)  # 100s idle > ttl 10
    assert rec["engines"]["A"].stop_calls == 1
    assert "A" not in mgr._engines


@pytest.mark.anyio
async def test_reaper_skips_busy(monkeypatch):
    mgr, rec = _make_manager(monkeypatch, ["A"], max_resident=1, ttl_s=10)
    release = asyncio.Event()

    async def hold_a():
        async with mgr.using("A"):
            await release.wait()

    holder = asyncio.ensure_future(hold_a())
    await asyncio.sleep(0.02)
    mgr._last_used["A"] = 100.0
    await mgr._reaper_tick(now=200.0)
    assert rec["engines"]["A"].stop_calls == 0  # busy, not reaped
    assert mgr.is_resident("A")
    release.set()
    await holder
