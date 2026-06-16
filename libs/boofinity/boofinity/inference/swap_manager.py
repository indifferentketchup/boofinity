# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Optional

from boofinity.engine import AsyncEmbeddingEngine

try:
    import torch
except ImportError:  # CPU-only boxes never require CUDA / torch
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from logging import Logger

    from boofinity.args import EngineArgs

SWAP_SLOT_WAIT_S_DEFAULT = 30


class SwapManager:
    """Demand-driven, memory-bounded in-process model residency.

    Holds the raw ``EngineArgs`` list (not pre-constructed engines), builds an
    engine lazily on first request, caps resident engines via LRU eviction, and
    optionally evicts idle engines after a TTL. Eviction is gated on an explicit
    per-model in-flight ref-count so a serving engine is never stopped. It only
    constructs, ``astart()``s, and ``astop()``s engines inside this process: no
    process spawning, no request forwarding, no GPU-memory accounting.
    """

    def __init__(
        self,
        engine_args_list: list["EngineArgs"],
        max_resident: int,
        ttl_s: int,
        slot_wait_s: int,
        logger: "Logger",
    ) -> None:
        self._args = {ea.served_model_name: ea for ea in engine_args_list}
        self._max_resident = max_resident  # 0 = unlimited
        self._ttl_s = ttl_s  # 0 = no idle reaper
        self._slot_wait_s = slot_wait_s
        self._logger = logger
        self._engines: dict[str, AsyncEmbeddingEngine] = {}  # resident only
        self._inflight: dict[str, int] = {}  # per-model in-flight request count
        self._last_used: dict[str, float] = {}  # monotonic timestamp
        self._loads: dict[str, asyncio.Task] = {}  # single-flight builds
        self._cond = asyncio.Condition()  # guards all maps; notified on release
        self._reaper: Optional[asyncio.Task] = None
        self.ready = False

    @property
    def engine_args(self) -> dict[str, "EngineArgs"]:
        """Public read view of configured args, keyed by served_model_name.

        Used by /v1/models to list all configured models without building them.
        """
        return dict(self._args)

    def is_resident(self, name: str) -> bool:
        engine = self._engines.get(name)
        return engine is not None and engine.is_running

    def resident_engine(self, name: str) -> Optional[AsyncEmbeddingEngine]:
        """Return the resident engine for live stats, or None if not loaded.

        Lets /v1/models read live data without touching internal maps."""
        engine = self._engines.get(name)
        return engine if engine is not None and engine.is_running else None

    def _resolve_name(self, model: str) -> str:
        """Resolve a request model to a configured served_model_name.

        Mirrors AsyncEngineArray.__getitem__: with exactly one model configured,
        any name resolves to it; otherwise an unknown name raises IndexError.
        """
        if model in self._args:
            return model
        if len(self._args) == 1:
            return next(iter(self._args))
        raise IndexError(
            f"Engine for model name `{model}` not found. "
            f"Available model names are {list(self._args)}"
        )

    @asynccontextmanager
    async def using(self, model: str):
        engine = await self._acquire(model)
        try:
            yield engine
        finally:
            await self._release(model)

    async def _acquire(self, model: str) -> AsyncEmbeddingEngine:
        name = self._resolve_name(model)
        async with self._cond:
            engine = self._engines.get(name)
            if engine is not None and engine.is_running:
                self._inflight[name] = self._inflight.get(name, 0) + 1
                self._last_used[name] = time.monotonic()
                return engine
        await self._ensure_built(name)
        async with self._cond:
            self._inflight[name] = self._inflight.get(name, 0) + 1
            self._last_used[name] = time.monotonic()
            return self._engines[name]

    async def _release(self, model: str) -> None:
        name = self._resolve_name(model)
        async with self._cond:
            self._inflight[name] = max(0, self._inflight.get(name, 1) - 1)
            self._last_used[name] = time.monotonic()
            self._cond.notify_all()

    async def _ensure_built(self, name: str) -> None:
        async with self._cond:
            engine = self._engines.get(name)
            if engine is not None and engine.is_running:
                return
            task = self._loads.get(name)
            if task is None:
                await self._make_room(name)
                task = asyncio.ensure_future(self._build_and_start(name))
                self._loads[name] = task
        try:
            await task  # raises to every waiter if the build failed
        finally:
            async with self._cond:
                if self._loads.get(name) is task:
                    del self._loads[name]  # cleared on success AND failure
                self._cond.notify_all()

    async def _build_and_start(self, name: str) -> None:
        # from_args -> select_model loads weights synchronously; offload it so
        # the event loop is not blocked for the full weight load + warmup.
        engine = await asyncio.to_thread(AsyncEmbeddingEngine.from_args, self._args[name])
        await engine.astart()
        async with self._cond:
            self._engines[name] = engine

    def _resident_count(self, target: str) -> int:
        return len(
            [n for n, e in self._engines.items() if e.is_running and n != target]
        )

    def _lru_idle_victim(self, target: str) -> Optional[str]:
        candidates = [
            n
            for n, e in self._engines.items()
            if e.is_running and n != target and self._inflight.get(n, 0) == 0
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda n: self._last_used.get(n, 0.0))

    async def _make_room(self, target: str) -> None:
        """Entered holding self._cond. Evicts LRU idle engines until under cap.

        On timeout (all resident engines busy past slot_wait_s) it logs a
        warning naming the target and proceeds (soft over-cap).
        """
        if self._max_resident == 0:
            return
        try:
            async with asyncio.timeout(self._slot_wait_s):
                while self._resident_count(target) >= self._max_resident:
                    victim = self._lru_idle_victim(target)
                    if victim is not None:
                        engine = self._engines.pop(victim)
                        self._loads.pop(victim, None)
                        self._cond.release()  # astop() must run OUTSIDE the lock
                        try:
                            await self._stop_and_free(engine)
                        finally:
                            await self._cond.acquire()
                        continue
                    await self._cond.wait()  # every resident engine is busy
        except asyncio.TimeoutError:
            self._logger.warning(
                "swap: loading `%s` over cap (max_resident=%s); all resident "
                "engines busy past swap_slot_wait_s=%ss",
                target,
                self._max_resident,
                self._slot_wait_s,
            )

    async def _stop_and_free(self, engine: AsyncEmbeddingEngine) -> None:
        await engine.astop()  # idempotent; joins worker threads
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()  # return freed blocks from the caching allocator
        del engine

    async def _reaper_tick(self, now: float) -> None:
        async with self._cond:
            victims = [
                n
                for n in list(self._engines)
                if self._inflight.get(n, 0) == 0
                and now - self._last_used.get(n, now) > self._ttl_s
            ]
            engines = [self._engines.pop(n) for n in victims]
            for n in victims:
                self._loads.pop(n, None)
        for engine in engines:  # astop OUTSIDE the lock
            await self._stop_and_free(engine)

    async def _reaper_loop(self) -> None:
        while True:
            await asyncio.sleep(min(self._ttl_s, 30))
            await self._reaper_tick(time.monotonic())

    def start_reaper(self) -> None:
        if self._ttl_s > 0 and self._reaper is None:
            self._reaper = asyncio.ensure_future(self._reaper_loop())

    async def stop_reaper(self) -> None:
        if self._reaper is None:
            return
        self._reaper.cancel()
        try:
            await self._reaper
        except asyncio.CancelledError:
            pass
        self._reaper = None

    async def shutdown(self) -> None:
        """Stop every resident engine. astop() is idempotent and serialized per
        engine, so a gather is safe even if a reaper tick raced a stop."""
        async with self._cond:
            engines = list(self._engines.values())
            self._engines.clear()
            self._loads.clear()
        await asyncio.gather(*(self._stop_and_free(e) for e in engines))
