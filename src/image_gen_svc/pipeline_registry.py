"""Warm pipeline cache with LRU=1 eviction. Only one pipeline is VRAM-resident
at a time; switching architectures evicts the previous via `aclose()` before
loading the new one.

The factories map is `architecture → callable returning a PipelineAdapter`.
Real factories load checkpoints from disk and move models to CUDA, which is
expensive — so they are called lazily, on first acquire of that architecture.

In `mock_only` mode, the registry ignores the requested architecture and
always returns a single `MockPipeline` instance."""

from __future__ import annotations

import asyncio
import gc
from collections.abc import Callable

from image_gen_svc.pipelines.base import PipelineAdapter
from image_gen_svc.pipelines.mock import MockPipeline

PipelineFactory = Callable[[], PipelineAdapter]


def _reclaim() -> None:
    """Force a GC cycle after evicting a pipeline.

    Adapters drop their pipeline reference in `aclose()`, but pipelines using
    accelerate's `enable_model_cpu_offload()` install hooks that form reference
    cycles, so their offloaded weights stay resident in host RAM until a cyclic
    GC pass runs. Without this, switching through several models accumulates
    host RAM and eventually OOM-kills the process when a large model (e.g.
    z-image) loads. `torch.cuda.empty_cache()` in the adapter only frees VRAM.
    """
    gc.collect()


class PipelineRegistry:
    def __init__(
        self,
        *,
        mock_only: bool,
        factories: dict[str, PipelineFactory],
    ):
        self._mock_only = mock_only
        self._factories = factories
        self._current: PipelineAdapter | None = None
        self._mock: MockPipeline | None = None
        self._lock = asyncio.Lock()

    async def acquire(self, architecture: str) -> PipelineAdapter:
        if self._mock_only:
            if self._mock is None:
                self._mock = MockPipeline()
            return self._mock

        async with self._lock:
            if self._current is not None and self._current.architecture == architecture:
                return self._current
            if self._current is not None:
                await self._current.aclose()
                self._current = None
                _reclaim()
            factory = self._factories.get(architecture)
            if factory is None:
                raise KeyError(f"no factory registered for architecture: {architecture!r}")
            self._current = factory()
            return self._current

    def loaded_names(self) -> list[str]:
        if self._mock_only:
            return ["mock"] if self._mock is not None else []
        return [self._current.name] if self._current is not None else []

    async def aclose(self) -> None:
        if self._current is not None:
            await self._current.aclose()
            self._current = None
            _reclaim()
        self._mock = None
