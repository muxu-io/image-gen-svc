from __future__ import annotations

from pathlib import Path

import pytest

from image_gen_svc.pipelines.base import PipelineAdapter, PipelineRequest
from image_gen_svc.pipelines.mock import MockPipeline


def _req(**overrides) -> PipelineRequest:
    base = dict(
        prompt="a wise persona",
        negative_prompt=None,
        seed=42,
        width=512,
        height=512,
        steps=20,
        guidance=7.5,
        reference_image=None,
        model_path=Path("/models/mock.bin"),
        extra={},
    )
    base.update(overrides)
    return PipelineRequest(**base)


@pytest.mark.asyncio
async def test_satisfies_protocol():
    pipe = MockPipeline()

    assert isinstance(pipe, PipelineAdapter)
    assert pipe.architecture == "mock"


@pytest.mark.asyncio
async def test_returns_image_at_requested_dims():
    pipe = MockPipeline()
    progress: list[float] = []

    result = await pipe.generate(_req(width=64, height=32), progress.append)

    assert result.width == 64
    assert result.height == 32
    assert len(result.rgb_bytes) == 64 * 32 * 3
    assert result.generation_time_s >= 0.0


@pytest.mark.asyncio
async def test_progress_emitted_monotonically():
    pipe = MockPipeline()
    progress: list[float] = []

    await pipe.generate(_req(), progress.append)

    assert progress[0] >= 0.0
    assert progress[-1] == 100.0
    assert progress == sorted(progress)
    assert len(progress) >= 2


@pytest.mark.asyncio
async def test_deterministic_for_same_seed():
    pipe = MockPipeline()

    r1 = await pipe.generate(_req(seed=7), lambda _: None)
    r2 = await pipe.generate(_req(seed=7), lambda _: None)

    assert r1.rgb_bytes == r2.rgb_bytes


@pytest.mark.asyncio
async def test_different_seeds_diverge():
    pipe = MockPipeline()

    r1 = await pipe.generate(_req(seed=7), lambda _: None)
    r2 = await pipe.generate(_req(seed=8), lambda _: None)

    assert r1.rgb_bytes != r2.rgb_bytes


@pytest.mark.asyncio
async def test_aclose_is_noop():
    pipe = MockPipeline()
    await pipe.aclose()  # must not raise
