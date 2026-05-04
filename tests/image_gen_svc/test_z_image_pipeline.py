"""Z-Image-Turbo adapter — mock-only registration check.

Real-engine validation is operator-driven (loading a 16 GB diffusers
ZImagePipeline on CUDA is out of scope for CI); same policy as the other
real-engine adapters under `pipelines/`. This file just verifies that the
`z_image` architecture flows through the pipeline registry's mock-only path
without surprises."""

from __future__ import annotations

import pytest

from image_gen_svc.pipeline_registry import PipelineRegistry
from image_gen_svc.pipelines.mock import MockPipeline


@pytest.mark.asyncio
async def test_z_image_registered_under_mock():
    reg = PipelineRegistry(mock_only=True, factories={})

    p = await reg.acquire("z_image")

    assert isinstance(p, MockPipeline)
    assert reg.loaded_names() == ["mock"]
