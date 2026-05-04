from __future__ import annotations

import pytest

from image_gen_svc.pipeline_registry import PipelineRegistry
from image_gen_svc.pipelines.mock import MockPipeline


class FakePipeline:
    def __init__(self, arch: str):
        self.architecture = arch
        self.name = arch
        self.closed = False

    async def generate(self, req, on_progress):
        raise NotImplementedError

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_mock_only_always_returns_mock():
    reg = PipelineRegistry(mock_only=True, factories={})

    p1 = await reg.acquire("sdxl")
    p2 = await reg.acquire("auraflow")

    assert isinstance(p1, MockPipeline)
    assert p1 is p2  # singleton
    assert reg.loaded_names() == ["mock"]


@pytest.mark.asyncio
async def test_lazy_load_on_first_acquire():
    created: list[str] = []

    def make(arch):
        def factory():
            created.append(arch)
            return FakePipeline(arch)

        return factory

    reg = PipelineRegistry(
        mock_only=False, factories={"sdxl": make("sdxl"), "chroma": make("chroma")}
    )
    assert reg.loaded_names() == []

    p = await reg.acquire("sdxl")

    assert p.architecture == "sdxl"
    assert created == ["sdxl"]
    assert reg.loaded_names() == ["sdxl"]


@pytest.mark.asyncio
async def test_acquire_same_architecture_returns_cached():
    reg = PipelineRegistry(mock_only=False, factories={"sdxl": lambda: FakePipeline("sdxl")})

    p1 = await reg.acquire("sdxl")
    p2 = await reg.acquire("sdxl")

    assert p1 is p2
    assert reg.loaded_names() == ["sdxl"]


@pytest.mark.asyncio
async def test_acquire_different_architecture_evicts_previous():
    reg = PipelineRegistry(
        mock_only=False,
        factories={
            "sdxl": lambda: FakePipeline("sdxl"),
            "chroma": lambda: FakePipeline("chroma"),
        },
    )

    sdxl = await reg.acquire("sdxl")
    chroma = await reg.acquire("chroma")

    assert sdxl.closed is True
    assert reg.loaded_names() == ["chroma"]
    assert chroma.closed is False


@pytest.mark.asyncio
async def test_acquire_unknown_architecture_raises():
    reg = PipelineRegistry(mock_only=False, factories={})

    with pytest.raises(KeyError, match="no factory registered for architecture"):
        await reg.acquire("unknown_arch")


@pytest.mark.asyncio
async def test_aclose_all_evicts():
    pipe = FakePipeline("sdxl")
    reg = PipelineRegistry(mock_only=False, factories={"sdxl": lambda: pipe})

    await reg.acquire("sdxl")
    await reg.aclose()

    assert pipe.closed is True
    assert reg.loaded_names() == []
