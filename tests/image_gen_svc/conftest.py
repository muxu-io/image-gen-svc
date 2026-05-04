from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager

from image_gen_svc.app import build_app
from image_gen_svc.config import ImageGenSvcConfig
from image_gen_svc.job_registry import JobRegistry
from image_gen_svc.model_registry import ModelRegistry
from image_gen_svc.pipeline_registry import PipelineRegistry


@pytest.fixture
def cfg(tmp_path: Path) -> ImageGenSvcConfig:
    return ImageGenSvcConfig(
        base_dir=tmp_path,
        port=7300,
        mock_only=True,
        models_dir=tmp_path / "models",
    )


@pytest.fixture
def model_reg() -> ModelRegistry:
    return ModelRegistry.load_default()


@pytest.fixture
def pipe_reg() -> PipelineRegistry:
    return PipelineRegistry(mock_only=True, factories={})


@pytest.fixture
def job_reg() -> JobRegistry:
    return JobRegistry()


@pytest_asyncio.fixture
async def client(cfg, model_reg, pipe_reg, job_reg) -> AsyncIterator[httpx.AsyncClient]:
    app = build_app(cfg, model_reg, pipe_reg, job_reg)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
