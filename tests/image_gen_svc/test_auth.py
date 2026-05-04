"""Optional bearer-token auth on /render. /health, /models, /version, /docs
remain public regardless. Auth is off when IMAGE_GEN_API_KEY is unset."""

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


@pytest_asyncio.fixture
async def authed_client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    cfg = ImageGenSvcConfig(
        base_dir=tmp_path,
        port=7300,
        mock_only=True,
        models_dir=tmp_path / "models",
        api_key="secret",
    )
    model_reg = ModelRegistry.load_default()
    pipe_reg = PipelineRegistry(mock_only=True, factories={})
    job_reg = JobRegistry()
    app = build_app(cfg, model_reg, pipe_reg, job_reg)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


def _body() -> dict:
    return {
        "prompt": "x",
        "seed": 1,
        "width": 64,
        "height": 64,
        "steps": 1,
        "guidance": 1.0,
    }


@pytest.mark.asyncio
async def test_render_unauthorized_when_api_key_set_and_missing(authed_client):
    r = await authed_client.post("/render", json=_body())
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized"


@pytest.mark.asyncio
async def test_render_unauthorized_with_wrong_token(authed_client):
    r = await authed_client.post("/render", json=_body(), headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_render_authorized_with_correct_bearer(authed_client):
    r = await authed_client.post(
        "/render", json=_body(), headers={"Authorization": "Bearer secret"}
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/webp"


@pytest.mark.asyncio
async def test_health_models_version_public_without_auth(authed_client):
    assert (await authed_client.get("/health")).status_code == 200
    assert (await authed_client.get("/models")).status_code == 200
    assert (await authed_client.get("/version")).status_code == 200


@pytest.mark.asyncio
async def test_no_auth_required_when_api_key_unset(client):
    """Default fixture has api_key=None; /render works without auth."""
    r = await client.post("/render", json=_body())
    assert r.status_code == 200
