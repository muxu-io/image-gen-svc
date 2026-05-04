from __future__ import annotations

import json

import pytest


def _body(**overrides) -> dict:
    base = {
        "prompt": "a contemplative scholar",
        "seed": 42,
        "width": 64,
        "height": 64,
        "steps": 20,
        "guidance": 7.5,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_render_returns_webp_bytes_with_metadata_headers(client):
    """Default request omits both model_id and safe → safe=True (default) →
    photorealistic alias resolves to z-image-turbo (chroma is unsafe)."""
    r = await client.post("/render", json=_body())

    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/webp"
    assert r.headers["x-model-used"] == "z-image-turbo"
    assert r.headers["x-seed"] == "42"
    assert int(r.headers["x-generation-time-ms"]) >= 0
    assert "x-job-id" in {k.lower() for k in r.headers.keys()}
    assert r.content[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_render_uses_caller_supplied_job_id(client):
    r = await client.post("/render", json=_body(job_id="caller-supplied-123"))

    assert r.headers["x-job-id"] == "caller-supplied-123"


@pytest.mark.asyncio
async def test_render_accepts_request_without_steps_or_guidance(client):
    """Both fields are optional; the orchestrator falls back to the resolved
    model's defaults (and architecture-tier fallbacks beneath those)."""
    r = await client.post(
        "/render",
        json={
            "prompt": "x",
            "seed": 1,
            "width": 64,
            "height": 64,
        },
    )
    assert r.status_code == 200, r.text
    assert r.content[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_render_auto_generates_seed_when_omitted(client):
    """Seed is optional; the service generates a 32-bit value and returns it
    in X-Seed so the caller can reproduce the render by passing it back."""
    r = await client.post(
        "/render",
        json={
            "prompt": "x",
            "width": 64,
            "height": 64,
        },
    )
    assert r.status_code == 200, r.text
    assert "x-seed" in {k.lower() for k in r.headers.keys()}
    seed = int(r.headers["x-seed"])
    assert 0 <= seed <= 2**31 - 1


@pytest.mark.asyncio
async def test_render_auto_generated_seeds_vary_across_requests(client):
    """Successive requests without an explicit seed should not all collide
    on the same value (auto-gen is actually drawing fresh each call)."""
    seeds = set()
    for _ in range(8):
        r = await client.post("/render", json={"prompt": "x", "width": 64, "height": 64})
        assert r.status_code == 200
        seeds.add(int(r.headers["x-seed"]))
    # 8 draws from a 31-bit space colliding entirely is astronomically unlikely.
    assert len(seeds) > 1


@pytest.mark.asyncio
async def test_render_body_rejects_persona_id(client):
    """Cleaned API has no persona_id field. extra=forbid → 400 envelope."""
    r = await client.post("/render", json=_body(persona_id="ada"))
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "invalid_request"


@pytest.mark.asyncio
async def test_render_body_rejects_target(client):
    r = await client.post("/render", json=_body(target="portrait_canonical"))
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_request"


@pytest.mark.asyncio
async def test_render_rejects_unknown_model(client):
    r = await client.post("/render", json=_body(model_id="does-not-exist"))

    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "invalid_request"
    assert "does-not-exist" in body["message"]


@pytest.mark.asyncio
async def test_render_resolves_alias_in_model_id(client):
    """anime alias under default safe=True → animagine (pony filtered)."""
    r = await client.post("/render", json=_body(model_id="anime"))

    assert r.status_code == 200
    assert r.headers["x-model-used"] == "animagine-xl-4.0"


@pytest.mark.asyncio
async def test_render_safe_false_resolves_to_unsafe_default(client):
    """safe=False on the request returns the legacy default for the alias
    (chroma for photorealistic, pony for anime)."""
    r = await client.post("/render", json=_body(safe=False))
    assert r.status_code == 200
    assert r.headers["x-model-used"] == "chroma-1-hd"

    r = await client.post("/render", json=_body(model_id="anime", safe=False))
    assert r.status_code == 200
    assert r.headers["x-model-used"] == "pony-v7-base"


@pytest.mark.asyncio
async def test_render_explicit_unsafe_model_id_bypasses_filter(client):
    """Explicit model_id with safe=True still resolves to the named (unsafe)
    model — caller has taken responsibility."""
    r = await client.post("/render", json=_body(model_id="chroma-1-hd", safe=True))
    assert r.status_code == 200
    assert r.headers["x-model-used"] == "chroma-1-hd"


@pytest.mark.asyncio
async def test_render_no_safe_model_for_alias_returns_400(tmp_path):
    """An alias whose only matches are unsafe → 400 with envelope
    {"error": "no_safe_model_for_alias", "alias": <x>, ...}."""
    import httpx
    from asgi_lifespan import LifespanManager

    from image_gen_svc.app import build_app
    from image_gen_svc.config import ImageGenSvcConfig
    from image_gen_svc.job_registry import JobRegistry
    from image_gen_svc.model_registry import ModelRegistry
    from image_gen_svc.pipeline_registry import PipelineRegistry

    cfg = ImageGenSvcConfig(
        base_dir=tmp_path,
        port=7300,
        mock_only=True,
        models_dir=tmp_path / "models",
    )
    raw = {
        "models": {
            "only-unsafe": {
                "path": "/m.safetensors",
                "aliases": ["risky"],
                "architecture": "sdxl",
                "vram_gb": 1,
                "seed_stability": "high",
                "license": "x",
                "safe": False,
            }
        },
        "default_alias": "risky",
    }
    reg = ModelRegistry._from_raw(raw)
    pipe_reg = PipelineRegistry(mock_only=True, factories={})
    app = build_app(cfg, reg, pipe_reg, JobRegistry())

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/render", json=_body(model_id="risky", safe=True))

    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "no_safe_model_for_alias"
    assert body["alias"] == "risky"


@pytest.mark.asyncio
async def test_render_pipeline_crash_returns_500_envelope(monkeypatch, client):
    from image_gen_svc.pipelines.mock import MockPipeline

    async def crash(self, req, on_progress):
        raise RuntimeError("synthetic crash")

    monkeypatch.setattr(MockPipeline, "generate", crash)

    r = await client.post("/render", json=_body(job_id="j-crash"))
    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "generation_failed"
    assert body["job_id"] == "j-crash"


@pytest.mark.asyncio
async def test_render_multipart_with_reference_image(client):
    payload = _body(job_id="j-mp")
    fake_ref = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    files = {
        "payload": (None, json.dumps(payload), "application/json"),
        "reference_image": ("ref.png", fake_ref, "image/png"),
    }
    r = await client.post("/render", files=files)

    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/webp"
    assert r.headers["x-job-id"] == "j-mp"


@pytest.mark.asyncio
async def test_render_multipart_missing_payload(client):
    files = {"reference_image": ("r.png", b"x", "image/png")}
    r = await client.post("/render", files=files)
    assert r.status_code == 400
    assert "payload" in r.json()["message"]


@pytest.mark.asyncio
async def test_render_multipart_missing_reference_image(client):
    files = {"payload": (None, json.dumps(_body(job_id="j-noref")), "application/json")}
    r = await client.post("/render", files=files)
    assert r.status_code == 400
    body = r.json()
    assert "reference_image" in body["message"]
    assert body["job_id"] == "j-noref"


@pytest.mark.asyncio
async def test_render_returns_503_when_model_missing(tmp_path):
    """When cfg.mock_only is False and the model file isn't on disk, /render
    returns 503 model_loading + schedules a background download."""

    import httpx
    from asgi_lifespan import LifespanManager

    from image_gen_svc.app import build_app
    from image_gen_svc.config import ImageGenSvcConfig
    from image_gen_svc.downloads import DownloadCoordinator
    from image_gen_svc.job_registry import JobRegistry
    from image_gen_svc.model_registry import ModelRegistry
    from image_gen_svc.pipeline_registry import PipelineRegistry

    cfg = ImageGenSvcConfig(
        base_dir=tmp_path,
        port=7300,
        mock_only=False,  # important — exercises the 503 path
        models_dir=tmp_path / "models",
    )
    raw = {
        "models": {
            "missing-model": {
                "path": str(tmp_path / "missing.safetensors"),
                "url": "http://example.com/missing.safetensors",
                "architecture": "sdxl",
                "aliases": [],
                "vram_gb": 1,
                "seed_stability": "high",
                "license": "x",
            }
        },
        "default_alias": "missing-model",
    }
    reg = ModelRegistry._from_raw(raw)

    captured: dict = {}

    async def fake_downloader(url: str, dest) -> None:
        captured["url"] = url
        # Don't actually fetch; the test only verifies scheduling.

    coord = DownloadCoordinator(downloader=fake_downloader)
    pipe_reg = PipelineRegistry(mock_only=False, factories={})
    app = build_app(cfg, reg, pipe_reg, JobRegistry(), download_coordinator=coord)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/render",
                json={
                    "prompt": "x",
                    "model_id": "missing-model",
                    "seed": 1,
                    "width": 64,
                    "height": 64,
                    "steps": 1,
                    "guidance": 1.0,
                    "job_id": "j-503",
                },
            )

    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "model_loading"
    assert body["job_id"] == "j-503"


@pytest.mark.asyncio
async def test_render_returns_503_when_snapshot_missing(tmp_path):
    """Same 503 model_loading path applies to repo_id snapshots: missing
    snapshot dir → background snapshot pull scheduled, 503 returned."""

    import httpx
    from asgi_lifespan import LifespanManager

    from image_gen_svc.app import build_app
    from image_gen_svc.config import ImageGenSvcConfig
    from image_gen_svc.downloads import DownloadCoordinator
    from image_gen_svc.job_registry import JobRegistry
    from image_gen_svc.model_registry import ModelRegistry
    from image_gen_svc.pipeline_registry import PipelineRegistry

    cfg = ImageGenSvcConfig(
        base_dir=tmp_path,
        port=7300,
        mock_only=False,
        models_dir=tmp_path / "models",
    )
    raw = {
        "models": {
            "snap-model": {
                "path": str(tmp_path / "snapshot-dir"),
                "repo_id": "org/snap-model",
                "architecture": "z_image",
                "aliases": [],
                "vram_gb": 16,
                "seed_stability": "medium",
                "license": "x",
                "safe": True,
            }
        },
        "default_alias": "snap-model",
    }
    reg = ModelRegistry._from_raw(raw)

    captured: dict = {}

    async def fake_snapshot(repo_id: str, dest, *, allow_patterns: list[str] | None = None) -> None:
        captured["repo_id"] = repo_id
        captured["allow_patterns"] = allow_patterns

    coord = DownloadCoordinator(snapshot_downloader=fake_snapshot)
    pipe_reg = PipelineRegistry(mock_only=False, factories={})
    app = build_app(cfg, reg, pipe_reg, JobRegistry(), download_coordinator=coord)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/render",
                json={
                    "prompt": "x",
                    "model_id": "snap-model",
                    "seed": 1,
                    "width": 64,
                    "height": 64,
                    "steps": 4,
                    "guidance": 3.5,
                    "job_id": "j-snap-503",
                },
            )

    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "model_loading"
    assert body["job_id"] == "j-snap-503"
    assert "snapshot" in body["message"]
