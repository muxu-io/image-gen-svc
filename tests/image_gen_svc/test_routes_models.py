from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_models_endpoint_lists_bundled_default(client):
    r = await client.get("/models")

    assert r.status_code == 200
    body = r.json()
    assert body["default_alias"] == "photorealistic"
    assert "z-image-turbo" in body["models"]
    assert "animagine-xl-4.0" in body["models"]
    assert "chroma-1-hd" in body["models"]
    assert "realvis-xl-v5" in body["models"]
    assert "pony-v7-base" in body["models"]
    assert body["models"]["chroma-1-hd"]["architecture"] == "chroma"
    assert "photorealistic" in body["models"]["chroma-1-hd"]["aliases"]
    assert body["default_models"]["photorealistic"] == "chroma-1-hd"


@pytest.mark.asyncio
async def test_models_endpoint_includes_loaded_and_safe_flags(client, tmp_path):
    """loaded reflects on-disk presence; safe comes from the registry."""
    r = await client.get("/models")
    body = r.json()
    for entry in body["models"].values():
        assert "loaded" in entry
        assert "safe" in entry
        assert isinstance(entry["loaded"], bool)
        assert isinstance(entry["safe"], bool)
    assert body["models"]["chroma-1-hd"]["safe"] is False
    assert body["models"]["z-image-turbo"]["safe"] is True
    # The bundled registry's checkpoint paths point at /models/* which doesn't
    # exist in the test environment — so loaded should be False for all.
    assert all(e["loaded"] is False for e in body["models"].values())


@pytest.mark.asyncio
async def test_models_endpoint_exposes_default_steps_and_guidance(client):
    """Each entry surfaces the effective defaults /render uses when callers
    omit steps/guidance, so callers can preview them without a probe render."""
    r = await client.get("/models")
    body = r.json()
    for entry in body["models"].values():
        assert isinstance(entry["default_steps"], int)
        assert isinstance(entry["default_guidance"], (int, float))
    assert body["models"]["z-image-turbo"]["default_steps"] == 4
    assert body["models"]["z-image-turbo"]["default_guidance"] == 3.5
    assert body["models"]["realvis-xl-v5"]["default_steps"] == 25
