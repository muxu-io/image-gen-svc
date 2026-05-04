from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_version_returns_self_and_dependency_keys(client):
    r = await client.get("/version")

    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "image-gen-svc"
    assert body["version"]
    # diffusers/torch may be None in test environments without CUDA stack.
    assert "diffusers" in body
    assert "torch" in body
    assert "transformers" in body
