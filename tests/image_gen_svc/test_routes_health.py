from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health_ok_in_mock_only(client):
    r = await client.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["mock_only"] is True
    assert body["gpu_vram_used_gb"] == 0.0
    assert isinstance(body["models_loaded"], list)
    # No pipelines acquired yet
    assert body["models_loaded"] == []
