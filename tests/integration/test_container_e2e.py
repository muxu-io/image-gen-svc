"""End-to-end probes against the running container.

These exercise the same paths a real caller would: HTTP over loopback, real
uvicorn, real FastAPI app, real (mock) pipeline. The container is started in
`IMAGE_GEN_SVC_MOCK_ONLY=true` so no GPU or model checkpoint is required."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.integration


def test_health(http: httpx.Client) -> None:
    r = http.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["mock_only"] is True


def test_version(http: httpx.Client) -> None:
    r = http.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "image-gen-svc"
    assert isinstance(body["version"], str) and body["version"]


def test_models(http: httpx.Client) -> None:
    r = http.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert "default_alias" in body
    assert isinstance(body["models"], dict) and body["models"]


def test_render_returns_webp(http: httpx.Client) -> None:
    payload = {
        "prompt": "integration test",
        "seed": 1,
        "width": 64,
        "height": 64,
        "steps": 4,
        "guidance": 3.5,
        "job_id": "integration-test",
    }
    r = http.post("/render", json=payload)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/webp"
    assert r.headers["X-Job-Id"] == "integration-test"
    assert r.headers["X-Model-Used"]
    assert r.headers["X-Seed"] == "1"
    assert int(r.headers["X-Generation-Time-Ms"]) >= 0
    # webp container magic: "RIFF" .... "WEBP"
    assert r.content[:4] == b"RIFF"
    assert r.content[8:12] == b"WEBP"


def test_render_rejects_bad_payload(http: httpx.Client) -> None:
    r = http.post("/render", json={"prompt": "x"})  # missing required fields
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "invalid_request"
