"""End-to-end probes against the running container with a real GPU.

Same shape as the mock-mode tests, but the container is started with
`--gpus all` and no MOCK_ONLY override, so renders go through the real
diffusers pipeline. First-run latency is dominated by checkpoint download
(cached in a docker volume); subsequent runs hit the cache.

The render test parametrizes over every model in the bundled registry so each
adapter, download path, and inference path is exercised. Models gated behind
HF auth (e.g. chroma → FLUX.1-schnell) skip cleanly when HF_TOKEN isn't set
on the container. To avoid hours of cold-cache downloads, pre-populate the
named volume with `scripts/pull_models.py` first.

Override the model set via IMAGE_GEN_SVC_INTEGRATION_RENDER_MODEL_IDS
(comma-separated). Unset = all registry entries.

Selected by `pytest -m gpu`. Skipped automatically when docker, the image,
or an NVIDIA host is unavailable.
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

from image_gen_svc.model_registry import ModelRegistry

pytestmark = pytest.mark.gpu

RETRY_INTERVAL_S = 5.0


def _model_ids_under_test() -> list[str]:
    all_ids = sorted(ModelRegistry.load_default().models.keys())
    raw = os.environ.get("IMAGE_GEN_SVC_INTEGRATION_RENDER_MODEL_IDS")
    if not raw:
        return all_ids
    requested = [m.strip() for m in raw.split(",") if m.strip()]
    unknown = set(requested) - set(all_ids)
    if unknown:
        raise ValueError(f"unknown model ids in env override: {sorted(unknown)}")
    return requested


MODEL_IDS = _model_ids_under_test()


def test_health(gpu_http: httpx.Client) -> None:
    r = gpu_http.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["mock_only"] is False


def test_version(gpu_http: httpx.Client) -> None:
    r = gpu_http.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "image-gen-svc"
    assert body["torch"] is not None
    assert body["diffusers"] is not None


def test_models(gpu_http: httpx.Client) -> None:
    r = gpu_http.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert body["models"]


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_render(
    gpu_http: httpx.Client,
    gpu_render_timeout_s: float,
    model_id: str,
) -> None:
    job_id = f"gpu-integration-{model_id}"
    payload = {
        "prompt": "a small red cube on a white background",
        "seed": 42,
        "model_id": model_id,
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance": 3.5,
        "job_id": job_id,
        # Explicit model_id bypasses the safe filter, but pass safe=False so
        # unsafe entries (chroma, pony) get through unambiguously.
        "safe": False,
    }
    deadline = time.monotonic() + gpu_render_timeout_s
    last_loading: dict | None = None
    while time.monotonic() < deadline:
        r = gpu_http.post("/render", json=payload)
        if r.status_code == 200:
            assert r.headers["content-type"] == "image/webp"
            assert r.headers["X-Job-Id"] == job_id
            assert r.headers["X-Model-Used"] == model_id
            assert r.headers["X-Seed"] == "42"
            assert r.content[:4] == b"RIFF"
            assert r.content[8:12] == b"WEBP"
            return
        if r.status_code == 503:
            body = r.json()
            if body.get("error") == "model_loading":
                last_loading = body
                time.sleep(RETRY_INTERVAL_S)
                continue
        if r.status_code == 500 and "GatedRepoError" in r.text:
            pytest.skip(f"{model_id}: gated HF repo; set HF_TOKEN on the container to run")
        pytest.fail(f"{model_id}: unexpected response {r.status_code}: {r.text}")

    pytest.fail(
        f"{model_id}: still loading after {gpu_render_timeout_s}s "
        f"(last status: {last_loading!r}); raise "
        "IMAGE_GEN_SVC_INTEGRATION_RENDER_TIMEOUT_S or pre-populate the volume"
    )
