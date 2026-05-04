"""Container-backed integration fixtures.

Two modes:

  * `container_base_url` / `http` — mock-mode container (`IMAGE_GEN_SVC_MOCK_ONLY=true`).
    No GPU, no model download. Selected by `pytest -m integration`.
  * `gpu_container_base_url` / `gpu_http` — real GPU container with `--gpus all`
    and a persistent models volume. Selected by `pytest -m gpu`. Requires an
    NVIDIA host with the container toolkit.

Both modes need the image built locally. Default tag is `image-gen-svc:integration`;
override with IMAGE_GEN_SVC_IMAGE.

    docker build -t image-gen-svc:integration .
    poetry run pytest -m integration       # mock mode
    poetry run pytest -m gpu               # real GPU

GPU-mode environment overrides:

  * IMAGE_GEN_SVC_INTEGRATION_MODELS_VOLUME — docker volume name or host path
    mounted at /models in the container. Default: `image-gen-svc-integration-models`
    (named volume, persists between runs — pre-populate with scripts/pull_models.py
    to avoid first-render downloads).
  * IMAGE_GEN_SVC_INTEGRATION_RENDER_TIMEOUT_S — per-model budget for the
    render test, including model download retries. Default 1800 (30 minutes).
  * IMAGE_GEN_SVC_INTEGRATION_RENDER_MODEL_IDS — comma-separated subset of
    registry ids the render test exercises. Unset = all entries. The render
    test parametrizes over this list, so each model is its own test case.
  * HF_TOKEN — forwarded into the container when set, so adapters that pull
    from gated HF repos (e.g. chroma → FLUX.1-schnell) can authenticate.
    Without it, tests for gated models skip with a clear reason.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator

import httpx
import pytest

DEFAULT_IMAGE = "image-gen-svc:integration"
DEFAULT_GPU_MODELS_VOLUME = "image-gen-svc-integration-models"
DEFAULT_GPU_RENDER_TIMEOUT_S = 1800.0
HEALTH_TIMEOUT_S = 30.0
HEALTH_POLL_INTERVAL_S = 0.5


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return True


def _image_present(image: str) -> bool:
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except subprocess.SubprocessError:
        return False
    return True


def _nvidia_host_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return True


def _wait_for_health(base_url: str, container_id: str) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200 and r.json().get("ok") is True:
                return
        except httpx.HTTPError as exc:
            last_err = exc
        time.sleep(HEALTH_POLL_INTERVAL_S)

    logs = subprocess.run(
        ["docker", "logs", container_id],
        capture_output=True,
        text=True,
        timeout=10,
    )
    raise RuntimeError(
        f"container did not become healthy in {HEALTH_TIMEOUT_S}s "
        f"(last error: {last_err!r})\n--- container logs ---\n"
        f"{logs.stdout}\n{logs.stderr}"
    )


@pytest.fixture(scope="session")
def container_base_url() -> Iterator[str]:
    if not _docker_available():
        pytest.skip("docker not available on PATH")

    image = os.environ.get("IMAGE_GEN_SVC_IMAGE", DEFAULT_IMAGE)
    if not _image_present(image):
        pytest.skip(
            f"image {image!r} not present; build with "
            f"`docker build -t {image} .` or set IMAGE_GEN_SVC_IMAGE"
        )

    port = _free_port()
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "-e",
            "IMAGE_GEN_SVC_MOCK_ONLY=true",
            "-p",
            f"127.0.0.1:{port}:7300",
            image,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    container_id = run.stdout.strip()
    base_url = f"http://127.0.0.1:{port}"

    try:
        _wait_for_health(base_url, container_id)
        yield base_url
    finally:
        subprocess.run(
            ["docker", "stop", container_id],
            capture_output=True,
            timeout=15,
        )


@pytest.fixture(scope="session")
def http(container_base_url: str) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=container_base_url, timeout=30.0) as client:
        yield client


@pytest.fixture(scope="session")
def gpu_container_base_url() -> Iterator[str]:
    if not _docker_available():
        pytest.skip("docker not available on PATH")
    if not _nvidia_host_available():
        pytest.skip("nvidia-smi not available; GPU integration requires an NVIDIA host")

    image = os.environ.get("IMAGE_GEN_SVC_IMAGE", DEFAULT_IMAGE)
    if not _image_present(image):
        pytest.skip(
            f"image {image!r} not present; build with "
            f"`docker build -t {image} .` or set IMAGE_GEN_SVC_IMAGE"
        )

    models_volume = os.environ.get(
        "IMAGE_GEN_SVC_INTEGRATION_MODELS_VOLUME", DEFAULT_GPU_MODELS_VOLUME
    )
    port = _free_port()
    docker_args = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{models_volume}:/models",
        "-p",
        f"127.0.0.1:{port}:7300",
    ]
    # Forward HF_TOKEN when set on the host so adapters that fetch from gated
    # HF repos (e.g. chroma → FLUX.1-schnell) can authenticate inside the
    # container. Unset → tests for gated models skip cleanly.
    if hf_token := os.environ.get("HF_TOKEN"):
        docker_args.extend(["-e", f"HF_TOKEN={hf_token}"])
    docker_args.append(image)
    run = subprocess.run(
        docker_args,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    container_id = run.stdout.strip()
    base_url = f"http://127.0.0.1:{port}"

    try:
        _wait_for_health(base_url, container_id)
        yield base_url
    finally:
        subprocess.run(
            ["docker", "stop", container_id],
            capture_output=True,
            timeout=15,
        )


@pytest.fixture(scope="session")
def gpu_http(gpu_container_base_url: str) -> Iterator[httpx.Client]:
    # Generous client timeout — generation latency varies by model and hardware.
    with httpx.Client(base_url=gpu_container_base_url, timeout=300.0) as client:
        yield client


@pytest.fixture(scope="session")
def gpu_render_timeout_s() -> float:
    raw = os.environ.get("IMAGE_GEN_SVC_INTEGRATION_RENDER_TIMEOUT_S")
    return float(raw) if raw else DEFAULT_GPU_RENDER_TIMEOUT_S
