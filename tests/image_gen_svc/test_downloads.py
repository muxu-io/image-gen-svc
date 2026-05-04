from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from image_gen_svc.downloads import DownloadCoordinator, DownloadState


@pytest.mark.asyncio
async def test_coordinator_serializes_concurrent_requests(tmp_path: Path):
    calls = {"n": 0}

    async def fake_download(url: str, dest: Path) -> None:
        calls["n"] += 1
        await asyncio.sleep(0.05)
        dest.write_bytes(b"\x00" * 16)

    coord = DownloadCoordinator(downloader=fake_download)
    target = tmp_path / "model.safetensors"

    started = [coord.ensure_present(model_id="m", url="http://x", dest=target) for _ in range(3)]
    states = await asyncio.gather(*started)

    assert calls["n"] == 1
    assert all(s == DownloadState.READY for s in states)
    assert target.exists()


@pytest.mark.asyncio
async def test_coordinator_already_present_is_noop(tmp_path: Path):
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"\x00")

    async def explode(url: str, dest: Path) -> None:
        raise AssertionError("should not be called")

    coord = DownloadCoordinator(downloader=explode)
    state = await coord.ensure_present(model_id="m", url="http://x", dest=target)
    assert state == DownloadState.READY


@pytest.mark.asyncio
async def test_coordinator_returns_failed_on_downloader_error(tmp_path: Path):
    async def boom(url: str, dest: Path) -> None:
        raise RuntimeError("network down")

    coord = DownloadCoordinator(downloader=boom)
    target = tmp_path / "model.safetensors"
    state = await coord.ensure_present(model_id="m", url="http://x", dest=target)
    assert state == DownloadState.FAILED
    assert not target.exists()


@pytest.mark.asyncio
async def test_coordinator_snapshot_serializes_concurrent_requests(tmp_path: Path):
    calls = {"n": 0}

    async def fake_snapshot(
        repo_id: str, dest: Path, *, allow_patterns: list[str] | None = None
    ) -> None:
        calls["n"] += 1
        await asyncio.sleep(0.05)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_bytes(b"{}")

    coord = DownloadCoordinator(snapshot_downloader=fake_snapshot)
    target = tmp_path / "snapshot-dir"

    started = [
        coord.ensure_snapshot_present(model_id="m", repo_id="org/repo", dest=target)
        for _ in range(3)
    ]
    states = await asyncio.gather(*started)

    assert calls["n"] == 1
    assert all(s == DownloadState.READY for s in states)
    assert (target / "config.json").exists()


@pytest.mark.asyncio
async def test_coordinator_snapshot_already_present_is_noop(tmp_path: Path):
    target = tmp_path / "snapshot-dir"
    target.mkdir()

    async def explode(repo_id: str, dest: Path, *, allow_patterns: list[str] | None = None) -> None:
        raise AssertionError("should not be called")

    coord = DownloadCoordinator(snapshot_downloader=explode)
    state = await coord.ensure_snapshot_present(model_id="m", repo_id="org/repo", dest=target)
    assert state == DownloadState.READY


@pytest.mark.asyncio
async def test_coordinator_snapshot_returns_failed_on_downloader_error(tmp_path: Path):
    async def boom(repo_id: str, dest: Path, *, allow_patterns: list[str] | None = None) -> None:
        raise RuntimeError("hub down")

    coord = DownloadCoordinator(snapshot_downloader=boom)
    target = tmp_path / "snapshot-dir"
    state = await coord.ensure_snapshot_present(model_id="m", repo_id="org/repo", dest=target)
    assert state == DownloadState.FAILED
    assert not target.exists()


@pytest.mark.asyncio
async def test_is_downloading_reflects_inflight_state(tmp_path: Path):
    started = asyncio.Event()
    finish = asyncio.Event()

    async def gated(url: str, dest: Path) -> None:
        started.set()
        await finish.wait()
        dest.write_bytes(b"\x00")

    coord = DownloadCoordinator(downloader=gated)
    target = tmp_path / "model.safetensors"

    task = asyncio.create_task(coord.ensure_present(model_id="m", url="http://x", dest=target))
    await started.wait()
    assert coord.is_downloading("m") is True

    finish.set()
    state = await task
    assert state == DownloadState.READY
    assert coord.is_downloading("m") is False
