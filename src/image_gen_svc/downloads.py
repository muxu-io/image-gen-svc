"""Single-flight download coordinator for lazy model fetches.

When the render route encounters a registry entry whose checkpoint isn't on
disk, it asks the coordinator to fetch it. Concurrent requests for the same
model join the inflight task instead of triggering parallel downloads.

Two fetch shapes are supported:
  - `ensure_present(url, dest)` — single-file URL → streamed to `<dest>.part`,
    atomically renamed on success.
  - `ensure_snapshot_present(repo_id, dest)` — Hugging Face snapshot →
    `huggingface_hub.snapshot_download` into `<dest>.part/`, directory rename
    on success. Resumable on retry via HF's content cache.

The default downloaders run in a thread; tests inject fakes to exercise the
coordinator without hitting the network."""

from __future__ import annotations

import asyncio
import enum
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path


class DownloadState(enum.StrEnum):
    READY = "ready"
    DOWNLOADING = "downloading"
    FAILED = "failed"


Downloader = Callable[[str, Path], Awaitable[None]]
SnapshotDownloader = Callable[..., Awaitable[None]]
"""Signature: (repo_id, dest, *, allow_patterns: list[str] | None = None) -> None.
The protocol is open over the kw-only allow_patterns so test fakes can declare
just (repo_id, dest) and still satisfy the type."""


async def _default_downloader(url: str, dest: Path) -> None:
    """Stream-download `url` to `dest` atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _do() -> None:
        with urllib.request.urlopen(url) as r, tmp.open("wb") as f:  # nosec B310
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
        tmp.replace(dest)

    await asyncio.to_thread(_do)


async def _default_snapshot_downloader(
    repo_id: str, dest: Path, *, allow_patterns: list[str] | None = None
) -> None:
    """Pull a Hugging Face snapshot into `dest` atomically."""
    from huggingface_hub import snapshot_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.mkdir(parents=True, exist_ok=True)

    def _do() -> None:
        snapshot_download(  # nosec B615
            repo_id=repo_id, local_dir=str(tmp), allow_patterns=allow_patterns
        )
        tmp.replace(dest)

    await asyncio.to_thread(_do)


class DownloadCoordinator:
    def __init__(
        self,
        downloader: Downloader = _default_downloader,
        snapshot_downloader: SnapshotDownloader = _default_snapshot_downloader,
    ):
        self._downloader = downloader
        self._snapshot_downloader = snapshot_downloader
        self._inflight: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def ensure_present(self, *, model_id: str, url: str, dest: Path) -> DownloadState:
        return await self._ensure(model_id, dest, lambda: self._downloader(url, dest))

    async def ensure_snapshot_present(
        self,
        *,
        model_id: str,
        repo_id: str,
        dest: Path,
        allow_patterns: list[str] | None = None,
    ) -> DownloadState:
        return await self._ensure(
            model_id,
            dest,
            lambda: self._snapshot_downloader(repo_id, dest, allow_patterns=allow_patterns),
        )

    async def _ensure(
        self,
        model_id: str,
        dest: Path,
        make_coro: Callable[[], Awaitable[None]],
    ) -> DownloadState:
        if dest.exists():
            return DownloadState.READY

        async with self._lock:
            task = self._inflight.get(model_id)
            if task is None:
                task = asyncio.create_task(make_coro())
                self._inflight[model_id] = task

        try:
            await task
        except Exception:
            async with self._lock:
                self._inflight.pop(model_id, None)
            return DownloadState.FAILED
        async with self._lock:
            self._inflight.pop(model_id, None)
        return DownloadState.READY

    def is_downloading(self, model_id: str) -> bool:
        task = self._inflight.get(model_id)
        return task is not None and not task.done()
