"""Deterministic in-process pipeline for tests. No GPU, no torch.

Image content is a function of (seed, prompt-length, width, height) so tests
can assert determinism without comparing against a fixture binary."""

from __future__ import annotations

import asyncio
import hashlib
import time

from image_gen_svc.pipelines.base import (
    PipelineRequest,
    PipelineResult,
    ProgressCallback,
)


class MockPipeline:
    name = "mock"
    architecture = "mock"

    async def generate(
        self,
        req: PipelineRequest,
        on_progress: ProgressCallback,
    ) -> PipelineResult:
        start = time.monotonic()

        # Emit 5 progress beats; yield to event loop between to be a fair
        # async citizen.
        for pct in (20.0, 40.0, 60.0, 80.0, 100.0):
            on_progress(pct)
            await asyncio.sleep(0)

        # Deterministic content: hash(seed | prompt | dims) → repeating byte
        # pattern of the requested length.
        digest = hashlib.sha256(
            f"{req.seed}|{req.prompt}|{req.width}x{req.height}".encode()
        ).digest()
        size = req.width * req.height * 3
        # Cycle the 32-byte digest to fill `size` bytes.
        repeats = (size + len(digest) - 1) // len(digest)
        rgb = (digest * repeats)[:size]

        return PipelineResult(
            rgb_bytes=rgb,
            width=req.width,
            height=req.height,
            generation_time_s=time.monotonic() - start,
        )

    async def aclose(self) -> None:
        return None
