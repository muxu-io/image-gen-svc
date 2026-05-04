"""Pipeline adapter protocol. Each image-generation backend (SDXL, Chroma DiT,
AuraFlow, mock) implements this. The adapter is the seam at which pipelines
are substitutable; the orchestrator never imports a concrete pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PipelineRequest:
    prompt: str
    negative_prompt: str | None
    seed: int
    width: int
    height: int
    steps: int
    guidance: float
    reference_image: bytes | None  # img2img conditioning bytes; None = txt2img
    model_path: Path  # path to checkpoint on disk
    extra: dict  # per-pipeline knobs (passed through)


@dataclass(frozen=True)
class PipelineResult:
    """RGB image as raw bytes (height × width × 3, uint8). The orchestrator
    encodes to webp before persisting. Generation_time_s is wall-clock from
    pipeline entry to image-ready, excluding upstream resolve / downstream
    encode."""

    rgb_bytes: bytes
    width: int
    height: int
    generation_time_s: float


ProgressCallback = Callable[[float], None]
"""percent in [0.0, 100.0]"""


@runtime_checkable
class PipelineAdapter(Protocol):
    name: str
    architecture: str  # matches ModelEntry.architecture

    async def generate(
        self,
        req: PipelineRequest,
        on_progress: ProgressCallback,
    ) -> PipelineResult: ...

    async def aclose(self) -> None:
        """Release VRAM and any held resources. Called by registry on
        eviction."""
        ...
