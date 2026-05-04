"""Render orchestrator. Resolves a model, acquires a pipeline, runs generation,
encodes the output as webp, and emits SSE events along the way. The orchestrator
returns image bytes — output paths and disk persistence are the caller's
concern."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from image_gen_svc.config import ImageGenSvcConfig
from image_gen_svc.job_registry import JobEvent, JobRegistry
from image_gen_svc.model_registry import ModelRegistry
from image_gen_svc.pipeline_registry import PipelineRegistry
from image_gen_svc.pipelines.base import PipelineRequest


@dataclass(frozen=True)
class RenderRequest:
    job_id: str
    prompt: str
    negative_prompt: str | None
    seed: int
    model_id: str | None
    width: int
    height: int
    steps: int
    guidance: float
    reference_image: bytes | None
    safe: bool = True


@dataclass(frozen=True)
class RenderResult:
    webp_bytes: bytes
    model_used: str
    seed: int
    generation_time_s: float


def _encode_webp(rgb_bytes: bytes, width: int, height: int) -> bytes:
    img = Image.frombytes("RGB", (width, height), rgb_bytes)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=92, method=4)
    return buf.getvalue()


async def run_render(
    req: RenderRequest,
    cfg: ImageGenSvcConfig,
    model_registry: ModelRegistry,
    pipeline_registry: PipelineRegistry,
    job_registry: JobRegistry,
) -> RenderResult:
    job_id = req.job_id
    started = time.monotonic()
    try:
        entry = model_registry.resolve(req.model_id, safe=req.safe)

        job_registry.publish(
            job_id,
            JobEvent(type="job_started", data={"model": entry.id}),
        )

        pipeline = await pipeline_registry.acquire(entry.architecture)

        def on_progress(pct: float) -> None:
            step = int(pct * req.steps / 100.0)
            job_registry.publish(
                job_id,
                JobEvent(type="step_progress", data={"step": step, "total_steps": req.steps}),
            )

        pipe_req = PipelineRequest(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            seed=req.seed,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance=req.guidance,
            reference_image=req.reference_image,
            model_path=Path(entry.path),
            extra={},
        )

        pipe_result = await pipeline.generate(pipe_req, on_progress)
        webp_bytes = _encode_webp(pipe_result.rgb_bytes, pipe_result.width, pipe_result.height)

        result = RenderResult(
            webp_bytes=webp_bytes,
            model_used=entry.id,
            seed=req.seed,
            generation_time_s=time.monotonic() - started,
        )

        job_registry.publish(
            job_id,
            JobEvent(
                type="job_completed",
                data={
                    "model_used": entry.id,
                    "seed": req.seed,
                    "generation_time_s": result.generation_time_s,
                },
            ),
        )
        return result

    except Exception as exc:
        job_registry.publish(
            job_id,
            JobEvent(
                type="job_failed",
                data={
                    "error": "generation_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                },
            ),
        )
        raise
