"""POST /render — synchronous render with optional client-supplied job_id.

Two content-types are accepted on the same endpoint:

  application/json     — txt2img path. Body is a JSON RenderBody.
  multipart/form-data  — img2img path. Form has a `payload` part (JSON) and a
                         `reference_image` part (image bytes for IP-Adapter /
                         img2img conditioning).

The response body is the generated image as `image/webp`; metadata travels in
`X-Job-Id`, `X-Model-Used`, `X-Seed`, and `X-Generation-Time-Ms` headers.
Errors are JSON envelopes — see app.py exception handlers.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import random
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field, ValidationError

from image_gen_svc.config import ImageGenSvcConfig
from image_gen_svc.downloads import DownloadCoordinator
from image_gen_svc.inference import RenderRequest, run_render
from image_gen_svc.job_registry import JobEvent, JobRegistry
from image_gen_svc.model_registry import (
    ModelRegistry,
    NoSafeModelForAlias,
    resolve_guidance,
    resolve_steps,
)
from image_gen_svc.pipeline_registry import PipelineRegistry

# Diffusion seeds don't need cryptographic strength — `random` is fine and
# matches what every other diffusion UI uses. 32-bit range keeps the value
# JSON-safe for clients that round 64-bit ints (e.g. JS).
_SEED_MAX = 2**31 - 1


class RenderBody(BaseModel):
    model_config = {"extra": "forbid"}

    prompt: str
    negative_prompt: str | None = None
    # Optional. When omitted, the service generates a random 32-bit seed and
    # returns it in the X-Seed response header so the caller can reproduce
    # the result by passing it back on a subsequent request.
    seed: int | None = None
    model_id: str | None = None
    width: int = Field(gt=0, le=4096)
    height: int = Field(gt=0, le=4096)
    # steps/guidance fall through to the resolved model's `default_steps` /
    # `default_guidance`, then to architecture-tier defaults (see
    # model_registry.resolve_steps / resolve_guidance). Callers that don't
    # care can omit them; callers that do care can override.
    steps: int | None = Field(default=None, gt=0, le=200)
    guidance: float | None = Field(default=None, ge=0.0, le=30.0)
    job_id: str | None = None
    safe: bool = True


def _envelope(status: int, error: str, message: str, job_id: str | None) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": error, "message": message, "job_id": job_id},
    )


def build_router(
    cfg: ImageGenSvcConfig,
    model_registry: ModelRegistry,
    pipeline_registry: PipelineRegistry,
    job_registry: JobRegistry,
    download_coordinator: DownloadCoordinator,
) -> APIRouter:
    router = APIRouter()

    async def _parse_json(request: Request) -> RenderBody:
        try:
            raw = await request.json()
        except json.JSONDecodeError as exc:
            raise _envelope(400, "invalid_request", f"malformed JSON: {exc}", None) from exc
        try:
            return RenderBody.model_validate(raw)
        except ValidationError as exc:
            jid = raw.get("job_id") if isinstance(raw, dict) else None
            raise _envelope(400, "invalid_request", str(exc.errors()[0]), jid) from exc

    async def _parse_multipart(request: Request) -> tuple[RenderBody, bytes]:
        form = await request.form()
        payload_part = form.get("payload")
        if payload_part is None:
            raise _envelope(400, "invalid_request", "multipart missing 'payload' part", None)
        payload_raw = await payload_part.read() if hasattr(payload_part, "read") else payload_part
        if isinstance(payload_raw, bytes):
            payload_raw = payload_raw.decode("utf-8")
        try:
            payload_json = json.loads(payload_raw)
            body = RenderBody.model_validate(payload_json)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise _envelope(400, "invalid_request", str(exc), None) from exc

        ref_part = form.get("reference_image")
        if ref_part is None:
            raise _envelope(
                400, "invalid_request", "multipart missing 'reference_image' part", body.job_id
            )
        ref_bytes = await ref_part.read() if hasattr(ref_part, "read") else ref_part
        if not isinstance(ref_bytes, bytes):
            raise _envelope(
                400, "invalid_request", "reference_image must be a file part", body.job_id
            )
        return body, ref_bytes

    async def _execute(body: RenderBody, reference_image: bytes | None) -> Response:
        job_id = body.job_id or job_registry.create_job()
        if body.job_id is not None:
            job_registry.register_with_id(job_id)
        job_registry.publish(job_id, JobEvent(type="job_queued", data={}))

        # Pre-flight model presence check. If the registry entry has a download
        # URL and the file isn't on disk, fire the download in the background
        # and return 503 — the caller can subscribe to /events/<job_id> to
        # watch the model_loading event and retry once the model is ready.
        try:
            entry = model_registry.resolve(body.model_id, safe=body.safe)
        except KeyError as exc:
            raise _envelope(400, "invalid_request", str(exc), job_id) from exc
        except NoSafeModelForAlias as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "no_safe_model_for_alias",
                    "alias": exc.alias,
                    "message": str(exc),
                    "job_id": job_id,
                },
            ) from exc
        if not cfg.mock_only and not Path(entry.path).exists():
            if entry.url:
                job_registry.publish(
                    job_id,
                    JobEvent(type="model_loading", data={"model": entry.id, "url": entry.url}),
                )
                asyncio.create_task(
                    download_coordinator.ensure_present(
                        model_id=entry.id, url=entry.url, dest=Path(entry.path)
                    )
                )
                raise _envelope(
                    503,
                    "model_loading",
                    f"model {entry.id!r} is downloading; retry shortly",
                    job_id,
                )
            if entry.repo_id:
                job_registry.publish(
                    job_id,
                    JobEvent(
                        type="model_loading",
                        data={"model": entry.id, "repo_id": entry.repo_id},
                    ),
                )
                asyncio.create_task(
                    download_coordinator.ensure_snapshot_present(
                        model_id=entry.id,
                        repo_id=entry.repo_id,
                        dest=Path(entry.path),
                        allow_patterns=entry.allow_patterns,
                    )
                )
                raise _envelope(
                    503,
                    "model_loading",
                    f"model {entry.id!r} snapshot is downloading; retry shortly",
                    job_id,
                )

        req = RenderRequest(
            job_id=job_id,
            prompt=body.prompt,
            negative_prompt=body.negative_prompt,
            seed=body.seed if body.seed is not None else random.randint(0, _SEED_MAX),  # nosec B311
            model_id=body.model_id,
            width=body.width,
            height=body.height,
            steps=resolve_steps(entry, body.steps),
            guidance=resolve_guidance(entry, body.guidance),
            reference_image=reference_image,
            safe=body.safe,
        )
        try:
            result = await run_render(req, cfg, model_registry, pipeline_registry, job_registry)
        except (ValueError, KeyError) as exc:
            raise _envelope(400, "invalid_request", str(exc), job_id) from exc
        except Exception as exc:
            raise _envelope(
                500, "generation_failed", f"{type(exc).__name__}: {exc}", job_id
            ) from exc

        return Response(
            content=result.webp_bytes,
            media_type="image/webp",
            headers={
                "X-Job-Id": job_id,
                "X-Model-Used": result.model_used,
                "X-Seed": str(result.seed),
                "X-Generation-Time-Ms": str(int(result.generation_time_s * 1000)),
            },
        )

    def _check_auth(request: Request) -> None:
        if cfg.api_key is None:
            return
        expected = f"Bearer {cfg.api_key}"
        provided = request.headers.get("authorization", "")
        # Constant-time compare so token contents don't leak through timing.
        if not hmac.compare_digest(provided, expected):
            raise _envelope(401, "unauthorized", "missing or invalid bearer token", None)

    @router.post("/render")
    async def render(request: Request) -> Response:
        _check_auth(request)
        ctype = request.headers.get("content-type", "").lower()
        if ctype.startswith("multipart/form-data"):
            body, reference_image = await _parse_multipart(request)
            return await _execute(body, reference_image)
        body = await _parse_json(request)
        return await _execute(body, None)

    return router
