"""GET /events/<job_id> — SSE stream of JobEvents."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from image_gen_svc.job_registry import JobRegistry


def _format_sse(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def build_router(job_registry: JobRegistry) -> APIRouter:
    router = APIRouter()

    @router.get("/events/{job_id}")
    async def events(job_id: str) -> StreamingResponse:
        async def gen() -> AsyncIterator[bytes]:
            async for ev in job_registry.subscribe(job_id):
                yield _format_sse(ev.type, ev.data)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
