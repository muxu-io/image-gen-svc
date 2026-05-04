"""GET /models — registry contents.

Each entry exposes:
  - registry metadata (path, aliases, architecture, vram_gb, license, ...)
  - `safe: bool` — model is not specifically NSFW-tuned; the request-time
    `safe` filter on /render uses this. False ≠ "renders NSFW only" — the
    designation gates filter behavior, not generation behavior.
  - `loaded: bool` — checkpoint file is on disk vs. would need lazy download.

The `loaded` flag is computed at request time from the filesystem so callers
can warn the operator about an impending wait before issuing the render.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from image_gen_svc.model_registry import ModelRegistry


def build_router(model_registry: ModelRegistry) -> APIRouter:
    router = APIRouter()

    @router.get("/models")
    async def list_models() -> dict:
        out = model_registry.to_dict()
        for model_id, entry in out["models"].items():
            entry["loaded"] = Path(model_registry.models[model_id].path).exists()
        return out

    return router
