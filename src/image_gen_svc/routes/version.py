"""GET /version — service version + key dependency versions.

Surfaces the diffusers/torch versions the running container was built with so
callers can detect drift between deployments. Always public (never gated by
the optional bearer-token auth)."""

from __future__ import annotations

from importlib import metadata

from fastapi import APIRouter


def _safe_version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/version")
    async def version() -> dict:
        return {
            "service": "image-gen-svc",
            "version": _safe_version("image-gen-svc") or "0.0.0+local",
            "diffusers": _safe_version("diffusers"),
            "torch": _safe_version("torch"),
            "transformers": _safe_version("transformers"),
        }

    return router
