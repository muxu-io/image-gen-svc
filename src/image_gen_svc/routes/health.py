"""GET /health — readiness probe."""

from __future__ import annotations

import contextlib

from fastapi import APIRouter

from image_gen_svc.config import ImageGenSvcConfig
from image_gen_svc.pipeline_registry import PipelineRegistry


def _gpu_vram_used_gb() -> float:
    """Best-effort VRAM read; returns 0.0 if torch/CUDA isn't available."""
    with contextlib.suppress(Exception):  # pragma: no cover — torch absent in unit tests
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / 1e9
    return 0.0


def build_router(cfg: ImageGenSvcConfig, pipeline_registry: PipelineRegistry) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "mock_only": cfg.mock_only,
            "models_loaded": pipeline_registry.loaded_names(),
            "gpu_vram_used_gb": 0.0 if cfg.mock_only else _gpu_vram_used_gb(),
        }

    return router
