"""Entry point: `python -m image_gen_svc`.

Wires concrete pipeline factories into the registry by inspecting each model
entry's architecture. Best-effort: a missing diffusers class for one
architecture doesn't prevent the service from starting — only that
architecture's renders fail."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

import uvicorn

from image_gen_svc.app import build_app
from image_gen_svc.config import ImageGenSvcConfig
from image_gen_svc.job_registry import JobRegistry
from image_gen_svc.model_registry import ModelRegistry
from image_gen_svc.pipeline_registry import PipelineFactory, PipelineRegistry

logger = logging.getLogger("image_gen_svc")


def _build_factories(
    model_registry: ModelRegistry,
) -> dict[str, PipelineFactory]:
    factories: dict[str, PipelineFactory] = {}

    for entry in model_registry.models.values():
        arch = entry.architecture
        if arch in factories:
            continue
        builder = _factory_for_architecture(arch, Path(entry.path))
        if builder is not None:
            factories[arch] = builder
    return factories


def _factory_for_architecture(arch: str, model_path: Path) -> Callable | None:
    try:
        if arch == "sdxl":
            from image_gen_svc.pipelines.sdxl import factory

            return factory(model_path)
        if arch == "chroma":
            from image_gen_svc.pipelines.chroma import factory

            return factory(model_path)
        if arch == "auraflow":
            from image_gen_svc.pipelines.auraflow import factory

            return factory(model_path)
        if arch == "z_image":
            from image_gen_svc.pipelines.z_image import factory

            return factory(model_path)
    except Exception:
        logger.exception("failed to build factory for architecture %s", arch)
        return None
    logger.warning("no pipeline adapter for architecture %r — skipping", arch)
    return None


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = ImageGenSvcConfig.from_env()
    logger.info("starting image-gen-svc on port %d (mock_only=%s)", cfg.port, cfg.mock_only)

    model_registry = ModelRegistry.load_default()
    pipeline_registry = PipelineRegistry(
        mock_only=cfg.mock_only,
        factories={} if cfg.mock_only else _build_factories(model_registry),
    )
    job_registry = JobRegistry()

    app = build_app(cfg, model_registry, pipeline_registry, job_registry)
    uvicorn.run(app, host="0.0.0.0", port=cfg.port, log_level="info")  # nosec B104
    return 0


if __name__ == "__main__":
    sys.exit(main())
