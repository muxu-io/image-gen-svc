"""FastAPI app factory for image-gen-svc.

Build dependencies are injected (cfg, model_registry, pipeline_registry,
job_registry) so tests can construct an app with a mock-only pipeline
registry and an empty model registry without touching disk."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from image_gen_svc.config import ImageGenSvcConfig
from image_gen_svc.downloads import DownloadCoordinator
from image_gen_svc.job_registry import JobRegistry
from image_gen_svc.model_registry import ModelRegistry
from image_gen_svc.pipeline_registry import PipelineRegistry
from image_gen_svc.routes import events as events_routes
from image_gen_svc.routes import health as health_routes
from image_gen_svc.routes import models as models_routes
from image_gen_svc.routes import render as render_routes
from image_gen_svc.routes import version as version_routes


def build_app(
    cfg: ImageGenSvcConfig,
    model_registry: ModelRegistry,
    pipeline_registry: PipelineRegistry,
    job_registry: JobRegistry,
    download_coordinator: DownloadCoordinator | None = None,
) -> FastAPI:
    app = FastAPI(title="image-gen-svc")
    download_coordinator = download_coordinator or DownloadCoordinator()
    app.state.cfg = cfg
    app.state.model_registry = model_registry
    app.state.pipeline_registry = pipeline_registry
    app.state.job_registry = job_registry
    app.state.download_coordinator = download_coordinator

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "http_error", "message": str(exc.detail), "job_id": None},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_request",
                "message": str(exc.errors()[0]) if exc.errors() else "schema validation failed",
                "job_id": None,
            },
        )

    app.include_router(health_routes.build_router(cfg, pipeline_registry))
    app.include_router(models_routes.build_router(model_registry))
    app.include_router(version_routes.build_router())
    app.include_router(
        render_routes.build_router(
            cfg, model_registry, pipeline_registry, job_registry, download_coordinator
        )
    )
    app.include_router(events_routes.build_router(job_registry))
    return app
