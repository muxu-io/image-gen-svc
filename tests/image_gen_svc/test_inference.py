from __future__ import annotations

from pathlib import Path

import pytest

from image_gen_svc.config import ImageGenSvcConfig
from image_gen_svc.inference import RenderRequest, run_render
from image_gen_svc.job_registry import JobRegistry
from image_gen_svc.model_registry import ModelRegistry
from image_gen_svc.pipeline_registry import PipelineRegistry


@pytest.fixture
def cfg(tmp_path: Path) -> ImageGenSvcConfig:
    return ImageGenSvcConfig(
        base_dir=tmp_path,
        port=7300,
        mock_only=True,
        models_dir=tmp_path / "models",
    )


@pytest.fixture
def model_reg() -> ModelRegistry:
    return ModelRegistry.load_default()


@pytest.fixture
def pipe_reg() -> PipelineRegistry:
    return PipelineRegistry(mock_only=True, factories={})


@pytest.fixture
def job_reg() -> JobRegistry:
    return JobRegistry()


def _req(job_id: str, **overrides) -> RenderRequest:
    base = dict(
        job_id=job_id,
        prompt="a contemplative scholar",
        negative_prompt=None,
        seed=42,
        model_id=None,
        width=64,
        height=64,
        steps=20,
        guidance=7.5,
        reference_image=None,
        safe=True,
    )
    base.update(overrides)
    return RenderRequest(**base)


@pytest.mark.asyncio
async def test_render_returns_valid_webp_bytes(cfg, model_reg, pipe_reg, job_reg):
    job_id = job_reg.create_job()
    # Default RenderRequest has safe=True → photorealistic alias resolves to
    # z-image-turbo (chroma is filtered as unsafe).
    result = await run_render(_req(job_id), cfg, model_reg, pipe_reg, job_reg)

    assert result.webp_bytes[:4] == b"RIFF"
    assert result.model_used == "z-image-turbo"
    assert result.seed == 42
    assert result.generation_time_s >= 0.0


@pytest.mark.asyncio
async def test_render_safe_false_uses_unsafe_default(cfg, model_reg, pipe_reg, job_reg):
    """safe=False on the request preserves the legacy chroma default."""
    job_id = job_reg.create_job()
    result = await run_render(_req(job_id, safe=False), cfg, model_reg, pipe_reg, job_reg)
    assert result.model_used == "chroma-1-hd"


@pytest.mark.asyncio
async def test_render_emits_lifecycle_events(cfg, model_reg, pipe_reg, job_reg):
    job_id = job_reg.create_job()
    await run_render(
        _req(job_id, model_id="realvis-xl-v5", width=32, height=32, steps=4, guidance=1.0),
        cfg,
        model_reg,
        pipe_reg,
        job_reg,
    )

    received = [ev async for ev in job_reg.subscribe(job_id)]
    types = [e.type for e in received]
    assert types[0] == "job_started"
    assert types[-1] == "job_completed"
    assert "step_progress" in types
    # Final step_progress event should reach total_steps.
    last_progress = [e for e in received if e.type == "step_progress"][-1]
    assert last_progress.data["total_steps"] == 4
    assert last_progress.data["step"] == 4
    last = received[-1].data
    assert last["model_used"] == "realvis-xl-v5"
    assert last["seed"] == 42


@pytest.mark.asyncio
async def test_render_failed_event_on_unknown_model(cfg, model_reg, pipe_reg, job_reg):
    job_id = job_reg.create_job()
    with pytest.raises(KeyError):
        await run_render(
            _req(job_id, model_id="does-not-exist", width=32, height=32, steps=1, guidance=1.0),
            cfg,
            model_reg,
            pipe_reg,
            job_reg,
        )

    received = [ev async for ev in job_reg.subscribe(job_id)]
    assert received[-1].type == "job_failed"
    assert "does-not-exist" in received[-1].data["message"]
    assert received[-1].data["error"] == "generation_failed"
