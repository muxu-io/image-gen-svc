"""SDXL pipeline adapter (RealVisXL V5 and other SDXL-architecture models).

Loaded lazily — only when the registry first acquires it. The constructor
imports torch and diffusers, loads the checkpoint into VRAM, and the
generate() method invokes the pipeline. `aclose()` deletes the pipe and
calls torch.cuda.empty_cache().

Not unit-tested. Verified by the operator smoke checklist (Task 22)."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

from image_gen_svc.pipelines.base import (
    PipelineRequest,
    PipelineResult,
    ProgressCallback,
)


class SDXLPipelineAdapter:
    name = "sdxl"
    architecture = "sdxl"

    def __init__(self, model_path: Path):
        import torch
        from diffusers import StableDiffusionXLPipeline

        self._pipe = StableDiffusionXLPipeline.from_single_file(
            str(model_path),
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")
        self._pipe.set_progress_bar_config(disable=True)
        # VAE tiling keeps the decode step within ~1 GB of VRAM so SDXL fits
        # comfortably on 16 GB cards alongside any other resident model.
        self._pipe.enable_vae_tiling()

    async def generate(
        self,
        req: PipelineRequest,
        on_progress: ProgressCallback,
    ) -> PipelineResult:
        import torch

        started = time.monotonic()
        gen = torch.Generator(device="cuda").manual_seed(req.seed)

        steps = req.steps

        def cb(_pipe, step_index, _timestep, callback_kwargs):
            on_progress(min(100.0, 100.0 * (step_index + 1) / steps))
            return callback_kwargs

        out = self._pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=steps,
            guidance_scale=req.guidance,
            width=req.width,
            height=req.height,
            generator=gen,
            callback_on_step_end=cb,
        )
        on_progress(100.0)

        img = out.images[0].convert("RGB")
        return PipelineResult(
            rgb_bytes=img.tobytes(),
            width=img.width,
            height=img.height,
            generation_time_s=time.monotonic() - started,
        )

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            import torch

            del self._pipe
            torch.cuda.empty_cache()


def factory(model_path: Path):
    """Returns a callable suitable for `PipelineRegistry.factories['sdxl']`."""
    return lambda: SDXLPipelineAdapter(model_path)
