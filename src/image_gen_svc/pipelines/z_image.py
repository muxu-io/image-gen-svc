"""Z-Image-Turbo pipeline adapter.

Distilled DiT (Tongyi-MAI/Z-Image-Turbo); 4-step inference. The model on disk
is a Hugging Face snapshot directory (Qwen2.5-VL-7B text encoder + DiT + VAE),
so we use `from_pretrained` rather than `from_single_file`.

Loaded in bfloat16 (not fp16) — the Qwen2.5-VL-7B text encoder ships in bf16
and produces silent NaNs in attention under fp16 on some prompts, which the
VAE then renders as black images.

VRAM strategy is decided at adapter init from total device memory:
  - ≥ 24 GB (3090/4090, A5000+, L40, H100): pipeline runs fully resident.
    Peak ~22 GB, full per-step speed.
  - < 24 GB (5060 Ti, 4070, etc.): `enable_model_cpu_offload()` pages each
    component on/off as it runs. Peak ~10 GB, 2-5x slower per step but the
    pipeline actually fits.

Operators wanting ~16 GB at full speed should build a derivative image that
adds bitsandbytes int4 quant on the text encoder.

Not unit-tested. Verified by the operator smoke checklist."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

from image_gen_svc.pipelines.base import (
    PipelineRequest,
    PipelineResult,
    ProgressCallback,
)


class ZImagePipelineAdapter:
    name = "z-image-turbo"
    architecture = "z_image"

    def __init__(self, model_path: Path):
        import torch
        from diffusers import ZImagePipeline

        self._pipe = ZImagePipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
        )
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_vram_gb < 24.0:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to("cuda")
        self._pipe.vae.enable_tiling()
        self._pipe.set_progress_bar_config(disable=True)

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
    return lambda: ZImagePipelineAdapter(model_path)
