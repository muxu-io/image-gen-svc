"""AuraFlow pipeline adapter (Pony V7 base; AuraFlow 0.3 architecture).

Loads from a diffusers-format snapshot directory via `from_pretrained`.
AuraFlowPipeline does not implement `from_single_file` (as of diffusers
0.38), so single-file safetensors releases need the registry to point at a
`repo_id` snapshot with `allow_patterns` filtering down to the diffusers
core.

VRAM strategy is decided at adapter init from total device memory:
  - ≥ 16 GB: full speed, fully resident.
  - < 16 GB: `enable_sequential_cpu_offload()` pages submodules per-layer.
We use sequential rather than model-level offload because Pony V7's
transformer alone (~13.6 GB at fp16) plus activations clears a 16 GB card
on its own. Sequential pages at the submodule level → peak ~1-2 GB, at the
cost of ~5-10x slowdown from PCIe traffic.

Not unit-tested."""

from __future__ import annotations

import time
from pathlib import Path

from image_gen_svc.pipelines.base import (
    PipelineRequest,
    PipelineResult,
    ProgressCallback,
)


class AuraFlowPipelineAdapter:
    name = "auraflow"
    architecture = "auraflow"

    def __init__(self, model_path: Path):
        import torch
        from diffusers import AuraFlowPipeline

        self._pipe = AuraFlowPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
        )
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_vram_gb < 16.0:
            self._pipe.enable_sequential_cpu_offload()
        else:
            self._pipe.to("cuda")
        self._pipe.set_progress_bar_config(disable=True)
        self._pipe.vae.enable_tiling()

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
        try:
            import torch

            del self._pipe
            torch.cuda.empty_cache()
        except Exception:
            pass


def factory(model_path: Path):
    return lambda: AuraFlowPipelineAdapter(model_path)
