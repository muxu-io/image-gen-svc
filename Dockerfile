# image-gen-svc — single-stage CUDA image with diffusers + torch.
#
# Base notes:
#   - Ubuntu 24.04 ships python3 = 3.12, which satisfies pyproject's >=3.11
#     floor without a separate apt-installed interpreter.
#   - CUDA 12.8 is the highest 12.x line with a published *-ubuntu24.04 base.
#     The plain `runtime` variant (no cuDNN at the OS layer) is sufficient
#     because the torch pip wheel bundles nvidia-cudnn-cu12 as a dependency.
FROM nvidia/cuda:12.8.2-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    git curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install into a venv to sidestep PEP 668's externally-managed-environment
# guard on 24.04. With /opt/venv on PATH, `python3`, `pip`, and `poetry` all
# resolve to the venv-local copies for the rest of the build and at runtime.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml /app/
COPY src /app/src

RUN pip install --upgrade pip && \
    pip install /app && \
    pip install \
        torch \
        torchvision \
        "diffusers>=0.38" \
        "transformers>=4.45,<5" \
        "accelerate>=0.34" \
        "safetensors>=0.4" \
        "gguf>=0.10" \
        sentencepiece \
        protobuf
# transformers is capped below 5.x: 5.7.0 dropped CLIPTextModel.text_model
# (and likely related internals), which diffusers <= 0.38 still relies on.

ENV IMAGE_GEN_SVC_BASE_DIR=/app \
    IMAGE_GEN_SVC_PORT=7300 \
    IMAGE_GEN_SVC_MODELS_DIR=/models \
    HF_HOME=/root/.cache/huggingface

EXPOSE 7300
CMD ["python3", "-m", "image_gen_svc"]
