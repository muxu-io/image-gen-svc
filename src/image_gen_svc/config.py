"""image-gen-svc configuration. Read from env in production, constructed
explicitly in tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageGenSvcConfig:
    base_dir: Path
    port: int = 7300
    mock_only: bool = False
    models_dir: Path = Path("/models")
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> ImageGenSvcConfig:
        base = Path(os.environ.get("IMAGE_GEN_SVC_BASE_DIR", "/app"))
        return cls(
            base_dir=base,
            port=int(os.environ.get("IMAGE_GEN_SVC_PORT", "7300")),
            mock_only=os.environ.get("IMAGE_GEN_SVC_MOCK_ONLY", "").lower() == "true",
            models_dir=Path(os.environ.get("IMAGE_GEN_SVC_MODELS_DIR", "/models")),
            api_key=os.environ.get("IMAGE_GEN_API_KEY") or None,
        )
