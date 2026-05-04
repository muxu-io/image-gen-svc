"""Model registry — parses the registry.yml that lists available checkpoints.

The render-time API resolves a `model_id` (or alias, including the configured
`default_alias`) to a `ModelEntry` whose `architecture` field selects the
pipeline adapter and whose `path` is passed to the loader. Resolution applies
a safety filter: callers pass `safe=True` to restrict alias matches to entries
marked `safe: true` in the registry; explicit model_id lookups bypass the
filter (the caller has taken responsibility)."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import yaml

_DEFAULT_REGISTRY_RESOURCE = "default_registry.yml"


# Architecture-tier fallbacks, used only when neither the request nor the
# registry entry specifies a value. Tuned to be sensible "first impression"
# defaults per family — callers that care about quality should override.
_ARCH_DEFAULT_STEPS: dict[str, int] = {
    "z_image": 4,  # distilled, designed for 4-step inference
    "sdxl": 25,
    "chroma": 28,
    "auraflow": 25,
    "mock": 4,
}
_ARCH_DEFAULT_GUIDANCE: dict[str, float] = {
    "z_image": 3.5,
    "sdxl": 7.0,
    "chroma": 4.0,  # FLUX-class flow-matching prefers low guidance
    "auraflow": 5.0,
    "mock": 3.5,
}
_FALLBACK_STEPS = 25
_FALLBACK_GUIDANCE = 7.0


def resolve_steps(entry: ModelEntry, request_steps: int | None) -> int:
    """Caller value > entry's `default_steps` > architecture-tier default."""
    if request_steps is not None:
        return request_steps
    if entry.default_steps is not None:
        return entry.default_steps
    return _ARCH_DEFAULT_STEPS.get(entry.architecture, _FALLBACK_STEPS)


def resolve_guidance(entry: ModelEntry, request_guidance: float | None) -> float:
    """Caller value > entry's `default_guidance` > architecture-tier default."""
    if request_guidance is not None:
        return request_guidance
    if entry.default_guidance is not None:
        return entry.default_guidance
    return _ARCH_DEFAULT_GUIDANCE.get(entry.architecture, _FALLBACK_GUIDANCE)


class NoSafeModelForAlias(Exception):
    """Raised by `resolve(..., safe=True)` when an alias has matches but none
    are `safe`. The render route maps this to HTTP 400."""

    def __init__(self, alias: str):
        super().__init__(f"no safe model available for alias: {alias!r}")
        self.alias = alias


@dataclass(frozen=True)
class ModelEntry:
    id: str
    path: str
    aliases: list[str]
    architecture: str
    vram_gb: float
    seed_stability: str
    license: str
    url: str | None = None
    repo_id: str | None = None
    # snapshot-only: forwarded to huggingface_hub.snapshot_download to skip
    # bloat folders (gguf/, lora/, comfy_nodes/, ...) when only the diffusers
    # core is needed. Ignored for url-shaped entries.
    allow_patterns: list[str] | None = None
    sha256: str | None = None
    safe: bool = False
    default_steps: int | None = None
    default_guidance: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRegistry:
    models: dict[str, ModelEntry]
    default_alias: str
    default_models: dict[str, str]

    @classmethod
    def load_default(cls) -> ModelRegistry:
        """Load the registry bundled with the package. Downstream consumers who
        need extra models build a derivative image (replace `default_registry.yml`)
        — the service does not honor a runtime overlay."""
        raw = yaml.safe_load(
            resources.files("image_gen_svc").joinpath(_DEFAULT_REGISTRY_RESOURCE).read_text()
        )
        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: dict[str, Any]) -> ModelRegistry:
        models: dict[str, ModelEntry] = {}
        for model_id, body in (raw.get("models") or {}).items():
            known = {
                "path",
                "aliases",
                "architecture",
                "vram_gb",
                "seed_stability",
                "license",
                "url",
                "repo_id",
                "allow_patterns",
                "sha256",
                "safe",
                "default_steps",
                "default_guidance",
            }
            extra = {k: v for k, v in body.items() if k not in known}
            models[model_id] = ModelEntry(
                id=model_id,
                path=body["path"],
                aliases=list(body.get("aliases") or []),
                architecture=body["architecture"],
                vram_gb=float(body.get("vram_gb", 0.0)),
                seed_stability=body.get("seed_stability", "unknown"),
                license=body.get("license", "unknown"),
                url=body.get("url"),
                repo_id=body.get("repo_id"),
                allow_patterns=body.get("allow_patterns"),
                sha256=body.get("sha256"),
                safe=bool(body.get("safe", False)),
                default_steps=body.get("default_steps"),
                default_guidance=body.get("default_guidance"),
                extra=extra,
            )
        default_alias = raw.get("default_alias")
        if not default_alias:
            raise ValueError("registry must define `default_alias`")
        default_models = dict(raw.get("default_models") or {})
        return cls(models=models, default_alias=default_alias, default_models=default_models)

    def resolve(self, identifier: str | None, *, safe: bool) -> ModelEntry:
        target = identifier or self.default_alias

        # Explicit model_id passes through unchanged. Caller has taken
        # responsibility for the safety axis.
        if target in self.models:
            return self.models[target]

        candidates = [m for m in self.models.values() if target in m.aliases]
        if not candidates:
            raise KeyError(f"unknown model_id or alias: {target!r}")

        if safe:
            candidates = [m for m in candidates if m.safe]
            if not candidates:
                raise NoSafeModelForAlias(target)

        if len(candidates) == 1:
            return candidates[0]

        preferred_id = self.default_models.get(target)
        if preferred_id:
            preferred = next((c for c in candidates if c.id == preferred_id), None)
            if preferred is not None:
                return preferred
        return sorted(candidates, key=lambda c: c.id)[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_alias": self.default_alias,
            "default_models": dict(self.default_models),
            "models": {
                m.id: {
                    "path": m.path,
                    "aliases": m.aliases,
                    "architecture": m.architecture,
                    "vram_gb": m.vram_gb,
                    "seed_stability": m.seed_stability,
                    "license": m.license,
                    "safe": m.safe,
                    # Effective defaults: what /render will use if the caller
                    # omits steps/guidance. Reflects the registry's declared
                    # defaults, falling back to architecture-tier defaults.
                    "default_steps": resolve_steps(m, None),
                    "default_guidance": resolve_guidance(m, None),
                }
                for m in self.models.values()
            },
        }
