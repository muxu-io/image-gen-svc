from __future__ import annotations

import inspect

import pytest
import yaml

from image_gen_svc.model_registry import (
    ModelEntry,
    ModelRegistry,
    NoSafeModelForAlias,
    resolve_guidance,
    resolve_steps,
)


def test_load_default_returns_packaged_registry():
    reg = ModelRegistry.load_default()

    assert "z-image-turbo" in reg.models
    assert "animagine-xl-4.0" in reg.models
    assert "chroma-1-hd" in reg.models
    assert "realvis-xl-v5" in reg.models
    assert "pony-v7-base" in reg.models
    assert reg.default_alias == "photorealistic"
    assert reg.default_models["photorealistic"] == "chroma-1-hd"


def test_load_default_takes_no_arguments():
    """The runtime overlay path is gone; load_default takes no path argument."""
    sig = inspect.signature(ModelRegistry.load_default)
    assert list(sig.parameters) == []


def test_from_raw_user_override():
    raw = yaml.safe_load("""
    models:
      custom-sdxl:
        path: /models/custom.safetensors
        aliases: [photorealistic, my_alias]
        architecture: sdxl
        vram_gb: 7
        seed_stability: high
        license: cc0
    default_alias: photorealistic
    """)
    reg = ModelRegistry._from_raw(raw)

    assert list(reg.models.keys()) == ["custom-sdxl"]
    entry = reg.models["custom-sdxl"]
    assert isinstance(entry, ModelEntry)
    assert entry.architecture == "sdxl"
    assert entry.aliases == ["photorealistic", "my_alias"]


def test_safe_field_defaults_false():
    """Entries without `safe:` parse as safe=False (uncensored is the design
    intent at the model-availability layer)."""
    raw = yaml.safe_load("""
    models:
      no-safe-flag:
        path: /m.safetensors
        aliases: []
        architecture: sdxl
        vram_gb: 1
        seed_stability: high
        license: x
    default_alias: no-safe-flag
    """)
    reg = ModelRegistry._from_raw(raw)
    assert reg.models["no-safe-flag"].safe is False


def test_resolve_explicit_id_bypasses_safety():
    """Explicit model_id lookups are not filtered by `safe`. Caller has
    accepted responsibility for the policy."""
    reg = ModelRegistry.load_default()
    assert reg.resolve("chroma-1-hd", safe=True).id == "chroma-1-hd"
    assert reg.resolve("pony-v7-base", safe=True).id == "pony-v7-base"


def test_resolve_filters_unsafe_when_safe_true():
    """Alias resolution under safe=True picks the safe candidate."""
    reg = ModelRegistry.load_default()
    assert reg.resolve("photorealistic", safe=True).id == "z-image-turbo"
    assert reg.resolve("anime", safe=True).id == "animagine-xl-4.0"


def test_resolve_alias_unfiltered_when_safe_false():
    """Alias resolution under safe=False uses the default_models tiebreaker
    (which preserves the pre-spec default of chroma/pony)."""
    reg = ModelRegistry.load_default()
    assert reg.resolve("photorealistic", safe=False).id == "chroma-1-hd"
    assert reg.resolve("anime", safe=False).id == "pony-v7-base"
    assert reg.resolve("photorealistic_lowvram", safe=False).id == "realvis-xl-v5"


def test_resolve_default_when_none():
    reg = ModelRegistry.load_default()
    # default_alias=photorealistic; safe=True → z-image-turbo (chroma filtered).
    assert reg.resolve(None, safe=True).id == "z-image-turbo"
    # safe=False → tiebreaker picks chroma.
    assert reg.resolve(None, safe=False).id == "chroma-1-hd"


def test_resolve_unknown_raises():
    reg = ModelRegistry.load_default()
    with pytest.raises(KeyError, match="unknown model_id or alias: 'nope'"):
        reg.resolve("nope", safe=True)


def test_resolve_no_safe_model_raises():
    """Alias with matches but no safe candidates raises NoSafeModelForAlias."""
    raw = yaml.safe_load("""
    models:
      only-unsafe:
        path: /m.safetensors
        aliases: [risky]
        architecture: sdxl
        vram_gb: 1
        seed_stability: high
        license: x
        safe: false
    default_alias: risky
    """)
    reg = ModelRegistry._from_raw(raw)
    with pytest.raises(NoSafeModelForAlias) as exc_info:
        reg.resolve("risky", safe=True)
    assert exc_info.value.alias == "risky"

    # safe=False still resolves.
    assert reg.resolve("risky", safe=False).id == "only-unsafe"


def test_resolve_default_models_tiebreaker():
    """When multiple unsafe candidates match an alias, default_models picks
    the preferred one. Without it, the resolver falls back to deterministic
    sort by id."""
    raw = yaml.safe_load("""
    models:
      zebra:
        path: /z.safetensors
        aliases: [shared]
        architecture: sdxl
        vram_gb: 1
        seed_stability: high
        license: x
      apple:
        path: /a.safetensors
        aliases: [shared]
        architecture: sdxl
        vram_gb: 1
        seed_stability: high
        license: x
    default_models:
      shared: zebra
    default_alias: shared
    """)
    reg = ModelRegistry._from_raw(raw)
    assert reg.resolve("shared", safe=False).id == "zebra"

    raw_no_pref = yaml.safe_load("""
    models:
      zebra:
        path: /z.safetensors
        aliases: [shared]
        architecture: sdxl
        vram_gb: 1
        seed_stability: high
        license: x
      apple:
        path: /a.safetensors
        aliases: [shared]
        architecture: sdxl
        vram_gb: 1
        seed_stability: high
        license: x
    default_alias: shared
    """)
    reg2 = ModelRegistry._from_raw(raw_no_pref)
    # No tiebreaker → deterministic sort picks 'apple'.
    assert reg2.resolve("shared", safe=False).id == "apple"


def test_id_lookup_takes_precedence_over_alias():
    raw = yaml.safe_load("""
    models:
      alpha:
        path: /models/a.safetensors
        aliases: [beta]
        architecture: sdxl
        vram_gb: 7
        seed_stability: high
        license: x
      beta:
        path: /models/b.safetensors
        aliases: []
        architecture: sdxl
        vram_gb: 7
        seed_stability: high
        license: x
    default_alias: alpha
    """)
    reg = ModelRegistry._from_raw(raw)
    assert reg.resolve("beta", safe=True).id == "beta"


def test_to_dict_for_models_endpoint():
    reg = ModelRegistry.load_default()
    payload = reg.to_dict()

    assert payload["default_alias"] == "photorealistic"
    assert "chroma-1-hd" in payload["models"]
    assert payload["models"]["chroma-1-hd"]["architecture"] == "chroma"
    assert payload["models"]["chroma-1-hd"]["safe"] is False
    assert payload["models"]["z-image-turbo"]["safe"] is True
    assert payload["default_models"]["photorealistic"] == "chroma-1-hd"


def test_to_dict_exposes_effective_default_steps_and_guidance():
    """Each entry surfaces the effective values /render will use when the
    caller omits steps/guidance. Uses entry-declared defaults where present,
    architecture-tier fallbacks otherwise."""
    reg = ModelRegistry.load_default()
    payload = reg.to_dict()
    z = payload["models"]["z-image-turbo"]
    chroma = payload["models"]["chroma-1-hd"]

    assert z["default_steps"] == 4
    assert z["default_guidance"] == 3.5
    assert chroma["default_steps"] == 28
    assert chroma["default_guidance"] == 4.0


def _entry(architecture: str, default_steps=None, default_guidance=None) -> ModelEntry:
    return ModelEntry(
        id="t",
        path="/m",
        aliases=[],
        architecture=architecture,
        vram_gb=1.0,
        seed_stability="high",
        license="x",
        default_steps=default_steps,
        default_guidance=default_guidance,
    )


def test_resolve_steps_caller_value_wins():
    e = _entry("sdxl", default_steps=25)
    assert resolve_steps(e, request_steps=12) == 12


def test_resolve_steps_falls_back_to_entry_default():
    e = _entry("sdxl", default_steps=22)
    assert resolve_steps(e, request_steps=None) == 22


def test_resolve_steps_falls_back_to_architecture_default():
    e = _entry("z_image", default_steps=None)
    assert resolve_steps(e, request_steps=None) == 4

    e_sdxl = _entry("sdxl", default_steps=None)
    assert resolve_steps(e_sdxl, request_steps=None) == 25


def test_resolve_steps_unknown_architecture_uses_global_fallback():
    e = _entry("brand-new-arch", default_steps=None)
    assert resolve_steps(e, request_steps=None) == 25


def test_resolve_guidance_caller_value_wins():
    e = _entry("sdxl", default_guidance=7.0)
    assert resolve_guidance(e, request_guidance=2.5) == 2.5


def test_resolve_guidance_falls_back_to_entry_default():
    e = _entry("sdxl", default_guidance=6.5)
    assert resolve_guidance(e, request_guidance=None) == 6.5


def test_resolve_guidance_falls_back_to_architecture_default():
    e = _entry("chroma", default_guidance=None)
    assert resolve_guidance(e, request_guidance=None) == 4.0

    e_z = _entry("z_image", default_guidance=None)
    assert resolve_guidance(e_z, request_guidance=None) == 3.5
