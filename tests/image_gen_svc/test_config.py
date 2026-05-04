from __future__ import annotations

from pathlib import Path

from image_gen_svc.config import ImageGenSvcConfig


def test_defaults_when_env_unset(monkeypatch):
    for var in (
        "IMAGE_GEN_SVC_BASE_DIR",
        "IMAGE_GEN_SVC_PORT",
        "IMAGE_GEN_SVC_MOCK_ONLY",
        "IMAGE_GEN_SVC_MODELS_DIR",
        "IMAGE_GEN_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = ImageGenSvcConfig.from_env()

    assert cfg.base_dir == Path("/app")
    assert cfg.port == 7300
    assert cfg.mock_only is False
    assert cfg.models_dir == Path("/models")
    assert cfg.api_key is None


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGE_GEN_SVC_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("IMAGE_GEN_SVC_PORT", "9999")
    monkeypatch.setenv("IMAGE_GEN_SVC_MOCK_ONLY", "true")
    monkeypatch.setenv("IMAGE_GEN_SVC_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("IMAGE_GEN_API_KEY", "secret")

    cfg = ImageGenSvcConfig.from_env()

    assert cfg.base_dir == tmp_path
    assert cfg.port == 9999
    assert cfg.mock_only is True
    assert cfg.models_dir == tmp_path / "models"
    assert cfg.api_key == "secret"


def test_mock_only_truthiness(monkeypatch):
    monkeypatch.delenv("IMAGE_GEN_SVC_MOCK_ONLY", raising=False)
    assert ImageGenSvcConfig.from_env().mock_only is False

    monkeypatch.setenv("IMAGE_GEN_SVC_MOCK_ONLY", "TRUE")
    assert ImageGenSvcConfig.from_env().mock_only is True

    monkeypatch.setenv("IMAGE_GEN_SVC_MOCK_ONLY", "")
    assert ImageGenSvcConfig.from_env().mock_only is False

    monkeypatch.setenv("IMAGE_GEN_SVC_MOCK_ONLY", "false")
    assert ImageGenSvcConfig.from_env().mock_only is False


def test_api_key_empty_string_is_none(monkeypatch):
    """An empty IMAGE_GEN_API_KEY env var means auth is disabled."""
    monkeypatch.setenv("IMAGE_GEN_API_KEY", "")
    assert ImageGenSvcConfig.from_env().api_key is None
