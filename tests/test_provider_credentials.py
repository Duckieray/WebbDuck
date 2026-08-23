from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core import provider_credentials as credentials


def _reset(monkeypatch, tmp_path: Path) -> Path:
    path = tmp_path / "provider_credentials.json"
    monkeypatch.setenv("WEBBDUCK_CREDENTIALS_FILE", str(path))
    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "CIVITAI_TOKEN",
        "CIVITAI_API_TOKEN",
        "CIVITAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    credentials._MANAGED_ENV.clear()
    return path


def test_settings_tokens_are_server_side_and_status_never_returns_secret(monkeypatch, tmp_path: Path):
    path = _reset(monkeypatch, tmp_path)
    credentials.save_provider_token("huggingface", "hf_super_secret")

    payload = credentials.provider_status()
    assert payload["providers"]["huggingface"] == {
        "configured": True,
        "source": "settings",
        "stored": True,
    }
    assert "hf_super_secret" not in json.dumps(payload)
    assert json.loads(path.read_text(encoding="utf-8"))["providers"]["huggingface"]["token"] == "hf_super_secret"
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600


def test_explicit_environment_override_wins_without_being_overwritten(monkeypatch, tmp_path: Path):
    _reset(monkeypatch, tmp_path)
    credentials.save_provider_token("civitai", "settings-token")
    monkeypatch.setenv("CIVITAI_API_KEY", "service-token")

    token, source = credentials.resolve_provider_token("civitai")
    assert token == "service-token"
    assert source == "environment"
    credentials.save_provider_token("civitai", "new-settings-token")
    assert os.environ["CIVITAI_API_KEY"] == "service-token"


def test_clear_removes_only_webbduck_managed_environment(monkeypatch, tmp_path: Path):
    _reset(monkeypatch, tmp_path)
    credentials.save_provider_token("huggingface", "first")
    assert os.environ["HF_TOKEN"] == "first"

    credentials.save_provider_token("huggingface", "second")
    assert os.environ["HF_TOKEN"] == "second"

    credentials.clear_provider_token("huggingface")
    assert "HF_TOKEN" not in os.environ
    assert credentials.provider_status()["providers"]["huggingface"]["configured"] is False


def test_unknown_provider_is_rejected(monkeypatch, tmp_path: Path):
    _reset(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="Unknown provider"):
        credentials.save_provider_token("mystery-cloud", "token")


def test_browser_module_does_not_persist_or_echo_tokens():
    source = (Path(__file__).resolve().parents[1] / "ui" / "modules" / "ProviderCredentialsSettings.js").read_text(encoding="utf-8")
    assert "type = 'password'" in source
    assert "localStorage" not in source
    assert "sessionStorage" not in source
    assert "Configured in WebbDuck Settings" in source
