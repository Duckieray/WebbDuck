"""Server-side storage for optional model-provider credentials.

Provider tokens are intentionally kept out of browser/localStorage state.  This
module stores only on the local WebbDuck host and exposes status without ever
returning secret values to API clients.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_SCHEMA_VERSION = 1
_PROVIDER_ENV = {
    "huggingface": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
    "civitai": ("CIVITAI_TOKEN", "CIVITAI_API_TOKEN", "CIVITAI_API_KEY"),
}
_CANONICAL_ENV = {
    "huggingface": "HF_TOKEN",
    "civitai": "CIVITAI_TOKEN",
}
_MANAGED_ENV: dict[str, str] = {}


def credentials_path() -> Path:
    raw = str(os.getenv("WEBBDUCK_CREDENTIALS_FILE") or "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".webbduck" / "provider_credentials.json"


def _empty_store() -> dict[str, Any]:
    return {"version": _SCHEMA_VERSION, "providers": {}}


def _load_store() -> dict[str, Any]:
    path = credentials_path()
    if not path.exists():
        return _empty_store()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_store()
    if not isinstance(value, dict) or value.get("version") != _SCHEMA_VERSION:
        return _empty_store()
    providers = value.get("providers") if isinstance(value.get("providers"), dict) else {}
    return {"version": _SCHEMA_VERSION, "providers": providers}


def _write_store(store: dict[str, Any]) -> None:
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")
    try:
        temp.chmod(0o600)
    except OSError:
        pass
    os.replace(temp, path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _validate_provider(provider: str) -> str:
    provider = str(provider or "").strip().lower()
    if provider not in _PROVIDER_ENV:
        raise ValueError(f"Unknown provider: {provider}")
    return provider


def stored_provider_token(provider: str) -> str:
    provider = _validate_provider(provider)
    raw = _load_store()["providers"].get(provider)
    if not isinstance(raw, dict):
        return ""
    return str(raw.get("token") or "").strip()


def environment_provider_token(provider: str) -> tuple[str, str | None]:
    provider = _validate_provider(provider)
    for name in _PROVIDER_ENV[provider]:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value, name
    return "", None


def resolve_provider_token(provider: str) -> tuple[str, str | None]:
    """Resolve provider token, preferring explicit environment overrides."""
    provider = _validate_provider(provider)
    token, env_name = environment_provider_token(provider)
    if token:
        managed = _MANAGED_ENV.get(provider)
        source = "settings" if managed and managed == token and env_name == _CANONICAL_ENV[provider] else "environment"
        return token, source
    token = stored_provider_token(provider)
    return (token, "settings") if token else ("", None)


def provider_status() -> dict[str, Any]:
    """Return non-secret provider configuration status."""
    providers: dict[str, Any] = {}
    for provider in _PROVIDER_ENV:
        token, source = resolve_provider_token(provider)
        providers[provider] = {
            "configured": bool(token),
            "source": source,
            "stored": bool(stored_provider_token(provider)),
        }
    return {"providers": providers}


def save_provider_token(provider: str, token: str) -> None:
    provider = _validate_provider(provider)
    token = str(token or "").strip()
    if not token:
        raise ValueError("Token cannot be empty")
    store = _load_store()
    store["providers"][provider] = {"token": token}
    _write_store(store)
    _sync_managed_environment(provider)


def clear_provider_token(provider: str) -> None:
    provider = _validate_provider(provider)
    store = _load_store()
    store["providers"].pop(provider, None)
    _write_store(store)
    _sync_managed_environment(provider)


def _sync_managed_environment(provider: str) -> None:
    """Refresh only environment values that WebbDuck itself manages.

    An explicit shell/service environment variable always wins.  If WebbDuck
    previously populated the canonical variable from Settings, a Settings update
    may replace or remove that managed value in the running process.
    """
    provider = _validate_provider(provider)
    canonical = _CANONICAL_ENV[provider]
    current = str(os.getenv(canonical) or "").strip()
    previous_managed = _MANAGED_ENV.get(provider)
    if current and (not previous_managed or current != previous_managed):
        return

    stored = stored_provider_token(provider)
    if stored:
        os.environ[canonical] = stored
        _MANAGED_ENV[provider] = stored
    else:
        if previous_managed and os.getenv(canonical) == previous_managed:
            os.environ.pop(canonical, None)
        _MANAGED_ENV.pop(provider, None)


def apply_provider_credentials_to_environment() -> None:
    """Populate canonical provider variables from Settings when not overridden."""
    for provider in _PROVIDER_ENV:
        _sync_managed_environment(provider)
