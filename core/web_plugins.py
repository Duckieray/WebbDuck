"""Generic WebbDuck web-plugin discovery and mounting."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from fastapi import FastAPI
from fastapi.routing import APIRouter
from fastapi.staticfiles import StaticFiles

from webbduck.core.captioning_config import get_plugins_dirs

PLUGIN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
REMOTE_STATE_DIR = Path.home() / ".webbduck" / "plugin_state"
REMOTE_STATE_FILE = REMOTE_STATE_DIR / "remote_web_plugins.json"
REMOTE_MANIFEST_PATHS = (
    "webbduck-plugin.json",
    ".well-known/webbduck-plugin.json",
    "plugin.json",
)


@dataclass(frozen=True)
class WebPlugin:
    plugin_id: str
    name: str
    description: str
    order: int
    plugin_dir: Path
    ui_dir: Path
    entry: str
    icon: str | None = None
    backend_file: Path | None = None

    @property
    def view_name(self) -> str:
        return f"plugin-{self.plugin_id}"

    @property
    def ui_url(self) -> str:
        return f"/plugins/web/{self.plugin_id}/ui/{self.entry}"

    @property
    def api_base(self) -> str:
        return f"/plugins/web/{self.plugin_id}/api"

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.plugin_id,
            "name": self.name,
            "description": self.description,
            "order": self.order,
            "entry": self.entry,
            "icon": self.icon,
            "view": self.view_name,
            "ui_url": self.ui_url,
            "api_base": self.api_base,
            "source": "local",
        }


@dataclass(frozen=True)
class RemoteWebPlugin:
    plugin_id: str
    name: str
    description: str
    order: int
    ui_url: str
    api_base: str | None
    icon: str | None
    remote_base: str

    @property
    def view_name(self) -> str:
        return f"plugin-{self.plugin_id}"

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.plugin_id,
            "name": self.name,
            "description": self.description,
            "order": self.order,
            "entry": "",
            "icon": self.icon,
            "view": self.view_name,
            "ui_url": self.ui_url,
            "api_base": self.api_base,
            "source": "remote",
            "remote_base": self.remote_base,
        }


def discover_web_plugins() -> list[WebPlugin]:
    discovered: dict[str, WebPlugin] = {}
    for plugins_root in get_plugins_dirs():
        webapps_dir = plugins_root / "webapps"
        if not webapps_dir.exists() or not webapps_dir.is_dir():
            continue

        for item in sorted(webapps_dir.iterdir(), key=lambda p: p.name.lower()):
            if not item.is_dir():
                continue

            plugin = _load_plugin_from_dir(item)
            if plugin is None:
                continue

            # Respect plugin directory search priority:
            # first discovered plugin_id wins.
            if plugin.plugin_id not in discovered:
                discovered[plugin.plugin_id] = plugin

    return sorted(
        discovered.values(),
        key=lambda p: (p.order, p.name.lower(), p.plugin_id),
    )


def mount_web_plugin_assets(app: FastAPI, plugins: list[WebPlugin]) -> None:
    for plugin in plugins:
        mount_path = f"/plugins/web/{plugin.plugin_id}/ui"
        app.mount(
            mount_path,
            StaticFiles(directory=str(plugin.ui_dir)),
            name=f"plugin_ui_{plugin.plugin_id}",
        )


def include_web_plugin_routers(app: FastAPI, plugins: list[WebPlugin]) -> None:
    for plugin in plugins:
        if plugin.backend_file is None or not plugin.backend_file.exists():
            continue

        spec = importlib.util.spec_from_file_location(
            f"webbduck_webplugin_{plugin.plugin_id}",
            str(plugin.backend_file),
        )
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        # Ensure module is available during execution (required by dataclasses on some Python builds).
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        router_factory = getattr(module, "get_router", None)
        if not callable(router_factory):
            continue

        router = _call_router_factory(router_factory, plugin)
        if not isinstance(router, APIRouter):
            continue

        app.include_router(router, prefix=f"/plugins/web/{plugin.plugin_id}/api")


def build_public_web_plugin_list(local_plugins: list[WebPlugin]) -> list[dict[str, Any]]:
    local_public = [plugin.to_public() for plugin in local_plugins]
    local_ids = {str(plugin["id"]) for plugin in local_public}
    remote_public = [
        plugin for plugin in list_remote_web_plugins() if str(plugin.get("id", "")) not in local_ids
    ]
    merged = [*local_public, *remote_public]
    merged.sort(
        key=lambda plugin: (
            int(plugin.get("order", 100)),
            str(plugin.get("name", "")).lower(),
            str(plugin.get("id", "")),
        )
    )
    return merged


def list_remote_web_plugins() -> list[dict[str, Any]]:
    plugins: list[RemoteWebPlugin] = []
    for raw in _load_remote_records():
        plugin = _coerce_remote_plugin(raw)
        if plugin is not None:
            plugins.append(plugin)
    plugins.sort(key=lambda p: (p.order, p.name.lower(), p.plugin_id))
    return [plugin.to_public() for plugin in plugins]


def connect_remote_web_plugin(
    base_url: str,
    *,
    reserved_ids: set[str] | None = None,
) -> dict[str, Any]:
    base = normalize_remote_base(base_url)
    if base is None:
        raise ValueError("Invalid base_url. Use host:port or http(s)://host:port.")

    manifest = _discover_remote_manifest(base)
    plugin = _build_remote_plugin(base, manifest)
    if reserved_ids and plugin.plugin_id in reserved_ids:
        raise ValueError(
            f"Plugin id '{plugin.plugin_id}' conflicts with an installed local plugin."
        )

    records = _load_remote_records()
    next_records = [
        record
        for record in records
        if str(record.get("id", "")).strip().lower() != plugin.plugin_id
    ]
    next_records.append(_remote_plugin_to_record(plugin))
    _save_remote_records(next_records)
    return plugin.to_public()


def disconnect_remote_web_plugin(plugin_id: str) -> bool:
    value = str(plugin_id or "").strip().lower()
    if not value:
        return False

    records = _load_remote_records()
    next_records = [
        record
        for record in records
        if str(record.get("id", "")).strip().lower() != value
    ]
    if len(next_records) == len(records):
        return False

    _save_remote_records(next_records)
    return True


def normalize_remote_base(value: str | None) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = f"http://{raw}"

    parsed = urllib_parse.urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    if parsed.path and parsed.path not in {"/", ""}:
        base = urllib_parse.urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", "")
        )
    else:
        base = urllib_parse.urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
    return base.rstrip("/")


def _load_plugin_from_dir(plugin_dir: Path) -> WebPlugin | None:
    manifest_path = plugin_dir / "plugin.json"
    if not manifest_path.exists() or not manifest_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    plugin_id = str(manifest.get("id") or plugin_dir.name).strip().lower()
    if not PLUGIN_ID_RE.match(plugin_id):
        return None

    if bool(manifest.get("enabled", True)) is False:
        return None

    ui_dir = plugin_dir / "ui"
    if not ui_dir.exists() or not ui_dir.is_dir():
        return None

    entry = str(manifest.get("entry") or "index.html").strip().lstrip("/")
    if not entry:
        entry = "index.html"
    if not (ui_dir / entry).exists():
        return None

    backend_name = str(manifest.get("backend") or "backend.py").strip()
    backend_file = plugin_dir / backend_name if backend_name else None
    if backend_file is not None and not backend_file.exists():
        backend_file = None

    return WebPlugin(
        plugin_id=plugin_id,
        name=str(manifest.get("name") or plugin_id),
        description=str(manifest.get("description") or ""),
        order=int(manifest.get("order", 100)),
        plugin_dir=plugin_dir,
        ui_dir=ui_dir,
        entry=entry,
        icon=manifest.get("icon"),
        backend_file=backend_file,
    )


def _call_router_factory(factory: Any, plugin: WebPlugin) -> APIRouter | None:
    try:
        # Preferred signature: get_router(plugin_manifest: dict) -> APIRouter
        result = factory(plugin.to_public())
    except TypeError:
        try:
            # Fallback: get_router() -> APIRouter
            result = factory()
        except Exception:
            return None
    except Exception:
        return None
    return result if isinstance(result, APIRouter) else None


def _discover_remote_manifest(base: str) -> dict[str, Any]:
    for relative_path in REMOTE_MANIFEST_PATHS:
        url = _join_remote_url(base, relative_path)
        if url is None:
            continue
        payload = _http_json_get(url)
        if payload is None:
            continue
        normalized = _normalize_manifest_payload(payload)
        if normalized is not None:
            return normalized
    return {}


def _normalize_manifest_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("plugin"), dict):
        return dict(payload["plugin"])
    if isinstance(payload.get("plugins"), list):
        for item in payload["plugins"]:
            if isinstance(item, dict):
                return dict(item)
    return dict(payload)


def _http_json_get(url: str) -> dict[str, Any] | None:
    request = urllib_request.Request(
        url=url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib_request.urlopen(request, timeout=3.5) as response:
            content_type = str(response.headers.get("Content-Type", "")).lower()
            body = response.read().decode("utf-8", errors="replace")
            if "application/json" not in content_type and not body.strip().startswith("{"):
                return None
            parsed = json.loads(body)
            return parsed if isinstance(parsed, dict) else None
    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, ValueError):
        return None
    except Exception:
        return None


def _build_remote_plugin(base: str, manifest: dict[str, Any]) -> RemoteWebPlugin:
    parsed_base = urllib_parse.urlparse(base)
    fallback_slug = f"{parsed_base.netloc}{parsed_base.path}".strip()

    plugin_id = str(manifest.get("id") or "").strip().lower()
    if not PLUGIN_ID_RE.match(plugin_id):
        plugin_id = _slugify_plugin_id(fallback_slug or "remote_plugin")

    name = str(manifest.get("name") or parsed_base.netloc or plugin_id).strip() or plugin_id
    description = str(manifest.get("description") or "").strip() or f"Remote plugin at {base}"

    order = _coerce_int(manifest.get("order"), default=150)
    icon = str(manifest.get("icon") or "").strip() or None

    ui_url = _resolve_remote_url(base, manifest.get("ui_url"))
    if ui_url is None:
        entry = str(manifest.get("entry") or "").strip().lstrip("/")
        if entry:
            ui_url = _join_remote_url(base, entry)
    if ui_url is None:
        ui_url = base

    api_base = _resolve_remote_url(base, manifest.get("api_base"))
    if api_base is not None:
        api_base = api_base.rstrip("/")

    return RemoteWebPlugin(
        plugin_id=plugin_id,
        name=name,
        description=description,
        order=order,
        ui_url=ui_url,
        api_base=api_base,
        icon=icon,
        remote_base=base,
    )


def _slugify_plugin_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "_", str(value).lower()).strip("_")
    if not slug:
        slug = "remote_plugin"
    if not re.match(r"^[a-z0-9]", slug):
        slug = f"r_{slug}"
    slug = slug[:64]
    if len(slug) < 2:
        slug = f"{slug}0"
    if not PLUGIN_ID_RE.match(slug):
        slug = "remote_plugin"
    return slug


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _join_remote_url(base: str, path: str) -> str | None:
    joined = urllib_parse.urljoin(f"{base.rstrip('/')}/", str(path or "").lstrip("/"))
    parsed = urllib_parse.urlparse(joined)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return joined


def _normalize_absolute_url(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = urllib_parse.urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return raw


def _resolve_remote_url(base: str, value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    absolute = _normalize_absolute_url(raw)
    if absolute is not None:
        return absolute
    return _join_remote_url(base, raw)


def _remote_plugin_to_record(plugin: RemoteWebPlugin) -> dict[str, Any]:
    return {
        "id": plugin.plugin_id,
        "name": plugin.name,
        "description": plugin.description,
        "order": plugin.order,
        "ui_url": plugin.ui_url,
        "api_base": plugin.api_base,
        "icon": plugin.icon,
        "remote_base": plugin.remote_base,
    }


def _coerce_remote_plugin(raw: Any) -> RemoteWebPlugin | None:
    if not isinstance(raw, dict):
        return None

    plugin_id = str(raw.get("id") or "").strip().lower()
    if not PLUGIN_ID_RE.match(plugin_id):
        return None

    remote_base = normalize_remote_base(raw.get("remote_base"))
    if remote_base is None:
        return None

    ui_url = _normalize_absolute_url(raw.get("ui_url"))
    if ui_url is None:
        ui_url = remote_base

    api_base = _normalize_absolute_url(raw.get("api_base"))
    if api_base is not None:
        api_base = api_base.rstrip("/")

    name = str(raw.get("name") or plugin_id).strip() or plugin_id
    description = str(raw.get("description") or "").strip()
    order = _coerce_int(raw.get("order"), default=150)
    icon = str(raw.get("icon") or "").strip() or None

    return RemoteWebPlugin(
        plugin_id=plugin_id,
        name=name,
        description=description,
        order=order,
        ui_url=ui_url,
        api_base=api_base,
        icon=icon,
        remote_base=remote_base,
    )


def _load_remote_records() -> list[dict[str, Any]]:
    if not REMOTE_STATE_FILE.exists():
        return []
    try:
        raw = json.loads(REMOTE_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _save_remote_records(records: list[dict[str, Any]]) -> None:
    REMOTE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    REMOTE_STATE_FILE.write_text(
        json.dumps(records, indent=2, sort_keys=True),
        encoding="utf-8",
    )
