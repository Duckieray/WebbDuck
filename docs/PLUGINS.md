# WebbDuck Plugins

WebbDuck supports two optional plugin types:

- captioner plugins in `plugins/captioners/` or another plugin search path
- web-app plugins in `plugins/webapps/` or another plugin search path

Core WebbDuck must continue working when plugins are missing, disabled, or failing.

## Plugin Search Order

WebbDuck checks plugin roots in this order:

1. `WEBBDUCK_PLUGINS_DIR`
2. `<repo>/plugins`
3. `~/.webbduck/plugins`

On Windows, the fallback user path is usually `%USERPROFILE%\.webbduck\plugins`.

## Captioner Plugins

Captioners add prompt-generation help for uploaded images.

### Folder Layout

```text
plugins/
`- captioners/
   `- joycaption/
      `- captioner.py
```

### Required Interface

Each captioner plugin must expose:

```python
def generate_caption(
    image_path: Path,
    prompt: str,
    max_tokens: int = 300,
) -> str:
    ...
```

### Runtime Notes

- Captioners are discovered by `core/captioning_config.py`.
- Loading and execution are handled by `core/captioner.py`.
- Captioning is GPU-heavy; WebbDuck unloads generation pipelines before captioning to free VRAM.
- If no captioner is installed, the rest of WebbDuck still works normally.

### Bundled Reference

The repo includes a reference captioner layout in `plugins/captioners/joycaption/`.

## Web-App Plugins

Web-app plugins render their own page inside WebbDuck and can optionally mount backend API routes.

### Folder Layout

```text
plugins/
`- webapps/
   `- my_plugin/
      |- plugin.json
      |- backend.py      # optional
      `- ui/
         `- index.html
```

> **Note:** The `plugins/webapps/` directory is gitignored and does not exist in the repo by default. It is created dynamically by external integrations (e.g., dnaduck, duckmotion).

### `plugin.json`

Required fields:

```json
{
  "id": "my_plugin",
  "name": "My Plugin",
  "entry": "index.html"
}
```

Optional fields:

- `description`
- `order`
- `backend` (defaults to `backend.py` if you use a backend file)

### Routes Exposed By WebbDuck

- `GET /plugins/web`: local bundled and discovered web-app plugins
- `GET /plugins/web/remote`: currently connected remote plugins
- `POST /plugins/web/remote/connect`: connect a remote plugin by URL
- `DELETE /plugins/web/remote/<id>`: disconnect a remote plugin
- `/plugins/web/<id>/ui/<entry>`: mounted plugin UI
- `/plugins/web/<id>/api/*`: mounted plugin backend routes when `backend.py` exports `get_router(...)`

### Backend Contract

If a plugin needs backend routes, `backend.py` should export `get_router(...) -> APIRouter`.

WebbDuck calls the router factory from `core/web_plugins.py` and mounts the result under `/plugins/web/<id>/api`.

## Remote Plugins

You can connect a plugin without installing it locally:

1. Open WebbDuck `Settings`.
2. Enter `ip:port` or a full URL in `Connect Remote Plugin`.
3. Click `Connect Plugin`.

Discovery tries these manifest URLs first:

- `/webbduck-plugin.json`
- `/.well-known/webbduck-plugin.json`
- `/plugin.json`

If no manifest is found, WebbDuck treats the entered URL as the UI entry directly.

## Updating Plugin Support In WebbDuck

### Captioner changes

- Discovery rules live in `core/captioning_config.py`.
- Runtime loading/unloading lives in `core/captioner.py`.
- API surfaces live in `server/app.py` (`/captioners`, `/caption_styles`, `/caption`).

### Web-app plugin changes

- Discovery and mounting live in `core/web_plugins.py`.
- HTTP endpoints live in `server/app.py`.
- UI wiring for tabs and remote plugin settings lives in `ui/app.js`.

When plugin contracts change, update this file and `plugins/README.md` together.

## Optional External Plugins

`plugins/webapps/` is gitignored and created dynamically by external integrations (dnaduck, duckmotion). `plugins/webapps/dnaduck/` and `plugins/webapps/duckmotion/` are integration points for separately managed plugin projects. WebbDuck should not require those repos to be present for core functionality.

Some external plugin repos may ship their own installer scripts. When they do, point them at the WebbDuck repo root. WebbDuck itself does not currently bundle a generic `tools/install_webbduck_plugin.py` helper.

## Troubleshooting

- Missing caption button: verify the captioner folder path and restart WebbDuck.
- Plugin tab missing: check `plugin.json` and plugin search path order.
- Backend plugin routes unavailable: make sure `backend.py` exports `get_router(...)` and returns an `APIRouter`.
- WebbDuck seems slower after plugin testing: confirm no plugin job is still holding the shared GPU lease.
