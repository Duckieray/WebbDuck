# WebbDuck Captioning Plugins

This guide explains how to set up and use image captioning plugins.

## Overview

Captioners are optional. When available, the `Caption` button appears in the Studio input-image area.

## Plugin Search Paths

WebbDuck checks these locations in order:

1. `WEBBDUCK_PLUGINS_DIR`
2. `webbduck/plugins/`
3. `~/.webbduck/plugins/`

On Windows this is usually `%USERPROFILE%\.webbduck\plugins\`.

## Plugin Structure

```text
captioners/
`- joycaption/
   `- captioner.py
```

## Required Interface

```python
def generate_caption(
    image_path: Path,
    prompt: str,
    max_tokens: int = 300,
) -> str:
    ...
```

## Installing JoyCaption

Create:

```text
~/.webbduck/plugins/captioners/joycaption/captioner.py
```

You can also use the bundled reference implementation at:

- `plugins/captioners/joycaption/captioner.py`

## Notes

- Captioning is GPU-intensive.
- WebbDuck offloads generation pipelines before captioning to free VRAM.

## Troubleshooting

- If `Caption` does not appear: verify plugin path and restart server.
- If out-of-memory occurs: close other GPU apps or reduce model load.

---

## Web-App Plugins

Web-app plugins can provide their own page inside WebbDuck without hardcoding
tool-specific logic in WebbDuck core.

### Structure

```text
webapps/
`- your_plugin/
   |- plugin.json
   |- backend.py      # optional
   `- ui/
      `- index.html
```

### plugin.json

```json
{
  "id": "your_plugin",
  "name": "Your Plugin",
  "description": "Optional short summary",
  "entry": "index.html",
  "order": 50,
  "backend": "backend.py"
}
```

### Runtime Routes

- Plugin list API: `GET /plugins/web`
- Plugin UI mount: `/plugins/web/<id>/ui/<entry>`
- Plugin backend mount: `/plugins/web/<id>/api/*` (if `backend.py` exports `get_router(...)`)
- Remote plugin list API: `GET /plugins/web/remote`
- Remote plugin connect API: `POST /plugins/web/remote/connect` with `{ "base_url": "127.0.0.1:8020" }`
- Remote plugin disconnect API: `DELETE /plugins/web/remote/<id>`

### Remote Plugin Connect (No Local Install)

You can connect plugins that are not installed in local plugin folders:

1. Open WebbDuck `Settings`.
2. In `Connect Remote Plugin`, enter `ip:port` or full URL (example `127.0.0.1:8020`).
3. Click `Connect Plugin`.
4. WebbDuck adds a new top tab for that plugin and persists it for future restarts.

Remote connection behavior:

- WebbDuck tries manifest discovery at:
  - `/webbduck-plugin.json`
  - `/.well-known/webbduck-plugin.json`
  - `/plugin.json`
- If no manifest is found, WebbDuck uses the entered URL as the plugin UI URL directly.

### DNADuck (Optional External Plugin)

DNADuck is not bundled in WebbDuck. Install it from the DNADuck repo:

```bash
cd /path/to/dnaduck
python3 tools/install_webbduck_plugin.py --webbduck-dir /path/to/webbduck --overwrite
```

Then restart WebbDuck. The DNADuck plugin supports:

1. `auto` (default): try managed API first, then fallback to local CLI.
2. `managed_api`: start/reuse DNADuck API from plugin backend.
3. `local_cli`: WebbDuck invokes DNADuck CLI directly.
4. `remote_api`: set API base (for example `http://127.0.0.1:8020`) and WebbDuck forwards plugin requests to that server.
