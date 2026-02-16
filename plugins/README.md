# Webbduck Plugins

This folder contains optional plugins that extend webbduck's functionality.

## Captioners

Image captioning plugins for generating prompts from uploaded images.

| Plugin | Description | VRAM |
|--------|-------------|------|
| [joycaption](captioners/joycaption/) | JoyCaption Alpha 2 - high quality image descriptions | ~22GB |

## Adding Your Own Captioner

Create a folder in `captioners/` with a `captioner.py` file that implements:

```python
def generate_caption(image_path: Path, prompt: str, max_tokens: int = 300) -> str:
    """Generate a caption for the image."""
    pass
```

See [docs/PLUGINS.md](../docs/PLUGINS.md) for full documentation.

## Web Apps

Web-app plugins can add a full page inside WebbDuck with minimal core coupling.

DNADuck is an external optional web-app plugin distributed from:

- `https://github.com/Duckieray/dnaduck`

Install from DNADuck repo:

```bash
python3 tools/install_webbduck_plugin.py --webbduck-dir /path/to/webbduck --overwrite
```

Or connect a remote plugin without local install:

1. Open WebbDuck `Settings`
2. Enter plugin `ip:port` in `Connect Remote Plugin`
3. Click `Connect Plugin`

Structure:

```text
webapps/
`- my_plugin/
   |- plugin.json
   |- backend.py      # optional FastAPI router factory
   `- ui/
      `- index.html
```

Required manifest (`plugin.json`) fields:

- `id`: stable plugin id (`[a-z0-9_-]`)
- `name`: tab label in WebbDuck UI
- `entry`: UI entry file inside `ui/` (usually `index.html`)

Optional:

- `description`
- `order` (lower appears earlier)
- `backend` (defaults to `backend.py`)

The UI is mounted at:

- `/plugins/web/<id>/ui/<entry>`

If `backend.py` exists and exports `get_router(...) -> APIRouter`, it is mounted at:

- `/plugins/web/<id>/api/*`
