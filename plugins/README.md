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
