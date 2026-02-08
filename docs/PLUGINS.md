# WebbDuck Captioning Plugins

This guide explains how to set up and use image captioning plugins.

## Overview

Captioners are optional. When available, the `Caption` button appears in the Studio input-image area.

## Plugin Search Paths

WebbDuck checks these locations in order:

1. `WEBBDUCK_PLUGINS_DIR`
2. `webbduck/plugins/`
3. `~/.webbduck/plugins/`

On Windows this is usually `C:\Users\<username>\.webbduck\plugins\`.

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
