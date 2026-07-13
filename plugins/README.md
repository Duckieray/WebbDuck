# WebbDuck Plugins Folder

This folder contains bundled optional plugin examples and integration points.

## Layout

```text
plugins/
|- captioners/
|  `- joycaption/
`- webapps/
   |- dnaduck/
   `- duckmotion/
```

## What Lives Here

- `captioners/`: image-captioning plugins discovered by WebbDuck.
- `webapps/`: web-app plugin manifests and UI/backend assets.

These plugins are optional. WebbDuck core should still run when they are absent or failing.

## Captioners

Captioner plugins expose a `generate_caption(...)` function from `captioner.py`.

Reference example:

- `plugins/captioners/joycaption/`

See `docs/PLUGINS.md` for the full plugin contract and search-path rules.

## Web-App Plugins

Web-app plugins can add a top-level page inside WebbDuck.

Expected structure:

```text
webapps/
`- my_plugin/
   |- plugin.json
   |- backend.py      # optional
   `- ui/
      `- index.html
```

`plugin.json` should at least define:

- `id`
- `name`
- `entry`

## External Plugin Repos

`dnaduck/` and `duckmotion/` are integration targets for separately managed plugin projects. Do not make WebbDuck core depend on those repos being installed locally.

If an external plugin repo provides its own installer, point it at the WebbDuck repo root. This repository does not currently bundle a generic plugin installer script.

## Related Docs

- `docs/PLUGINS.md` for plugin contracts and routing behavior
- `docs/ARCHITECTURE.md` for where plugin loading lives in WebbDuck core
