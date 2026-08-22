"""Standalone worker for WebbDuck's mature SDXL stack.

The WebbDuck host process intentionally does not import the SDXL Diffusers
pipeline implementation. This worker is launched with WEBBDUCK_SDXL_PYTHON;
all model construction, adapters, LoRAs and generation stay inside this process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Retained only for the lightweight core.pipeline facade used by a few legacy
# utilities. Backend-owned SDXL modules now import each other directly.
os.environ["WEBBDUCK_SDXL_WORKER"] = "1"


_METADATA_KEYS = (
    "prompt",
    "prompt_2",
    "lora_trigger_phrase",
    "identity_adapter_debug",
    "performance_timing",
    "smart_extend_source_box",
    "width",
    "height",
    "seed",
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _install_progress_bridge(progress_path: Path | None) -> None:
    if progress_path is None:
        return

    import server.state as state_module

    current: dict[str, Any] = {
        "stage": "Preparing",
        "progress": 0.0,
        "step": 0,
        "total_steps": 0,
    }

    def publish() -> None:
        try:
            _atomic_write_json(progress_path, current)
        except Exception:
            pass

    def update_stage(stage: str) -> None:
        current["stage"] = str(stage)
        publish()

    def update_progress(p: float, step: int = 0, total_steps: int = 0) -> None:
        current["progress"] = float(p)
        current["step"] = int(step or 0)
        current["total_steps"] = int(total_steps or 0)
        publish()

    state_module.update_stage = update_stage
    state_module.update_progress = update_progress
    publish()


def _metadata_updates(settings: dict[str, Any], seed: int) -> dict[str, Any]:
    payload: dict[str, Any] = {"seed": int(seed)}
    for key in _METADATA_KEYS:
        if key not in settings:
            continue
        value = settings[key]
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        payload[key] = value
    return payload


def _generate(request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    from core.backends.sdxl import _hydrate_execution_registry
    from core.backends.sdxl_runtime import run_generation

    settings = dict(request.get("settings") or {})
    if not settings.get("base_model"):
        raise ValueError("SDXL worker request is missing base_model")

    _hydrate_execution_registry()
    images, seed = run_generation(settings, cancel_event=None)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for index, image in enumerate(images):
        path = output_dir / f"sdxl_{index:03d}.png"
        image.save(path)
        saved.append(str(path))

    if not saved:
        raise RuntimeError("SDXL runtime returned no images")

    return {
        "ok": True,
        "images": saved,
        "seed": int(seed),
        "metadata_updates": _metadata_updates(settings, int(seed)),
    }


def _tokenize(request: dict[str, Any]) -> dict[str, Any]:
    from core.backends.sdxl import _hydrate_execution_registry
    from core.backends.sdxl_pipeline import get_tokenizer
    from models.registry import MODEL_REGISTRY

    model_name = str(request.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("SDXL tokenizer request is missing model_name")

    _hydrate_execution_registry()
    entry = MODEL_REGISTRY.get(model_name)
    if not entry:
        raise ValueError(f"Unknown SDXL checkpoint: {model_name}")

    tokenizer, tokenizer_2 = get_tokenizer(Path(entry["path"]))
    which = str(request.get("which") or "prompt").strip().lower()
    active = tokenizer_2 if which == "prompt_2" else tokenizer
    token_ids = active(
        str(request.get("text") or ""),
        truncation=False,
        add_special_tokens=True,
    )["input_ids"]
    return {
        "ok": True,
        "tokens": len(token_ids),
        "limit": int(getattr(active, "model_max_length", 77) or 77),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--progress")
    args = parser.parse_args()

    result_path = Path(args.result)
    progress_path = Path(args.progress) if args.progress else None
    _install_progress_bridge(progress_path)

    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        action = str(request.get("action") or "generate").strip().lower()
        if action == "generate":
            if not args.output_dir:
                raise ValueError("--output-dir is required for SDXL generation")
            result = _generate(request, Path(args.output_dir))
        elif action == "tokenize":
            result = _tokenize(request)
        else:
            raise ValueError(f"Unknown SDXL worker action: {action}")
    except Exception as exc:
        result = {
            "ok": False,
            "error": str(exc or exc.__class__.__name__),
            "traceback": traceback.format_exc(),
        }

    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
