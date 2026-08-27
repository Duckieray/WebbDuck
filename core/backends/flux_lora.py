"""Shared FLUX LoRA normalization and worker-side adapter loading."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable


def resolve_flux_loras(raw_loras: Any) -> list[dict[str, Any]]:
    """Resolve the same name/weight LoRA contract used by SDXL to FLUX files.

    Studio submits LoRAs by registry name. Keep that public contract unchanged,
    but send explicit paths/weights to the isolated FLUX worker so it does not
    depend on the host registry implementation.
    """
    if not isinstance(raw_loras, list) or not raw_loras:
        return []

    # Lazy import keeps backend module import side effects small and matches the
    # existing server-owned registry refresh lifecycle.
    from models.registry import LORA_REGISTRY

    resolved: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_loras):
        if isinstance(raw, str):
            name = raw.strip()
            requested_weight = None
        elif isinstance(raw, dict):
            name = str(raw.get("name") or raw.get("model") or "").strip()
            requested_weight = raw.get("weight", raw.get("strength"))
        else:
            raise ValueError(f"Invalid LoRA entry at index {index}: expected name or object")

        if not name:
            raise ValueError(f"Invalid LoRA entry at index {index}: missing name")
        registry_entry = LORA_REGISTRY.get(name)
        if registry_entry is None:
            raise ValueError(f"Unknown LoRA: {name}")

        arch = str(registry_entry.get("arch") or "").strip().lower()
        if arch not in ("flux", "flux1", "flux2"):
            raise ValueError(
                f"LoRA '{name}' targets {arch or 'an unknown architecture'}, not FLUX."
            )

        path = Path(str(registry_entry.get("path") or "")).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"LoRA file not found for '{name}': {path}")

        weight_raw = requested_weight
        if weight_raw is None:
            weight_raw = registry_entry.get("weight", 1.0)
        try:
            weight = float(weight_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid LoRA weight for '{name}': {weight_raw!r}") from exc

        resolved.append(
            {
                "name": name,
                "path": str(path),
                "weight": weight,
                "trigger": str(registry_entry.get("trigger") or "").strip() or None,
            }
        )
    return resolved


def lora_trigger_phrase(loras: list[dict[str, Any]]) -> str:
    """Build natural-language trigger text for FLUX.2's Qwen prompt encoder.

    The Studio weight slider controls PEFT adapter strength separately. Unlike
    the mature SDXL conditioning path, FLUX.2 does not parse ``(token:weight)``
    syntax, so passing that notation would only feed literal punctuation to Qwen.
    """
    phrases: list[str] = []
    for entry in loras:
        trigger = str(entry.get("trigger") or "").strip()
        if trigger and trigger not in phrases:
            phrases.append(trigger)
    return ", ".join(phrases)


def inject_lora_trigger(prompt: str, trigger_phrase: str) -> str:
    prompt = str(prompt or "").strip()
    trigger_phrase = str(trigger_phrase or "").strip()
    if not trigger_phrase or trigger_phrase in prompt:
        return prompt
    return f"{prompt}, {trigger_phrase}" if prompt else trigger_phrase


def _adapter_name(name: str, index: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", str(name or "")).strip("_")
    return f"webbduck_flux_{index}_{clean or 'lora'}"


def apply_flux_loras(
    pipe: Any,
    loras: list[dict[str, Any]],
    *,
    progress: Callable[[str, float], None] | None = None,
) -> list[dict[str, Any]]:
    """Load active PEFT adapters without fusing them into quantized base weights."""
    if not loras:
        return []

    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    applied: list[dict[str, Any]] = []
    total = len(loras)

    for index, entry in enumerate(loras):
        path = Path(str(entry.get("path") or "")).expanduser()
        name = str(entry.get("name") or path.stem or f"lora_{index}")
        weight = float(entry.get("weight", 1.0))
        if not path.is_file():
            raise FileNotFoundError(f"FLUX LoRA file does not exist: {path}")

        adapter_name = _adapter_name(name, index)
        if progress is not None:
            progress(
                f"Loading FLUX.2 LoRA {index + 1}/{total}: {name}",
                0.49 + 0.03 * ((index + 1) / total),
            )
        print(f"[flux] loading LoRA {name} from {path} at weight={weight}", flush=True)

        # Diffusers 0.39 documents local LoRA loading as a directory plus
        # ``weight_name``. Using that explicit form also avoids a local file path
        # being interpreted as a Hub model id by loader validation.
        pipe.load_lora_weights(
            str(path.parent),
            weight_name=path.name,
            adapter_name=adapter_name,
            low_cpu_mem_usage=True,
        )
        adapter_names.append(adapter_name)
        adapter_weights.append(weight)
        applied.append(
            {
                "name": name,
                "weight": weight,
                "path": str(path),
                "adapter_name": adapter_name,
            }
        )

    # Keep adapters live instead of fusing them. Fusing is intentionally avoided
    # for GGUF because it would require materializing/dequantizing base weights and
    # defeats the representation/memory advantages of the selected checkpoint.
    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
    if progress is not None:
        progress("FLUX.2 LoRAs ready", 0.53)
    return applied
