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


def _load_lora_tensor_dict(path: Path) -> dict[str, Any]:
    """Load a safetensors LoRA file into a plain tensor dict.

    Prefer ``safetensors.torch.load_file`` when available so tensor dtypes are
    preserved; fall back to the registry's lazy ``safe_open`` otherwise.
    """
    try:
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    except Exception:
        pass

    from models.registry import safe_open

    state_dict: dict[str, Any] = {}
    # safe_open exposes a keys() API; drain it directly to avoid importing torch
    # tensors in this lightweight module.
    try:
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
        import torch

        for key in keys:
            state_dict[key] = torch.zeros((0,), dtype=torch.float32)
    except Exception:
        state_dict = {}
    return state_dict


def _uses_kohya_down_suffix(state_dict: dict[str, Any]) -> bool:
    return any(k.endswith(".lora_down.weight") for k in state_dict)


def _is_native_diffusers_paths(state_dict: dict[str, Any]) -> bool:
    """True when keys already carry diffusers module paths (``transformer.``).

    FLUX.2 uses the ``transformer`` component namespace. Real Kohya (sd-scripts)
    FLUX.2 exports flatten module paths into ``lora_unet_*`` keys, and PEFT
    exports carry ``base_model.model.`` / ``diffusion_model.`` prefixes. When a
    LoRA uses ``transformer.*`` with ``lora_down``/``lora_up`` suffixes, its
    module paths are already identical to the diffusers model so only the leaf
    suffix needs to become PEFT format.
    """
    return any(k.startswith("transformer.") for k in state_dict)


def _convert_flux2_diffusers_suffix_lora_to_peft(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Suffix-only FLUX.2 LoRA conversion (native diffusers paths + Kohya suffix).

    Some FLUX.2 trainers (e.g. OneTrainer export and direct diffusers-style
    exports) save LoRAs with the real diffusers module paths (``transformer.*``)
    but the Kohya leaf format (``.lora_down.weight`` / ``.lora_up.weight`` /
    ``.alpha``). Diffusers' own ``_convert_kohya_flux2_lora_to_diffusers`` only
    understands the flattened ``lora_unet_*`` convention, so it discards every
    key from this layout and registers an empty adapter, which later fails with
    "Adapter name(s) ... not in the list of present adapters".

    Because the module paths are already correct, we only need to rename the
    leaf suffix and fold the ``alpha`` scale into ``lora_A``, producing genuine
    PEFT keys — no path remapping is required.

    ``dora_scale`` tensors are dropped; diffusers does not support DoRA adapters.
    """
    import torch

    state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}
    converted: dict[str, Any] = {}
    for key, val in state_dict.items():
        if key.endswith(".lora_down.weight"):
            base = key[: -len(".lora_down.weight")]
            alpha_tensor = state_dict.get(base + ".alpha")
            alpha: float
            if alpha_tensor is not None and not isinstance(alpha_tensor, float):
                try:
                    alpha = float(alpha_tensor.item())
                except Exception:
                    alpha = float(val.shape[0])
            else:
                alpha = float(alpha_tensor) if alpha_tensor is not None else float(val.shape[0])
            rank = int(val.shape[0])
            scale = alpha / rank if rank else 1.0
            converted[base + ".lora_A.weight"] = val.to(torch.float32) * scale
        elif key.endswith(".lora_up.weight"):
            base = key[: -len(".lora_up.weight")]
            converted[base + ".lora_B.weight"] = val
    return converted


def _maybe_convert_diffusers_suffix_lora_to_peft(path: Path) -> dict[str, Any] | None:
    """Return a PEFT-converted state dict for native-suffix FLUX.2 LoRAs, else None.

    Loads the LoRA file from ``path`` and detects the layout where module paths
    are already native diffusers (``transformer.*``) but the leaf suffix is the
    Kohya ``lora_down``/``lora_up`` form. Only that layout needs the suffix-only
    conversion; all other formats (PEFT ``lora_A``, flat ``lora_unet_*``, etc.)
    are left for diffusers to handle as before, so this returns ``None``.
    """
    state_dict = _load_lora_tensor_dict(path)
    if not state_dict:
        return None
    if not _uses_kohya_down_suffix(state_dict):
        return None
    if not _is_native_diffusers_paths(state_dict):
        return None
    converted = _convert_flux2_diffusers_suffix_lora_to_peft(state_dict)
    return converted if converted else None


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

        # Diffusers 0.39/0.40 misdetect FLUX.2 LoRAs that use native diffusers
        # module paths (``transformer.*``) with the Kohya leaf suffix
        # (``lora_down``/``lora_up``/``alpha``) as flat Kohya and discard every
        # key, registering an empty adapter. Handle that layout here with a
        # suffix-only conversion (see _convert_flux2_diffusers_suffix_lora_to_peft).
        converted_peft = _maybe_convert_diffusers_suffix_lora_to_peft(path)
        if converted_peft is not None:
            print(
                f"[flux] LoRA {name} uses native diffusers paths; "
                f"applying suffix-only PEFT conversion ({len(converted_peft)} keys)",
                flush=True,
            )
            pipe.load_lora_weights(
                converted_peft,
                adapter_name=adapter_name,
                low_cpu_mem_usage=True,
            )
        else:
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
