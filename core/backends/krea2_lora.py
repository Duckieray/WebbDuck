"""Ordinary Krea 2 LoRA support shared by the host, catalog, and worker.

WebbDuck keeps the public LoRA contract architecture-neutral: Studio submits a
registry name + weight, the host resolves that name to a local safetensors file,
and the isolated Krea worker applies the low-rank residual before any group
CPU-offload hooks are installed.

The Krea worker can preserve base checkpoint linears as FP8.  Those linears have
one compact residual slot for the identity-edit adapter; ordinary user LoRAs
must be able to coexist with it.  ``install_worker_lora_patch`` therefore makes
that slot additive: multiple LoRAs targeting the same projection are combined
into one mathematically-equivalent low-rank residual by concatenating ranks.
That keeps the existing FP8/native-FP8 execution paths unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


_KREA_NAMESPACE_NAMES = {"krea", "krea2", "krea-2"}


def _path_has_krea_namespace(path: Path) -> bool:
    """Treat an explicit ``lora/krea2`` style directory as authoritative."""
    lowered = {part.lower() for part in Path(path).parts}
    return bool(lowered & _KREA_NAMESPACE_NAMES)


def _safetensors_looks_krea2(path: Path) -> bool:
    """Best-effort Krea 2 identification without loading tensor payloads.

    Original Krea/Comfy and AI-Toolkit exports have distinctive ``blocks.*`` +
    ``attn.wq``/``txtfusion`` names.  Native Diffusers attention-only LoRAs are
    less distinguishable from other transformer families, so users can always
    make intent explicit by placing them under ``<LoRA root>/krea2/`` or setting
    ``arch: krea2`` in loras.json.
    """
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata() or {}
    except Exception:
        return False

    meta_text = " ".join(str(value) for value in metadata.values()).lower()
    if "krea2" in meta_text or "krea-2" in meta_text or "krea 2" in meta_text:
        return True

    joined = " ".join(str(key).lower() for key in keys)
    if "txtfusion" in joined or "text_fusion" in joined:
        return True
    if (
        ("diffusion_model.blocks." in joined or "base_model.model.blocks." in joined)
        and any(marker in joined for marker in (".attn.wq", ".attn.wk", ".attn.wv", ".attn.wo"))
    ):
        return True
    if "lora_unet_blocks_" in joined and any(
        marker in joined for marker in ("_attn_wq", "_attn_wk", "_attn_wv", "_attn_wo")
    ):
        return True
    if "time_mod_proj" in joined and ("img_in" in joined or "txt_in" in joined):
        return True
    return False


def is_krea2_lora_entry(entry: dict[str, Any] | None) -> bool:
    """Return whether a registry row belongs to Krea 2."""
    if not isinstance(entry, dict):
        return False
    arch = str(entry.get("arch") or "").strip().lower()
    if arch in {"krea", "krea2", "krea-2"}:
        return True
    path = Path(str(entry.get("path") or "")).expanduser()
    if not path.is_file():
        return False
    if _path_has_krea_namespace(path):
        return True
    return _safetensors_looks_krea2(path)


def resolve_krea2_loras(raw_loras: Any) -> list[dict[str, Any]]:
    """Resolve Studio LoRA selections to explicit Krea worker inputs."""
    if not isinstance(raw_loras, list) or not raw_loras:
        return []

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
            raise ValueError(f"Invalid Krea LoRA entry at index {index}: expected name or object")

        if not name:
            raise ValueError(f"Invalid Krea LoRA entry at index {index}: missing name")
        registry_entry = LORA_REGISTRY.get(name)
        if registry_entry is None:
            raise ValueError(f"Unknown LoRA: {name}")
        if not is_krea2_lora_entry(registry_entry):
            arch = str(registry_entry.get("arch") or "unknown")
            raise ValueError(f"LoRA '{name}' targets {arch}, not Krea 2.")

        path = Path(str(registry_entry.get("path") or "")).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Krea LoRA file not found for '{name}': {path}")

        weight_raw = requested_weight
        if weight_raw is None:
            weight_raw = registry_entry.get("weight", 1.0)
        try:
            weight = float(weight_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid LoRA weight for '{name}': {weight_raw!r}") from exc
        if not math.isfinite(weight):
            raise ValueError(f"Invalid LoRA weight for '{name}': {weight_raw!r}")

        resolved.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "weight": weight,
                "trigger": str(registry_entry.get("trigger") or "").strip() or None,
            }
        )
    return resolved


def lora_trigger_phrase(loras: list[dict[str, Any]]) -> str:
    """Return unique trigger words as plain text for Krea's Qwen encoder."""
    triggers: list[str] = []
    for entry in loras:
        trigger = str(entry.get("trigger") or "").strip()
        if trigger and trigger not in triggers:
            triggers.append(trigger)
    return ", ".join(triggers)


def inject_lora_trigger(prompt: str, trigger_phrase: str) -> str:
    prompt = str(prompt or "").strip()
    trigger_phrase = str(trigger_phrase or "").strip()
    if not trigger_phrase or trigger_phrase.lower() in prompt.lower():
        return prompt
    return f"{prompt}, {trigger_phrase}" if prompt else trigger_phrase


def _strip_known_prefixes(module: str) -> str:
    value = str(module)
    prefixes = (
        "base_model.model.",
        "model.diffusion_model.",
        "diffusion_model.",
        "transformer.",
        "_orig_mod.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if value.startswith(prefix):
                value = value[len(prefix):]
                changed = True
                break
    return value


def _unflatten_kohya_module(module: str) -> str:
    """Best-effort musubi/Kohya ``lora_unet_*`` module-name recovery."""
    value = str(module)
    if value.startswith("lora_unet_"):
        value = value[len("lora_unet_"):]
    value = value.replace("_", ".")
    # Reconstitute Krea/Diffusers identifiers that contain underscores.
    replacements = (
        ("transformer.blocks", "transformer_blocks"),
        ("text.fusion", "text_fusion"),
        ("time.mod.proj", "time_mod_proj"),
        ("time.embed", "time_embed"),
        ("final.layer", "final_layer"),
        ("img.in", "img_in"),
        ("txt.in", "txt_in"),
        ("linear.1", "linear_1"),
        ("linear.2", "linear_2"),
        ("to.out.0", "to_out.0"),
        ("to.q", "to_q"),
        ("to.k", "to_k"),
        ("to.v", "to_v"),
        ("to.gate", "to_gate"),
    )
    for source, target in replacements:
        value = value.replace(source, target)
    return value


def _map_krea_module(module: str) -> str:
    value = _strip_known_prefixes(module)
    if value.startswith("lora_unet_"):
        value = _unflatten_kohya_module(value)
    if value.startswith("blocks."):
        value = "transformer_blocks." + value[len("blocks."):]
    elif value.startswith("txtfusion."):
        value = "text_fusion." + value[len("txtfusion."):]

    replacements = (
        (".attn.wq", ".attn.to_q"),
        (".attn.wk", ".attn.to_k"),
        (".attn.wv", ".attn.to_v"),
        (".attn.wo", ".attn.to_out.0"),
        (".attn.gate", ".attn.to_gate"),
        (".mlp.gate", ".ff.gate"),
        (".mlp.up", ".ff.up"),
        (".mlp.down", ".ff.down"),
    )
    for source, target in replacements:
        value = value.replace(source, target)
    return value


def _split_lora_suffix(key: str) -> tuple[str | None, str | None]:
    """Split exporter-specific leaf syntax into module + WebbDuck suffix."""
    normalized = str(key).replace(".lora_A.default.weight", ".lora_A.weight")
    normalized = normalized.replace(".lora_B.default.weight", ".lora_B.weight")
    normalized = normalized.replace(".lora_down.default.weight", ".lora_down.weight")
    normalized = normalized.replace(".lora_up.default.weight", ".lora_up.weight")

    suffixes = (
        (".lora_A.weight", ".lora_A.weight"),
        (".lora_B.weight", ".lora_B.weight"),
        (".lora_down.weight", ".lora_A.weight"),
        (".lora_up.weight", ".lora_B.weight"),
        (".lora_alpha", ".lora_alpha"),
        (".alpha", ".lora_alpha"),
    )
    for source_suffix, target_suffix in suffixes:
        if normalized.endswith(source_suffix):
            return normalized[: -len(source_suffix)], target_suffix
    return None, None


def convert_krea2_lora_state(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize common Krea 2 LoRA exports to transformer-relative PEFT keys.

    Supported layouts include native Diffusers/PEFT, original Krea/Comfy
    ``diffusion_model.*``, Ostris AI-Toolkit ``base_model.model.*``, dotted
    Kohya suffixes, and best-effort flattened ``lora_unet_*`` (musubi/Kohya)
    module names. Unknown *LoRA weight* tensors fail loudly so WebbDuck never
    reports an adapter as loaded when it was silently ignored.
    """
    converted: dict[str, Any] = {}
    unresolved: list[str] = []

    for raw_key, tensor in state_dict.items():
        key = str(raw_key)
        lowered = key.lower()
        if "dora_scale" in lowered or "lora_magnitude_vector" in lowered:
            raise RuntimeError("Krea 2 DoRA adapters are not supported by the FP8 residual loader yet.")

        module, suffix = _split_lora_suffix(key)
        if module is None or suffix is None:
            # Ignore ordinary metadata/non-LoRA tensors. Anything that looks like
            # an adapter weight is an error rather than a partial silent load.
            if "lora" in lowered and (lowered.endswith("weight") or lowered.endswith(".alpha")):
                unresolved.append(key)
            continue

        module = _map_krea_module(module)
        if not module:
            unresolved.append(key)
            continue
        converted[module + suffix] = tensor

    if unresolved:
        preview = "\n  - ".join(unresolved[:20])
        extra = "" if len(unresolved) <= 20 else f"\n  ... and {len(unresolved) - 20} more"
        raise RuntimeError(
            f"Krea 2 LoRA contains {len(unresolved)} unsupported adapter tensors:\n  - {preview}{extra}"
        )
    if not any(key.endswith(".lora_A.weight") for key in converted):
        raise RuntimeError("Krea 2 LoRA contains no recognized low-rank A weights.")
    if not any(key.endswith(".lora_B.weight") for key in converted):
        raise RuntimeError("Krea 2 LoRA contains no recognized low-rank B weights.")
    return converted


def _install_stacking_scaled_fp8(base_module: Any) -> None:
    """Make ScaledFP8Linear's single residual slot additive across LoRAs."""
    cls = base_module.ScaledFP8Linear
    if bool(getattr(cls, "_webbduck_multi_lora", False)):
        return

    original_install = cls.install_lora

    def install_lora(self: Any, lora_a: Any, lora_b: Any, alpha: Any = None, lora_scale: float = 1.0) -> None:
        import torch

        if int(getattr(self, "lora_rank", 0) or 0) <= 0 or getattr(self, "lora_a", None) is None:
            original_install(self, lora_a, lora_b, alpha, lora_scale)
            return

        old_a = self.lora_a.detach().to(device="cpu")
        old_b = self.lora_b.detach().to(device="cpu")
        old_coef = base_module._linear_lora_coefficient(
            int(self.lora_rank), self.lora_alpha, float(getattr(self, "lora_scale", 1.0))
        )

        new_a = torch.as_tensor(lora_a).detach().to(device="cpu", dtype=old_a.dtype)
        new_b = torch.as_tensor(lora_b).detach().to(device="cpu", dtype=old_b.dtype)
        new_rank = int(new_a.shape[0])
        if (
            new_a.ndim != 2
            or new_b.ndim != 2
            or int(new_a.shape[1]) != int(self.in_features)
            or int(new_b.shape[0]) != int(self.out_features)
            or int(new_b.shape[1]) != new_rank
        ):
            raise RuntimeError(
                f"Krea LoRA shape mismatch for {self.in_features}x{self.out_features}: "
                f"A={tuple(new_a.shape)} B={tuple(new_b.shape)}."
            )
        new_coef = base_module._linear_lora_coefficient(new_rank, alpha, lora_scale)

        # sum_i c_i B_i A_i == [c1*B1 c2*B2 ...] @ [A1; A2; ...]
        combined_a = torch.cat((old_a, new_a), dim=0)
        combined_b = torch.cat((old_b * float(old_coef), new_b * float(new_coef)), dim=1)
        total_rank = int(combined_a.shape[0])
        combined_alpha = torch.tensor(float(total_rank), dtype=combined_a.dtype)
        original_install(self, combined_a, combined_b, combined_alpha, 1.0)

    cls.install_lora = install_lora
    cls._webbduck_multi_lora = True


def install_krea2_user_loras(
    base_module: Any,
    transformer: Any,
    loras: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load and apply ordinary user LoRAs to a Krea transformer."""
    if not loras:
        return []
    from safetensors.torch import load_file

    applied: list[dict[str, Any]] = []
    for entry in loras:
        path = Path(str(entry.get("path") or "")).expanduser()
        name = str(entry.get("name") or path.stem)
        weight = float(entry.get("weight", 1.0))
        if not path.is_file():
            raise FileNotFoundError(f"Krea LoRA file does not exist: {path}")

        raw = load_file(str(path), device="cpu")
        converted = convert_krea2_lora_state(raw)
        info = base_module.apply_identity_lora_residuals(
            transformer,
            converted_state=converted,
            lora_scale=weight,
        )
        applied_modules = int(info.get("applied_modules") or 0)
        if applied_modules <= 0:
            raise RuntimeError(f"Krea LoRA '{name}' did not match any transformer modules.")
        applied.append(
            {
                "name": name,
                "weight": weight,
                "path": str(path),
                "applied_modules": applied_modules,
                "converted_tensors": len(converted),
            }
        )
    return applied


def install_worker_lora_patch(base_module: Any) -> None:
    """Install ordinary LoRAs immediately after each Krea transformer load.

    This placement is deliberate: both normal execution and the safe OOM retry
    create transformers through ``_load_pipeline``. Installing here guarantees
    the same adapters survive retries and, crucially, registers FP8 residual
    buffers before diffusers group-offload snapshots module params/buffers.
    """
    if bool(getattr(base_module, "_webbduck_user_lora_patch", False)):
        return

    _install_stacking_scaled_fp8(base_module)
    original_load = base_module._load_pipeline

    def load_pipeline(request: dict[str, Any], dtype: Any, hardware: dict[str, Any], report: Any = None):
        pipe, load_info = original_load(request, dtype, hardware, report=report)
        loras = request.get("loras")
        if not isinstance(loras, list) or not loras:
            return pipe, load_info

        if callable(report):
            report("Installing Krea LoRAs", 0.35)
        try:
            applied = install_krea2_user_loras(base_module, pipe.transformer, loras)
        except Exception as exc:
            raise RuntimeError(f"Krea 2 user LoRA install failed: {exc}") from exc

        info = dict(load_info)
        info["user_loras"] = applied
        info["user_lora_count"] = len(applied)
        return pipe, info

    base_module._load_pipeline = load_pipeline
    base_module._webbduck_user_lora_patch = True
