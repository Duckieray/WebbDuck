"""Standalone MOSS-Image Diffusers worker.

MOSS-Image is distributed as a small remote Diffusers pipeline
(``pipeline_moss_image.py``) plus a Qwen3.5 text encoder and a FLUX.2 VAE.
Community/ComfyUI releases usually ship only the 5B DiT as one safetensors
file, so this worker:

1. loads the pipeline module from the component repository,
2. builds the DiT on the meta device from the published transformer config,
3. streams the local single-file DiT into that scaffold, and
4. runs text-to-image (or reference-conditioned editing) with the component
   text encoder + VAE.

Keeping the component source separate from the checkpoint is what lets any
MOSS-Image-based DiT (base, SFT, merges, LoRA-fused community mixes) run without
hardcoding a single model name.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_COMPONENT_MODEL = "OpenMOSS-Team/MOSS-Image-Base"
PIPELINE_FILE = "pipeline_moss_image.py"
TRANSFORMER_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")

# Tensors that are derived at runtime instead of stored in checkpoints.
DERIVED_TRANSFORMER_KEYS = {"rope.bands"}

# ComfyUI quantization sidecars are not DiT weights. MOSS-Image checkpoints in
# these formats are detected but not runnable yet, so the overlay fails loudly.
_IGNORED_SOURCE_SUFFIXES = (
    ".comfy_quant",
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".pre_quant_scale",
)
_QUANTIZED_DTYPE_TOKENS = ("FLOAT8", "F8_E", "INT8", "UINT8", "I8", "U8")

# Some ComfyUI exports bundle support components in the same file. Only the DiT
# belongs to the transformer scaffold, so these namespaces are skipped rather
# than treated as unknown transformer keys.
_FOREIGN_KEY_PREFIXES = (
    "text_encoders.",
    "text_encoder.",
    "cond_stage_model.",
    "first_stage_model.",
    "vae.",
    "clip_l.",
    "clip_g.",
    "t5xxl.",
    "scheduler.",
)


def _is_quantized_dtype(dtype_name: str) -> bool:
    return any(token in dtype_name for token in _QUANTIZED_DTYPE_TOKENS)


def _snap(value: int, multiple: int = 16) -> int:
    return max(multiple, int(round(float(value) / multiple) * multiple))


def _write_progress(
    progress_path: Path | None,
    stage: str,
    progress: float,
    step: int = 0,
    total_steps: int = 0,
) -> None:
    if progress_path is None:
        return
    try:
        payload = {
            "stage": str(stage),
            "progress": max(0.0, min(1.0, float(progress))),
            "step": int(step),
            "total_steps": int(total_steps),
        }
        temp = progress_path.with_name(f".{progress_path.name}.tmp")
        temp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temp, progress_path)
    except Exception:
        pass


def strip_moss_transformer_prefix(key: str) -> str:
    """Drop the ComfyUI/Diffusers wrappers around MOSS transformer tensors."""
    value = str(key)
    for prefix in TRANSFORMER_PREFIXES:
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def map_moss_transformer_key(key: str) -> list[str]:
    """Map one checkpoint tensor name onto Diffusers transformer names.

    ComfyUI exports fuse the image-stream attention projections into a single
    ``proj.qkv`` matrix, while the reference Diffusers module keeps separate
    ``proj.q``/``proj.k``/``proj.v`` linears. Text-embedder blocks already use a
    fused ``qkv`` linear in both layouts, so only ``.proj.qkv.weight`` is split.
    """
    normalized = strip_moss_transformer_prefix(key)
    fused = re.fullmatch(r"(?P<base>.+)\.proj\.qkv\.weight", normalized)
    if not fused:
        return [normalized]
    base = fused.group("base")
    return [
        f"{base}.proj.q.weight",
        f"{base}.proj.k.weight",
        f"{base}.proj.v.weight",
    ]


def _load_pipeline_module(component_source: str) -> Any:
    """Import ``pipeline_moss_image.py`` from a local directory or the Hub.

    The component repository exposes the same file at its root and inside the
    component folders for Diffusers custom-component discovery, so accept any
    of those layouts.
    """
    candidates: list[Path] = []
    root = Path(component_source).expanduser()
    if root.is_dir():
        candidates = [
            root / PIPELINE_FILE,
            root / "transformer" / PIPELINE_FILE,
            root / "scheduler" / PIPELINE_FILE,
        ]
    else:
        try:
            from huggingface_hub import hf_hub_download

            token = str(os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
            candidates = [
                Path(
                    hf_hub_download(
                        repo_id=component_source,
                        filename=PIPELINE_FILE,
                        token=token or None,
                    )
                )
            ]
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {PIPELINE_FILE} from component source '{component_source}'. "
                "Download the MOSS-Image component repository (the single-file checkpoint "
                "only contains the DiT)."
            ) from exc

    pipeline_path = next((path for path in candidates if path.is_file()), None)
    if pipeline_path is None:
        raise RuntimeError(
            f"Component source '{component_source}' does not contain {PIPELINE_FILE}. "
            "Point WEBBDUCK_MOSS_IMAGE_COMPONENT_MODEL at a MOSS-Image Diffusers "
            "repository or local snapshot."
        )

    spec = importlib.util.spec_from_file_location("webbduck_moss_image_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import MOSS-Image pipeline from {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_COMPONENT_SNAPSHOT_MARKERS = (
    PIPELINE_FILE,
    "transformer/config.json",
    "tokenizer/tokenizer.json",
    "vae/config.json",
)


def _hf_cache_roots() -> list[Path]:
    """Return candidate Hugging Face hub caches, most specific first."""
    roots: list[Path] = []
    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = str(os.getenv(env_name) or "").strip()
        if value:
            roots.append(Path(value).expanduser())
    hf_home = str(os.getenv("HF_HOME") or "").strip()
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _snapshot_is_complete(snapshot: Path) -> bool:
    """Check that a cached component snapshot has every file the worker needs."""
    if not snapshot.is_dir():
        return False
    if any(not (snapshot / marker).is_file() for marker in _COMPONENT_SNAPSHOT_MARKERS):
        return False

    text_encoder = snapshot / "text_encoder"
    index_path = text_encoder / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        except Exception:
            return False
        shards = {str(name) for name in weight_map.values()}
        if not shards or any(not (text_encoder / shard).is_file() for shard in shards):
            return False
    elif not any(text_encoder.glob("*.safetensors")):
        return False

    if not any((snapshot / "vae").glob("*.safetensors")):
        return False
    return True


def _resolve_component_source(component_source: str) -> str:
    """Prefer a complete local snapshot over re-downloading component weights.

    The configured Hugging Face cache can differ from the one that already holds
    the MOSS-Image components (for example, a large external drive cache that is
    empty on first run while the default ``~/.cache/huggingface/hub`` is
    populated). Downloading the ~16 GB Qwen3.5 text encoder again is slow and
    looks like a stalled job, so scan known caches for a usable snapshot first.
    """
    root = Path(component_source).expanduser()
    if root.is_dir():
        return str(root)
    if component_source.startswith(("/", "~", ".")) or "/" not in component_source:
        return component_source

    repo_dir = "models--" + component_source.replace("/", "--")
    for cache_root in _hf_cache_roots():
        repo_root = cache_root / repo_dir
        snapshots = repo_root / "snapshots"
        if not snapshots.is_dir():
            continue

        candidates: list[Path] = []
        ref_path = repo_root / "refs" / "main"
        if ref_path.is_file():
            ref = ref_path.read_text(encoding="utf-8").strip()
            if ref:
                candidates.append(snapshots / ref)
        candidates.extend(
            sorted(
                (path for path in snapshots.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
        for candidate in candidates:
            if _snapshot_is_complete(candidate):
                return str(candidate)
    return component_source


def _local_files_only() -> bool:
    return str(os.getenv("WEBBDUCK_MOSS_IMAGE_LOCAL_ONLY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _build_empty_transformer(module: Any, component_source: str) -> Any:
    """Instantiate the MOSS DiT without allocating its 5B weights."""
    from accelerate import init_empty_weights

    config = module.MOSSImageTransformer.load_config(component_source, subfolder="transformer")
    with init_empty_weights():
        transformer = module.MOSSImageTransformer.from_config(config)
    return transformer


def overlay_single_file_transformer(
    transformer: Any,
    checkpoint_path: Path,
    *,
    target_dtype: Any | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Stream a MOSS-Image single-file DiT into a meta-device scaffold."""
    import torch
    from safetensors import safe_open

    checkpoint_path = Path(checkpoint_path)
    expected = set(transformer.state_dict())
    mapped: set[str] = set()
    ignored: list[str] = []

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        source_keys = [key for key in sorted(keys) if not key.endswith(_IGNORED_SOURCE_SUFFIXES)]
        total = max(1, len(source_keys))
        report_every = max(1, total // 50)

        with torch.no_grad():
            for index, source_key in enumerate(source_keys, start=1):
                if source_key.startswith(_FOREIGN_KEY_PREFIXES):
                    ignored.append(source_key)
                    continue

                targets = map_moss_transformer_key(source_key)
                if len(targets) == 1 and targets[0] in DERIVED_TRANSFORMER_KEYS:
                    ignored.append(source_key)
                    continue

                source = handle.get_tensor(source_key)
                dtype_name = str(source.dtype).replace("torch.", "").upper()
                if _is_quantized_dtype(dtype_name):
                    raise RuntimeError(
                        f"MOSS-Image checkpoint tensor {source_key!r} is {dtype_name}; "
                        "quantized MOSS checkpoints are not supported yet. Use a bf16/fp16 "
                        "single-file checkpoint."
                    )
                value = source.to(target_dtype) if target_dtype is not None else source
                del source

                if len(targets) == 3:
                    if any(target not in expected for target in targets):
                        raise RuntimeError(
                            f"MOSS-Image checkpoint tensor {source_key!r} maps to unknown "
                            f"Diffusers keys {targets!r}."
                        )
                    for offset, target in enumerate(targets):
                        chunk = value.chunk(3, dim=0)[offset].contiguous()
                        transformer.load_state_dict({target: chunk}, strict=False, assign=True)
                        mapped.add(target)
                    del value
                else:
                    target = targets[0]
                    if target not in expected:
                        raise RuntimeError(
                            f"MOSS-Image checkpoint tensor {source_key!r} maps to unknown "
                            f"Diffusers key {target!r}."
                        )
                    transformer.load_state_dict({target: value}, strict=False, assign=True)
                    mapped.add(target)
                    del value

                if progress_callback is not None and (index == total or index % report_every == 0):
                    try:
                        progress_callback(index, total)
                    except Exception:
                        pass

    missing = sorted(expected - mapped - DERIVED_TRANSFORMER_KEYS)
    if missing:
        preview = ", ".join(missing[:20])
        raise RuntimeError(
            f"MOSS-Image single-file overlay was incomplete: {len(missing)} Diffusers "
            f"transformer tensors were not found (for example: {preview})."
        )

    # ``rope.bands`` is a derived frequency table. Meta-device construction
    # leaves it unpopulated, so rebuild it in fp32 exactly like the reference
    # pipeline does after loading a checkpoint.
    rope = getattr(transformer, "rope", None)
    if rope is not None and hasattr(rope, "_default_bands"):
        import torch

        bands = rope.bands
        if bands is None or bool(getattr(bands, "is_meta", False)):
            rope.bands = rope._default_bands(torch.device("cpu"))
        else:
            rope.bands = rope._default_bands(bands.device)

    return {
        "mapped_tensors": len(mapped),
        "ignored_tensors": ignored[:40],
        "missing_tensors": [],
    }


_LORA_LEAF_SUFFIXES = (
    ".lora_A.weight",
    ".lora_B.weight",
    ".lora_down.weight",
    ".lora_up.weight",
    ".alpha",
    ".dora_scale",
)
_LORA_WRAPPER_PREFIXES = ("base_model.model.",)


def _split_lora_leaf(key: str) -> tuple[str | None, str | None]:
    for suffix in _LORA_LEAF_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix
    return None, None


def _normalize_lora_module(module: str) -> str:
    """Map a LoRA module path onto the scaffold's transformer module names."""
    value = str(module)
    changed = True
    while changed:
        changed = False
        for prefix in _LORA_WRAPPER_PREFIXES:
            if value.startswith(prefix):
                value = value[len(prefix) :]
                changed = True
    return strip_moss_transformer_prefix(value)


def _lora_delta(pair: dict[str, Any], weight: float) -> Any:
    """Build ``B @ A * (alpha / rank) * weight`` for one adapter module."""
    import torch

    down = pair.get(".lora_A.weight")
    if down is None:
        down = pair.get(".lora_down.weight")
    up = pair.get(".lora_B.weight")
    if up is None:
        up = pair.get(".lora_up.weight")
    if down is None or up is None:
        return None

    rank = int(down.shape[0]) if down.ndim else 1
    alpha = pair.get(".alpha")
    scale = 1.0
    if alpha is not None and rank:
        try:
            scale = float(alpha.item()) / float(rank)
        except (AttributeError, TypeError, ValueError):
            scale = 1.0
    return (up.to(torch.float32) @ down.to(torch.float32)) * (float(weight) * scale)


def apply_moss_loras(
    transformer: Any,
    loras: list[dict[str, Any]],
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Merge LoRA deltas into the MOSS-Image DiT before it moves to the GPU.

    ai-toolkit MOSS adapters use ``diffusion_model.layers.N.proj.*`` module paths
    with PEFT ``lora_A``/``lora_B`` leaves, which map directly onto the scaffold
    this worker builds. Kohya ``lora_down``/``lora_up`` leaves and fused
    ``proj.qkv`` adapters are normalized here as well. Fusing is intentional:
    the worker process exits after one job, so there is nothing to unload, and
    merging avoids a PEFT dependency plus per-forward adapter dispatch.
    """
    import torch
    from safetensors import safe_open

    modules = dict(transformer.named_modules())
    applied: list[dict[str, Any]] = []
    total = len(loras)

    for index, entry in enumerate(loras):
        path = Path(str(entry.get("path") or "")).expanduser()
        name = str(entry.get("name") or path.stem or f"lora_{index}")
        weight = float(entry.get("weight", 1.0))
        if not path.is_file():
            raise RuntimeError(f"MOSS-Image LoRA file does not exist: {path}")
        if progress_callback is not None:
            try:
                progress_callback(index + 1, total, name)
            except Exception:
                pass

        pairs: dict[str, dict[str, Any]] = {}
        ignored: list[str] = []
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                module, suffix = _split_lora_leaf(key)
                if module is None or suffix == ".dora_scale":
                    ignored.append(key)
                    continue
                pairs.setdefault(_normalize_lora_module(module), {})[suffix] = (
                    handle.get_tensor(key)
                )

        touched = 0
        unmatched: list[str] = []
        fused = 0
        with torch.no_grad():
            for module_name, pair in pairs.items():
                if module_name.endswith(".proj.qkv"):
                    base = module_name[: -len(".proj.qkv")]
                    targets = [f"{base}.proj.{leaf}" for leaf in ("q", "k", "v")]
                    fused += 1
                else:
                    targets = [module_name]

                delta = _lora_delta(pair, weight)
                if delta is None:
                    unmatched.append(module_name)
                    continue
                if len(targets) > 1 and int(delta.shape[0]) % len(targets) != 0:
                    raise RuntimeError(
                        f"MOSS-Image LoRA {name!r} has a fused adapter for {module_name!r} "
                        f"with {int(delta.shape[0])} output rows, which cannot be split "
                        f"into {len(targets)} projections."
                    )

                chunks = delta.chunk(len(targets), dim=0) if len(targets) > 1 else [delta]
                for target, chunk in zip(targets, chunks):
                    module = modules.get(target)
                    parameter = getattr(module, "weight", None)
                    if parameter is None or not isinstance(parameter, torch.Tensor):
                        unmatched.append(target)
                        continue
                    if bool(getattr(parameter, "is_meta", False)):
                        raise RuntimeError(
                            f"MOSS-Image LoRA {name!r} targets {target!r} before the "
                            "checkpoint overlay populated it."
                        )
                    parameter.add_(chunk.to(parameter.dtype))
                    touched += 1
                del delta

        if touched == 0:
            preview = ", ".join(sorted(ignored)[:5]) or ", ".join(sorted(unmatched)[:5])
            raise RuntimeError(
                f"MOSS-Image LoRA {name!r} does not match this checkpoint's DiT layout "
                f"({len(pairs)} adapter modules, {len(ignored)} ignored keys). "
                f"Expected keys like 'diffusion_model.layers.0.proj.q.lora_A.weight'. "
                f"Sample keys: {preview or 'none'}"
            )

        applied.append(
            {
                "name": name,
                "weight": weight,
                "modules": touched,
                "fused_qkv_modules": fused,
                "unmatched_modules": len(unmatched),
            }
        )
        print(
            f"[moss] applied LoRA {name} at weight={weight} to {touched} modules "
            f"({fused} fused qkv, {len(unmatched)} unmatched)",
            flush=True,
        )

    return applied


def _configure_memory(pipe: Any, device: str, total_vram_gb: float) -> str:
    """Choose where the two large models live.

    The 5B DiT and the 9B text encoder cannot co-reside on consumer cards. The
    default for mid-size cards (roughly 14-22 GB) is ``split``: encode the prompt
    with the text encoder streamed through the GPU, drop it, then keep the DiT
    resident for denoising. That avoids per-block transfers on every sampling
    step, which makes sequential offload an order of magnitude slower.

    Cards below ~14 GB keep the fully sequential fallback because the resident
    DiT alone would not fit.
    """
    if device != "cuda":
        pipe.to("cpu")
        return "cpu"

    mode = str(os.getenv("WEBBDUCK_MOSS_IMAGE_OFFLOAD", "auto")).strip().lower()
    if mode == "auto":
        if total_vram_gb >= 22.0:
            mode = "model"
        elif total_vram_gb >= 14.0:
            mode = "split"
        else:
            mode = "sequential"

    if mode in {"split", "swap", "encode"}:
        # Placement is deferred until the prompt embeddings exist.
        return "split"
    if mode in {"sequential", "seq"}:
        pipe.enable_sequential_cpu_offload(device="cuda")
        return "sequential"
    if mode in {"model", "cpu", "1", "true", "yes"}:
        pipe.enable_model_cpu_offload(device="cuda")
        return "model"

    pipe.to("cuda")
    return "none"


def _encode_prompt_texts(
    pipe: Any,
    prompts: list[str],
    max_sequence_length: int,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Tokenize and encode like ``MOSSImagePipeline.encode_prompt``.

    The pipeline's own helper sends inputs to ``self._execution_device``. With
    only the text encoder hooked for streaming that property resolves to
    ``meta`` (the encoder weights are meta between forwards), so keep the
    inputs on CPU and let Accelerate's hooks move each layer instead.
    """
    import torch

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    if tokenizer is None or text_encoder is None:
        raise RuntimeError("MOSS-Image text encoder/tokenizer are required for prompts")

    rows = []
    with torch.no_grad():
        for value in prompts:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": value}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=max_sequence_length,
            )
            if device is not None:
                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                    if hasattr(value, "to")
                }
            output = text_encoder(**inputs, use_cache=False, return_dict=True)
            rows.append(output.last_hidden_state[0])

    length = max(row.shape[0] for row in rows)
    embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, device=embeds.device, dtype=torch.bool)
    for index, row in enumerate(rows):
        embeds[index, : row.shape[0]] = row
        mask[index, : row.shape[0]] = True
    return embeds, mask


def _encode_prompt_embeds(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    max_sequence_length: int,
    progress_path: Path | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Encode prompts with the text encoder streamed through the GPU, then free it."""
    import gc

    import torch
    from accelerate import cpu_offload
    from accelerate.hooks import remove_hook_from_module

    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        raise RuntimeError("MOSS-Image text encoder is required for text prompts")

    _write_progress(progress_path, "Encoding MOSS-Image prompt", 0.46)
    cpu_offload(text_encoder, execution_device=torch.device("cuda"))
    try:
        positive, positive_mask = _encode_prompt_texts(
            pipe, [prompt], max_sequence_length
        )
        negative = negative_mask = None
        if float(guidance_scale) != 1.0:
            negative, negative_mask = _encode_prompt_texts(
                pipe, [negative_prompt], max_sequence_length
            )
    finally:
        remove_hook_from_module(text_encoder, recurse=True)

    # Drop the ~17 GB text encoder so the DiT can stay resident on the GPU.
    pipe.text_encoder = None
    pipe.tokenizer = None
    del text_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return positive, positive_mask, negative, negative_mask


def _encode_prompt_embeds_with_refs(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    images: list[Any],
    guidance_scale: float,
    max_sequence_length: int,
    progress_path: Path | None = None,
    *,
    stream_encoder: bool,
) -> tuple[Any, Any, Any, Any]:
    """Encode a ``Picture N:`` prompt plus a plain negative prompt.

    ``stream_encoder`` matches the ``split`` placement: the 9B text encoder is
    streamed through the GPU and dropped afterwards. Other placements already
    own the encoder's device (models on CUDA or Accelerate hooks), so the
    encoder stays loaded there.
    """
    import gc

    import torch
    from accelerate import cpu_offload
    from accelerate.hooks import remove_hook_from_module

    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        raise RuntimeError("MOSS-Image text encoder is required for references")

    if stream_encoder:
        _write_progress(
            progress_path, "Encoding MOSS-Image prompt with references", 0.46
        )
        cpu_offload(text_encoder, execution_device=torch.device("cuda"))

    try:
        device = _text_input_device(text_encoder)
        positive = _encode_prompt_with_refs(
            pipe, prompt, images, max_sequence_length
        )[None].to(device)
        positive_mask = torch.ones(
            positive.shape[:2], dtype=torch.bool, device=device
        )
        negative = negative_mask = None
        if float(guidance_scale) != 1.0:
            negative, negative_mask = _encode_prompt_texts(
                pipe, [negative_prompt], max_sequence_length, device=device
            )
    finally:
        if stream_encoder:
            remove_hook_from_module(text_encoder, recurse=True)

    if stream_encoder:
        # Drop the ~17 GB text encoder so the DiT can stay resident on the GPU.
        pipe.text_encoder = None
        pipe.tokenizer = None
        del text_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return positive, positive_mask, negative, negative_mask


def _move_split_models_to_gpu(pipe: Any, progress_path: Path | None = None) -> None:
    """Place the DiT and VAE on the GPU after prompt encoding is complete."""
    import gc

    import torch

    _write_progress(progress_path, "Moving MOSS-Image models to GPU", 0.49)
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Reference-image (identity) conditioning
# ---------------------------------------------------------------------------
#
# MOSS-Image ships as a text-to-image model. Reference images follow the
# ai-toolkit ``edit`` training layout used by the community identity adapters
# (MOSS-Image-CharRef, MOSS-Image-Edit-HQ):
#
# 1. each reference is downscaled (never upscaled) to <=384^2 pixels and read by
#    the Qwen3.5 vision tower through ``Picture N:`` placeholders in the prompt;
# 2. the reference is also VAE-encoded, 2x2 packed and batch-norm normalized
#    into the DiT latent space, then appended to the token stream after the
#    text with t=0 modulation on its own RoPE index (1, 2, ...);
# 3. with ``isolate_refs`` reference queries only attend to reference keys, so
#    their per-block K/V are constant across sampling steps and are computed
#    once. Adapters trained with ai-toolkit's ``kv_cache: true`` expect this.

REFERENCE_VLM_PIXELS = 384 * 384
REFERENCE_LATENT_PIXELS = 640 * 640
# 640^2 reference latents fit beside the resident DiT on a 16 GB card; below
# this VRAM size fall back to 512^2 instead of running out of memory.
REFERENCE_LATENT_PIXELS_LOW_VRAM = 512 * 512
LOW_VRAM_GB = 15.0


def _effective_reference_pixels(requested: int, total_vram_gb: float) -> int:
    if 0.0 < total_vram_gb < LOW_VRAM_GB:
        return min(requested, REFERENCE_LATENT_PIXELS_LOW_VRAM)
    return requested
MAX_REFERENCE_IMAGES = 4


def _reference_paths(request: dict) -> list[Path]:
    raw = request.get("reference_images") or []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("MOSS-Image reference_images must be a list of paths")
    paths: list[Path] = []
    for index, value in enumerate(raw):
        text = str(value or "").strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"MOSS-Image reference image {index + 1} does not exist: {text}"
            )
        paths.append(path)
    if len(paths) > MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"MOSS-Image reference conditioning supports at most {MAX_REFERENCE_IMAGES} images"
        )
    return paths


def _load_reference_images(paths: list[Path]) -> list[Any]:
    from PIL import Image

    images = []
    for path in paths:
        with Image.open(path) as handle:
            images.append(handle.convert("RGB").copy())
    return images


def _reference_tensors(images: list[Any]) -> list[Any]:
    """PIL references -> [0, 1] (3, H, W) torch tensors."""
    import numpy as np
    import torch

    tensors = []
    for image in images:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
    return tensors


def _build_image_processor() -> Any:
    """Qwen3.5 vision preprocessing constants (the component ships no config)."""
    try:
        from transformers import Qwen2VLImageProcessor as _Processor
    except ImportError:  # pragma: no cover - transformers moved this class before
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor as _Processor,
        )

    return _Processor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        size={"longest_edge": REFERENCE_VLM_PIXELS, "shortest_edge": 65536},
    )


def _vlm_reference_images(images: list[Any]) -> list[Any]:
    """Downscale references for the vision tower (never upscale)."""
    import math

    from PIL import Image

    prepared = []
    for image in images:
        width, height = image.size
        scale = min(1.0, math.sqrt(REFERENCE_VLM_PIXELS / float(width * height)))
        if scale < 1.0:
            size = (
                max(round(height * scale), 32),
                max(round(width * scale), 32),
            )
            image = image.resize(size, Image.BICUBIC)
        prepared.append(image)
    return prepared


def _text_input_device(text_encoder: Any) -> str:
    """Where to place text-encoder inputs.

    Accelerate's offload hooks stream each submodule's weights to the execution
    device and move that module's inputs there too, so CUDA inputs are correct
    for every placement strategy. Without CUDA the encoder runs on CPU.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    try:
        parameter = next(text_encoder.parameters())
    except StopIteration:
        return "cpu"
    return str(parameter.device)


def _encode_prompt_with_refs(
    pipe: Any, prompt: str, images: list[Any], max_sequence_length: int
) -> Any:
    """Chat-template prompt with ``Picture N:`` vision blocks -> (S, text_dim)."""
    import torch

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    if tokenizer is None or text_encoder is None:
        raise RuntimeError("MOSS-Image text encoder/tokenizer are required for references")

    processor = _build_image_processor()
    device = _text_input_device(text_encoder)
    pixels = processor(
        images=_vlm_reference_images(images), return_tensors="pt", do_rescale=False
    )
    grid = pixels["image_grid_thw"]
    counts = (grid.prod(-1) // processor.merge_size**2).tolist()
    content = "".join(
        f"Picture {index + 1}: <|vision_start|>"
        f"{'<|image_pad|>' * count}<|vision_end|>"
        for index, count in enumerate(counts)
    )
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": content + prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    image_token_id = int(getattr(text_encoder.config, "image_token_id", 248056))
    mm_token_type_ids = (input_ids == image_token_id).int()
    if int(mm_token_type_ids.sum()) != sum(counts):
        raise RuntimeError(
            "MOSS-Image reference placeholder count does not match the vision grid"
        )
    if input_ids.shape[1] > max_sequence_length:
        raise RuntimeError(
            f"MOSS-Image prompt is {input_ids.shape[1]} tokens, above the "
            f"{max_sequence_length}-token limit"
        )
    with torch.no_grad():
        output = text_encoder(
            input_ids=input_ids,
            pixel_values=pixels["pixel_values"].to(device),
            image_grid_thw=grid.to(device),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            return_dict=True,
        )
    return output.last_hidden_state[0]


def _encode_ref_latents(
    pipe: Any, tensors: list[Any], target_pixels: int
) -> list[Any]:
    """[0, 1] references -> normalized (128, h, w) latents on the VAE device."""
    import math

    import torch
    import torch.nn.functional as F

    vae = pipe.vae
    if vae is None:
        raise RuntimeError("MOSS-Image VAE is required for reference conditioning")
    bn = vae.bn
    eps = float(vae.config.batch_norm_eps)
    # With Accelerate offload the module can report ``meta`` between forwards;
    # feed inputs on CPU and let the hooks stream the layers.
    vae_device = vae.device
    if str(vae_device) == "meta":
        vae_device = "cpu"
    latents = []
    for tensor in tensors:
        _, height, width = tensor.shape
        if height * width > target_pixels:
            ratio = height / float(width)
            new_h = math.sqrt(target_pixels * ratio)
            new_w = new_h / ratio
        else:
            new_h, new_w = float(height), float(width)
        new_h = max(16, int(round(new_h / 16.0)) * 16)
        new_w = max(16, int(round(new_w / 16.0)) * 16)
        if (new_h, new_w) != (height, width):
            tensor = F.interpolate(
                tensor[None].float(), size=(new_h, new_w), mode="bilinear"
            )[0]
        pixels = (tensor * 2 - 1).to(device=vae_device, dtype=vae.dtype)[None]
        with torch.no_grad():
            encoded = vae.encode(pixels).latent_dist.mode()
        batch, channels, h8, w8 = encoded.shape
        packed = (
            encoded.reshape(batch, channels, h8 // 2, 2, w8 // 2, 2)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(batch, channels * 4, h8 // 2, w8 // 2)
        )
        packed = (packed - bn.running_mean.to(packed).view(1, -1, 1, 1)) / torch.sqrt(
            bn.running_var.to(packed).view(1, -1, 1, 1) + eps
        )
        latents.append(packed[0])
    return latents


def _ref_rope_coords(index: int, frame: int, height: int, width: int, device: str):
    """(H*W, 4) RoPE coordinates of one reference latent grid."""
    import torch

    rows = torch.arange(height, device=device, dtype=torch.float32)
    cols = torch.arange(width, device=device, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(rows, cols, indexing="ij")
    coords = torch.zeros(height * width, 4, device=device, dtype=torch.float32)
    coords[:, 0] = float(index)
    coords[:, 1] = float(frame)
    coords[:, 2] = (grid_h - (height - 1) / 2.0).flatten()
    coords[:, 3] = (grid_w - (width - 1) / 2.0).flatten()
    return coords


def _rope_from_coords(bands: Any, coords: Any) -> Any:
    import torch

    phase = coords.unsqueeze(-1) * bands
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)


def _rotate(x: Any, rope: Any) -> Any:
    import torch

    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _situ(hidden: Any, gate: Any) -> Any:
    import torch

    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )


def _ref_attention(q, k, v, rope, heads, split, isolate, kv_reuse, kv_capture):
    """Attention that can isolate reference tokens and reuse cached ref K/V."""
    import torch
    import torch.nn.functional as F

    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    query = _rotate(q.reshape(shape), rope).transpose(1, 2)
    key = _rotate(k.reshape(shape), rope).transpose(1, 2)
    value = v.reshape(shape).transpose(1, 2)

    def attend(qq, kk, vv):
        return F.scaled_dot_product_attention(qq.to(vv.dtype), kk.to(vv.dtype), vv)

    if split is not None and isolate:
        if kv_capture is not None:
            kv_capture.append(
                (key[:, :, split:].clone(), value[:, :, split:].clone())
            )
        out = torch.cat(
            (
                attend(query[:, :, :split], key[:, :, :split], value[:, :, :split]),
                attend(query[:, :, split:], key[:, :, split:], value[:, :, split:]),
            ),
            2,
        )
    else:
        if kv_reuse is not None:
            cached_key, cached_value = kv_reuse
            key = torch.cat((key, cached_key), 2)
            value = torch.cat((value, cached_value), 2)
        out = attend(query, key, value)
    return out.transpose(1, 2).flatten(2)


def _ref_block_forward(block, x, rope, mod, ref_mod, split, isolate, kv_reuse, kv_capture):
    """One DiT block with ``[live | reference]`` token modulation.

    Mirrors the vendor ``_Block.forward``; live tokens follow the sampling
    timestep while reference tokens stay at t=0.
    """
    if split is not None:
        live_mod, ref_only_mod = mod.unbind(1), ref_mod.unbind(1)

        def affine(hidden, scale, shift):
            import torch

            return torch.cat(
                (
                    hidden[:, :split] * (1 + live_mod[scale][:, None])
                    + live_mod[shift][:, None],
                    hidden[:, split:] * (1 + ref_only_mod[scale][:, None])
                    + ref_only_mod[shift][:, None],
                ),
                1,
            )

        def gate(hidden, index):
            import torch

            return torch.cat(
                (
                    hidden[:, :split] * live_mod[index][:, None],
                    hidden[:, split:] * ref_only_mod[index][:, None],
                ),
                1,
            )

    else:
        live_mod = mod.unbind(1)

        def affine(hidden, scale, shift):
            return hidden * (1 + live_mod[scale][:, None]) + live_mod[shift][:, None]

        def gate(hidden, index):
            return hidden * live_mod[index][:, None]

    dtype = block.proj["q"].weight.dtype
    normalized = affine(block.norm1(x), 1, 0).to(dtype)
    q = block.q_norm(
        block.proj["q"](normalized).unflatten(-1, (block.heads, -1))
    ).flatten(2)
    k = block.k_norm(
        block.proj["k"](normalized).unflatten(-1, (block.heads, -1))
    ).flatten(2)
    attention = _ref_attention(
        q,
        k,
        block.proj["v"](normalized),
        rope,
        block.heads,
        split,
        isolate,
        kv_reuse,
        kv_capture,
    )
    x = x + gate(block.proj["o"](attention), 2)
    normalized = affine(block.norm2(x), 4, 3).to(dtype)
    hidden, ffn_gate = block.proj["ffn_up_gate"](normalized).chunk(2, -1)
    return x + gate(block.proj["ffn_down"](_situ(hidden, ffn_gate)), 5)


def _forward_with_refs(
    transformer: Any,
    module: Any,
    hidden_states: Any,
    timestep: Any,
    encoder_hidden_states: Any,
    encoder_attention_mask: Any = None,
    *,
    skip_layers: Any = None,
    ref_latents: Any = None,
    isolate_refs: bool = False,
    ref_kv_cache: Any = None,
):
    """Vendor DiT forward plus clean reference tokens (ai-toolkit edit layout).

    Reference latents ride after the text tokens with t=0 modulation. With
    ``isolate_refs`` their K/V are captured on the first call and reused on
    later calls, which keeps the per-step cost at the text+image sequence.
    """
    import torch

    skip = module._validate_skip_layers(skip_layers, len(transformer.layers))
    if ref_kv_cache is not None and not isolate_refs:
        raise ValueError("Reference K/V caching requires isolate_refs")
    reuse = ref_kv_cache is not None and bool(ref_kv_cache.get("kv"))

    batch, _, height, width = hidden_states.shape
    mask = encoder_attention_mask
    if mask is None:
        mask = torch.ones(
            encoder_hidden_states.shape[:2],
            dtype=torch.bool,
            device=hidden_states.device,
        )
    device = hidden_states.device
    times = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1)
    times = times.expand(batch)

    half = transformer.config.freq_dim // 2
    freq = 10000 ** (
        -torch.arange(half, device=device, dtype=torch.float64) / half
    )

    def modulation(values):
        phase = values.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = transformer.time_embedding(
            embedding.to(transformer.time_embedding[0].weight.dtype)
        ).float()
        mod = transformer.shared_block_modulation(
            time.to(transformer.shared_block_modulation.weight.dtype)
        ).view(values.shape[0], 6, -1)
        return time, mod

    time, mod = modulation(times)
    _, ref_mod_all = modulation(torch.zeros_like(times))
    head = transformer.video_head_modulation(
        time.to(transformer.video_head_modulation.weight.dtype)
    ).view(batch, 2, -1)

    captured: dict[int, Any] = {}
    outputs = []
    for index in range(batch):
        context = encoder_hidden_states[index : index + 1, mask[index].bool()]
        if context.shape[1] == 0:
            raise ValueError("Each sample needs at least one valid text token.")
        dtype = transformer.video_embedder.weight.dtype
        video = transformer.video_embedder(
            hidden_states[index : index + 1]
            .flatten(2)
            .transpose(1, 2)
            .to(dtype)
            .contiguous()
        )
        rope = transformer.rope(height, width, context.shape[1], device).to(video.dtype)
        text = transformer.text_embedder(context, rope[:, height * width :])
        x = torch.cat((video, text), 1)
        num_live = height * width + context.shape[1]

        refs = ref_latents[index] if ref_latents is not None else None
        split = None
        if refs and not reuse:
            tokens, coords = [], []
            for ref_index, latent in enumerate(refs):
                _, ref_h, ref_w = latent.shape
                tokens.append(latent.to(device=device, dtype=dtype).flatten(1).transpose(0, 1))
                coords.append(
                    _ref_rope_coords(
                        ref_index + 1, context.shape[1], ref_h, ref_w, device
                    )
                )
            ref_tokens = torch.cat(tokens)
            ref_rope = _rope_from_coords(
                transformer.rope.bands.float(), torch.cat(coords)
            ).to(video.dtype)
            x = torch.cat(
                (x, transformer.video_embedder(ref_tokens.contiguous().unsqueeze(0))), 1
            )
            rope = torch.cat((rope, ref_rope.unsqueeze(0)), 1)
            split = num_live

        sample_mod = mod[index : index + 1]
        sample_ref_mod = ref_mod_all[index : index + 1]

        for layer_index, layer in enumerate(transformer.layers):
            if layer_index in skip:
                continue
            if split is None and not reuse:
                x = layer(x, rope, sample_mod)
                continue
            slot: list[Any] | None = (
                [] if (ref_kv_cache is not None and not reuse) else None
            )
            x = _ref_block_forward(
                layer,
                x,
                rope,
                sample_mod,
                sample_ref_mod,
                split,
                isolate_refs,
                ref_kv_cache["kv"].get(layer_index) if reuse else None,
                slot,
            )
            if slot is not None:
                captured[layer_index] = slot[0]

        if ref_kv_cache is not None and not reuse and captured:
            ref_kv_cache["kv"] = captured

        x = transformer.final_norm_video(
            x[:, : height * width].to(transformer.final_norm_video.weight.dtype)
        )
        x = x * (1 + head[index : index + 1, 1, None]) + head[
            index : index + 1, 0, None
        ]
        x = transformer.final_linear_video(
            x.to(transformer.final_linear_video.weight.dtype)
        )
        outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))

    sample = torch.cat(outputs)
    return module.Transformer2DModelOutput(sample=sample)


def _sample_with_refs(
    pipe: Any,
    module: Any,
    positive: Any,
    positive_mask: Any,
    negative: Any,
    negative_mask: Any,
    *,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    generator: Any,
    uncond_skip_layers: tuple[int, ...],
    ref_latents: list[Any],
    isolate_refs: bool,
    progress_path: Path | None = None,
) -> list[Any]:
    """Vendor sampling loop with reference conditioning and per-branch K/V caches."""
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    transformer = pipe.transformer
    scheduler = pipe.scheduler
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else next(transformer.parameters()).device
    )
    if (
        device.type == "cuda"
        and getattr(transformer, "_hf_hook", None) is None
        and next(transformer.parameters()).device.type != "cuda"
    ):
        # A previous sample parked the DiT for VAE decoding.
        transformer.to("cuda")
    positive = positive.to(device)
    positive_mask = positive_mask.to(device)
    if negative is not None:
        negative = negative.to(device)
    if negative_mask is not None:
        negative_mask = negative_mask.to(device)
    scheduler.set_timesteps(steps, device=device)
    timesteps = list(scheduler.timesteps)
    shape = (1, transformer.config.latent_channels, height // 16, width // 16)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)

    caches: dict[tuple[int, tuple[int, ...]], dict[str, Any]] = {}

    def cache_for(mask: Any, skip: tuple[int, ...]) -> dict[str, Any] | None:
        if not isolate_refs:
            return None
        key = (int(mask[0].sum()), tuple(sorted(skip)))
        return caches.setdefault(key, {})

    total = max(1, len(timesteps))
    with torch.no_grad():
        for step_index, timestep in enumerate(timesteps):
            _write_progress(
                progress_path,
                "Denoising with MOSS-Image",
                0.55 + (0.35 * step_index / total),
                step_index + 1,
                total,
            )
            prediction = _forward_with_refs(
                transformer,
                module,
                latents,
                timestep,
                positive,
                positive_mask,
                ref_latents=[ref_latents],
                isolate_refs=isolate_refs,
                ref_kv_cache=cache_for(positive_mask, ()),
            ).sample.float()
            if float(guidance_scale) != 1.0:
                uncond = _forward_with_refs(
                    transformer,
                    module,
                    latents,
                    timestep,
                    negative,
                    negative_mask,
                    skip_layers=uncond_skip_layers,
                    ref_latents=[ref_latents],
                    isolate_refs=isolate_refs,
                    ref_kv_cache=cache_for(negative_mask, uncond_skip_layers),
                ).sample.float()
                prediction = uncond + float(guidance_scale) * (prediction - uncond)
            latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]

    # 1024^2 VAE decoding does not fit beside the resident 5B DiT on 16 GB
    # cards. This worker exits after the job, so park the DiT for the decode.
    # Models managed by Accelerate hooks already live on CPU between forwards.
    if torch.cuda.is_available() and getattr(transformer, "_hf_hook", None) is None:
        transformer.to("cpu")
        torch.cuda.empty_cache()

    vae = pipe.vae
    latents = latents.to(dtype=vae.dtype)
    bn = vae.bn
    eps = float(vae.config.batch_norm_eps)
    latents = latents * (bn.running_var + eps).sqrt().to(latents).view(
        1, -1, 1, 1
    ) + bn.running_mean.to(latents).view(1, -1, 1, 1)
    batch, channels, latent_h, latent_w = latents.shape
    latents = latents.reshape(
        batch, channels // 4, 2, 2, latent_h, latent_w
    ).permute(0, 1, 4, 2, 5, 3).reshape(
        batch, channels // 4, 2 * latent_h, 2 * latent_w
    )
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0].clamp(-1, 1)
    return pipe.image_processor.postprocess(image, output_type="pil")


def _attach_step_progress(pipe: Any, progress_path: Path | None) -> None:
    """Report denoising progress through the pipeline's step iterator."""
    if progress_path is None:
        return

    def wrapper(iterable: Iterable[Any]):
        items = list(iterable)
        total = max(1, len(items))
        for index, item in enumerate(items, start=1):
            _write_progress(
                progress_path,
                "Denoising with MOSS-Image",
                0.55 + (0.35 * (index - 1) / total),
                index,
                len(items),
            )
            yield item

    pipe.progress_bar = wrapper


def _load_pipeline(request: dict, dtype: Any, progress_path: Path | None) -> tuple[Any, dict[str, Any]]:
    import torch
    from diffusers import AutoencoderKLFlux2
    from transformers import AutoModel, AutoTokenizer

    component_source = _resolve_component_source(
        str(request.get("component_model") or DEFAULT_COMPONENT_MODEL).strip()
    )
    model_path = Path(str(request["model_path"])).expanduser()
    model_format = str(request.get("model_format") or "").lower()
    local_files_only = _local_files_only()

    _write_progress(progress_path, "Loading MOSS-Image pipeline", 0.08)
    module = _load_pipeline_module(component_source)

    # Validate adapter paths before paying for the 10 GB checkpoint overlay.
    loras = request.get("loras") or []
    for entry in loras:
        lora_path = Path(str(entry.get("path") or "")).expanduser()
        if not lora_path.is_file():
            raise RuntimeError(f"MOSS-Image LoRA file does not exist: {lora_path}")

    scheduler = module.MOSSImageScheduler()

    if model_format == "diffusers" and model_path.is_dir():
        _write_progress(progress_path, "Loading MOSS-Image Diffusers model", 0.12)
        transformer = module.MOSSImageTransformer.from_pretrained(
            str(model_path),
            subfolder="transformer",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval()
        component_root = str(model_path)
        runtime = {"source": "diffusers", "component_source": component_root}
    else:
        if not model_path.is_file():
            raise RuntimeError(f"MOSS-Image checkpoint path does not exist: {model_path}")

        _write_progress(progress_path, "Building MOSS-Image transformer", 0.10)
        transformer = _build_empty_transformer(module, component_source)

        def overlay_progress(current: int, total: int) -> None:
            fraction = float(current) / float(max(1, total))
            _write_progress(
                progress_path,
                "Loading MOSS-Image checkpoint",
                0.12 + (0.18 * fraction),
                current,
                total,
            )

        _write_progress(progress_path, "Loading MOSS-Image checkpoint", 0.12)
        overlay = overlay_single_file_transformer(
            transformer,
            model_path,
            target_dtype=dtype,
            progress_callback=overlay_progress,
        )
        transformer.eval()
        component_root = component_source
        runtime = {
            "source": "single",
            "component_source": component_source,
            "mapped_tensors": overlay["mapped_tensors"],
        }

    if loras:
        def lora_progress(current: int, total: int, name: str) -> None:
            fraction = float(current) / float(max(1, total))
            _write_progress(
                progress_path,
                f"Loading MOSS-Image LoRA {current}/{total}: {name}",
                0.30 + (0.03 * fraction),
            )

        runtime["loras"] = apply_moss_loras(
            transformer, loras, progress_callback=lora_progress
        )

    _write_progress(progress_path, "Loading MOSS-Image text encoder", 0.34)
    tokenizer = AutoTokenizer.from_pretrained(
        component_root,
        subfolder="tokenizer",
        local_files_only=local_files_only,
    )
    text_encoder = AutoModel.from_pretrained(
        component_root,
        subfolder="text_encoder",
        dtype=dtype,
        local_files_only=local_files_only,
    ).eval()

    _write_progress(progress_path, "Loading MOSS-Image VAE", 0.44)
    vae = AutoencoderKLFlux2.from_pretrained(
        component_root,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
    ).eval()

    pipe = module.MOSSImagePipeline(
        transformer=transformer,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
    )
    # The reference-conditioning sampler builds vendor ``Transformer2DModelOutput``
    # values, so the pipeline module it came from must stay reachable.
    pipe._webbduck_module = module
    return pipe, runtime


def _run(request: dict, output_dir: Path, progress_path: Path | None = None) -> dict:
    import torch

    prompt = str(request.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Prompt is required")

    negative_prompt = str(request.get("negative_prompt") or "")
    width = _snap(int(request.get("width") or 1024))
    height = _snap(int(request.get("height") or 1024))
    steps = max(1, int(request.get("steps") or 50))
    cfg = float(request.get("guidance_scale") if request.get("guidance_scale") is not None else 3.0)
    num_images = max(1, int(request.get("num_images") or 1))
    seed = int(request.get("seed") if request.get("seed") is not None else 0)
    uncond_skip_layers = tuple(request.get("uncond_skip_layers") or (22,))
    max_sequence_length = max(1, int(request.get("max_sequence_length") or 4096))

    # Validate reference images before paying for the ~10 GB checkpoint overlay.
    reference_paths = _reference_paths(request)
    reference_images = _load_reference_images(reference_paths) if reference_paths else []
    reference_tensors = _reference_tensors(reference_images) if reference_images else []
    reference_pixels = max(
        16 * 16, int(request.get("reference_pixels") or REFERENCE_LATENT_PIXELS)
    )
    isolate_references = bool(request.get("isolate_references", True))
    use_references = bool(reference_tensors)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            dtype = torch.float16
    else:
        dtype = torch.float32

    total_vram_gb = 0.0
    if device == "cuda":
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            total_vram_gb = 0.0

    if use_references:
        reference_pixels = _effective_reference_pixels(reference_pixels, total_vram_gb)

    _write_progress(progress_path, "Preparing MOSS-Image runtime", 0.04)
    pipe, runtime = _load_pipeline(request, dtype, progress_path)

    offload = _configure_memory(pipe, device, total_vram_gb)
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    prompt_embeds = None
    reference_embeds = None
    if use_references:
        reference_embeds = _encode_prompt_embeds_with_refs(
            pipe,
            prompt,
            negative_prompt,
            reference_images,
            cfg,
            max_sequence_length,
            progress_path,
            stream_encoder=offload == "split",
        )
    elif offload == "split":
        prompt_embeds = _encode_prompt_embeds(
            pipe,
            prompt,
            negative_prompt,
            cfg,
            max_sequence_length,
            progress_path,
        )
    if offload == "split":
        _move_split_models_to_gpu(pipe, progress_path)

    reference_latents = None
    if use_references:
        _write_progress(progress_path, "Encoding MOSS-Image reference images", 0.52)
        reference_latents = _encode_ref_latents(
            pipe, reference_tensors, reference_pixels
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _attach_step_progress(pipe, progress_path)
    saved: list[str] = []
    for index in range(num_images):
        generator = torch.Generator("cpu").manual_seed(seed + index)
        if use_references:
            module = getattr(pipe, "_webbduck_module", None)
            if module is None:
                raise RuntimeError(
                    "MOSS-Image pipeline module is required for reference conditioning"
                )
            positive, positive_mask, negative, negative_mask = reference_embeds
            images = _sample_with_refs(
                pipe,
                module,
                positive,
                positive_mask,
                negative,
                negative_mask,
                height=height,
                width=width,
                steps=steps,
                guidance_scale=cfg,
                generator=generator,
                uncond_skip_layers=uncond_skip_layers,
                ref_latents=reference_latents,
                isolate_refs=isolate_references,
                progress_path=progress_path,
            )
            image = images[0]
        else:
            call_kwargs: dict[str, Any] = {
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
                "generator": generator,
                "uncond_skip_layers": uncond_skip_layers,
                "max_sequence_length": max_sequence_length,
            }
            if prompt_embeds is not None:
                positive, positive_mask, negative, negative_mask = prompt_embeds
                call_kwargs.update(
                    prompt_embeds=positive,
                    prompt_attention_mask=positive_mask,
                    negative_prompt_embeds=negative,
                    negative_prompt_attention_mask=negative_mask,
                )
            else:
                if index == 0:
                    _write_progress(progress_path, "Encoding MOSS-Image prompt", 0.50)
                call_kwargs.update(prompt=prompt, negative_prompt=negative_prompt)
            result = pipe(**call_kwargs)
            image = result.images[0]
        path = output_dir / f"moss_image_{index:03d}.png"
        image.save(path)
        saved.append(str(path))

    _write_progress(progress_path, "MOSS-Image complete", 0.98)
    result_runtime = {
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "offload": offload,
        "total_vram_gb": round(total_vram_gb, 2),
        **runtime,
    }
    if use_references:
        result_runtime["reference_images"] = len(reference_paths)
        result_runtime["reference_pixels"] = reference_pixels
        result_runtime["isolate_references"] = isolate_references
    return {
        "ok": True,
        "images": saved,
        "seed": seed,
        "runtime": result_runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress")
    args = parser.parse_args()

    result_path = Path(args.result)
    progress_path = Path(args.progress) if args.progress else None
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        result = _run(request, Path(args.output_dir), progress_path)
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
