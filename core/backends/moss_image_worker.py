"""Standalone MOSS-Image Diffusers worker.

MOSS-Image is distributed as a small remote Diffusers pipeline
(``pipeline_moss_image.py``) plus a Qwen3.5 text encoder and a FLUX.2 VAE.
Community/ComfyUI releases usually ship only the 5B DiT as one safetensors
file, so this worker:

1. loads the pipeline module from the component repository,
2. builds the DiT on the meta device from the published transformer config,
3. streams the local single-file DiT into that scaffold, and
4. runs text-to-image with the component text encoder + VAE.

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
    pipe: Any, prompts: list[str], max_sequence_length: int
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


def _move_split_models_to_gpu(pipe: Any, progress_path: Path | None = None) -> None:
    """Place the DiT and VAE on the GPU after prompt encoding is complete."""
    import gc

    import torch

    _write_progress(progress_path, "Moving MOSS-Image models to GPU", 0.49)
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    gc.collect()
    torch.cuda.empty_cache()


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
    if offload == "split":
        prompt_embeds = _encode_prompt_embeds(
            pipe,
            prompt,
            negative_prompt,
            cfg,
            max_sequence_length,
            progress_path,
        )
        _move_split_models_to_gpu(pipe, progress_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    _attach_step_progress(pipe, progress_path)
    saved: list[str] = []
    for index in range(num_images):
        generator = torch.Generator("cpu").manual_seed(seed + index)
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
    return {
        "ok": True,
        "images": saved,
        "seed": seed,
        "runtime": {
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "offload": offload,
            "total_vram_gb": round(total_vram_gb, 2),
            **runtime,
        },
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
