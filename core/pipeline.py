"""SDXL pipeline construction and management."""

import os
import json
import torch
import threading
import gc
import time
import re
import logging
from pathlib import Path

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    UniPCMultistepScheduler,
)

from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from huggingface_hub import snapshot_download

from safetensors.torch import load_file
from models.registry import MODEL_REGISTRY, LORA_REGISTRY, EMBEDDING_REGISTRY
from core.exceptions import GenerationCancelledError
from core.perf import stage_timer
from core.runtime import resolve_runtime_profile

from core.schedulers import create_scheduler

log = logging.getLogger(__name__)

# Disable torch compilation
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ.pop("TORCH_LOGS", None)
os.environ.pop("TORCHINDUCTOR_VERBOSE", None)
os.environ["ACCELERATE_DISABLE_RICH"] = "1"

# Torch settings
torch._dynamo.disable()
torch.set_float32_matmul_precision("high")

RUNTIME_PROFILE = resolve_runtime_profile(logger=log)
DEVICE = RUNTIME_PROFILE.device
DTYPE = RUNTIME_PROFILE.dtype

# ── GPU-aware backend tuning ──────────────────────────────────────────────
_ATTENTION_SLICING = None   # set after GPU detection below

def _configure_torch_backends():
    global _ATTENTION_SLICING
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    cc = torch.cuda.get_device_capability(0)
    sdp = getattr(torch.backends.cuda, "cudnn_sdp_enabled", None)
    if sdp is not None:
        enable_sdp = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
        if enable_sdp is not None and cc >= (8, 0):
            enable_sdp(True)
        elif enable_sdp is not None:
            enable_sdp(False)

    # Choose attention slicing based on VRAM
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb < 8:
        _ATTENTION_SLICING = "max"       # very conservative
    elif vram_gb < 12:
        _ATTENTION_SLICING = "auto"       # diffuser picks slice size
    else:
        _ATTENTION_SLICING = None         # disable – plenty of VRAM

if torch.cuda.is_available():
    _configure_torch_backends()

# Component caches
_UNET_CACHE = {}
_TOKENIZER_CACHE = {}
_TEXT_COMPONENT_CACHE = {}
_VAE_CACHE = {}
_SCHEDULER_CACHE = {}
_REFINER_UNET_CACHE = {}

MAX_CACHE_SIZE = 1
_SINGLE_FILE_CONFIG_CACHE = {}


def _get_single_file_config_path(repo_id: str) -> str | None:
    """Prepare a local Diffusers config snapshot for single-file checkpoints.

    This avoids Windows symlink privilege errors from HF cache operations.
    """
    cached = _SINGLE_FILE_CONFIG_CACHE.get(repo_id)
    if cached:
        return cached

    local_root = Path(
        os.getenv(
            "WEBBDUCK_HF_CONFIG_DIR",
            str(Path.home() / ".cache" / "webbduck" / "hf-configs"),
        )
    )
    local_dir = local_root / repo_id.replace("/", "--")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Only fetch small config/tokenizer files. This avoids large weight downloads.
    allow_patterns = [
        "*.json",
        "**/*.json",
        "*.txt",
        "**/*.txt",
        "*.model",
        "**/*.model",
    ]

    snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        local_dir=str(local_dir),
    )

    config_path = str(local_dir)
    _SINGLE_FILE_CONFIG_CACHE[repo_id] = config_path
    return config_path

def _release_cached_obj(obj):
    """Best-effort release of cached torch modules before eviction."""
    if obj is None:
        return

    if isinstance(obj, dict):
        for v in obj.values():
            _release_cached_obj(v)
        return

    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            _release_cached_obj(v)
        return

    # Move heavy torch modules to CPU before dropping references.
    if hasattr(obj, "to"):
        try:
            obj.to("cpu")
        except Exception:
            pass


def manage_cache(cache, key):
    """Enforce LRU-style cache limit."""
    if key in cache:
        # Move to end (most recently used)
        val = cache.pop(key)
        cache[key] = val
        return

    if len(cache) >= MAX_CACHE_SIZE:
        # Remove oldest (first item)
        oldest_key = next(iter(cache))
        _release_cached_obj(cache.get(oldest_key))
        del cache[oldest_key]
        gc.collect()
        if DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def sanitize_adapter_name(name: str) -> str:
    """Convert LoRA name to valid adapter identifier."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def load_unet_auto(unet_dir) -> UNet2DConditionModel:
    """Load UNet from diffusers directory with automatic shard handling."""
    unet_dir = Path(unet_dir)
    config_path = unet_dir / "config.json"
    
    try:
        config_exists = config_path.exists()
    except OSError as exc:
        raise FileNotFoundError(
            f"Cannot access UNet config at {config_path}. "
            f"This may be a Windows symlink/permission issue with the HuggingFace cache. "
            f"Try enabling Windows Developer Mode or clearing the HF cache. "
            f"Original error: {exc}"
        ) from exc

    if not config_exists:
        raise FileNotFoundError(f"Missing UNet config: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    unet = UNet2DConditionModel.from_config(config, torch_dtype=DTYPE)

    safe_index = unet_dir / "diffusion_pytorch_model.safetensors.index.json"
    safe_file = unet_dir / "diffusion_pytorch_model.safetensors"
    bin_file = unet_dir / "diffusion_pytorch_model.bin"

    try:
        has_safe_index = safe_index.exists()
    except OSError:
        has_safe_index = False

    try:
        has_safe_file = safe_file.exists()
    except OSError:
        has_safe_file = False

    try:
        has_bin_file = bin_file.exists()
    except OSError:
        has_bin_file = False

    if has_safe_index:
        state_dict = {}
        with open(safe_index) as f:
            index = json.load(f)
        for shard in set(index["weight_map"].values()):
            state_dict.update(load_file(unet_dir / shard, device="cpu"))

    elif has_safe_file:
        state_dict = load_file(safe_file, device="cpu")

    elif has_bin_file:
        state_dict = torch.load(bin_file, map_location="cpu")

    else:
        raise FileNotFoundError(f"No UNet weights found in {unet_dir}")

    unet.load_state_dict(state_dict, strict=False)
    return unet


def apply_loras(pipe, loras: list[dict]) -> str:
    """Apply LoRA weights to pipeline and return trigger phrase."""
    trigger_phrases = []

    for entry in loras:
        name = entry["name"]
        weight = float(entry.get("weight", 1.0))
        lora = LORA_REGISTRY[name]

        pipe.load_lora_weights(lora["path"], adapter_name=name)

        if lora.get("trigger"):
            trigger_phrases.append(f"({lora['trigger']}:{weight})")

    pipe.fuse_lora()
    return ", ".join(trigger_phrases)


def _load_sdxl_dual_clip_embedding(pipe, embedding_path: Path, token: str) -> bool:
    """Load SDXL textual inversion exported with `clip_l`/`clip_g` tensors.

    Returns True when this format was detected and loaded, otherwise False.
    """
    suffix = embedding_path.suffix.lower()
    if suffix not in {".safetensors", ".pt", ".bin"}:
        return False

    state = None
    # Some embedding files are safetensors content with a .pt extension.
    # Try both loaders to preserve compatibility with mixed community formats.
    loaders = []
    if suffix == ".safetensors":
        loaders = [
            lambda: load_file(str(embedding_path), device="cpu"),
            lambda: torch.load(str(embedding_path), map_location="cpu"),
        ]
    else:
        loaders = [
            lambda: torch.load(str(embedding_path), map_location="cpu"),
            lambda: load_file(str(embedding_path), device="cpu"),
        ]

    for loader in loaders:
        try:
            state = loader()
            break
        except Exception:
            continue

    if state is None:
        return False

    if not isinstance(state, dict) or "clip_l" not in state or "clip_g" not in state:
        return False

    clip_l = state.get("clip_l")
    clip_g = state.get("clip_g")
    if not isinstance(clip_l, torch.Tensor) or not isinstance(clip_g, torch.Tensor):
        raise ValueError("Embedding file has non-tensor `clip_l`/`clip_g` entries")

    if clip_l.ndim == 1:
        clip_l = clip_l.unsqueeze(0)
    if clip_g.ndim == 1:
        clip_g = clip_g.unsqueeze(0)
    if clip_l.ndim != 2 or clip_g.ndim != 2:
        raise ValueError("Expected 2D tensors for `clip_l` and `clip_g`")
    if clip_l.shape[0] != clip_g.shape[0]:
        raise ValueError(
            f"Mismatched SDXL embedding vectors: clip_l={tuple(clip_l.shape)}, clip_g={tuple(clip_g.shape)}"
        )

    if not hasattr(pipe, "tokenizer_2") or not hasattr(pipe, "text_encoder_2"):
        raise ValueError("SDXL dual-clip embedding requires tokenizer_2/text_encoder_2")

    pipe.load_textual_inversion(
        {token: clip_l},
        token=token,
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
    )
    pipe.load_textual_inversion(
        {token: clip_g},
        token=token,
        tokenizer=pipe.tokenizer_2,
        text_encoder=pipe.text_encoder_2,
    )
    return True


def set_inference_mode(pipe):
    """Set all pipeline components to eval mode."""
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()


def configure_sdxl_additions(pipe):
    """Configure SDXL conditioning based on UNet architecture."""
    unet = pipe.unet
    add_dim = getattr(unet.config, "addition_time_embed_dim", None)
    requires_aesthetics = (add_dim == 512)
    pipe.register_to_config(requires_aesthetics_score=requires_aesthetics)


def build_second_pass_pipeline(model_path: Path, shared):
    """Build a second-pass pipeline (refiner or img2img)."""
    key = str(model_path.resolve())
    manage_cache(_REFINER_UNET_CACHE, key)

    if key not in _REFINER_UNET_CACHE:
        if model_path.is_file():
            unet = UNet2DConditionModel.from_single_file(
                model_path,
                torch_dtype=DTYPE,
                use_safetensors=True,
            )
        else:
            unet = load_unet_auto(model_path / "unet")
        
        unet = unet.to(dtype=DTYPE)
        unet.eval()
        _REFINER_UNET_CACHE[key] = unet

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=DTYPE,
    )
    vae.enable_slicing()
    vae.enable_tiling()

    pipe = StableDiffusionXLImg2ImgPipeline(
        vae=vae,
        unet=_REFINER_UNET_CACHE[key],
        tokenizer=shared["tokenizer"],
        tokenizer_2=shared["tokenizer_2"],
        text_encoder=shared["text_encoder"],
        text_encoder_2=shared["text_encoder_2"],
        scheduler=shared["scheduler"],
    )

    configure_sdxl_additions(pipe)
    set_inference_mode(pipe)
    
    pipe.unet.to("cpu")
    pipe.vae.to("cpu")
    
    return pipe

def get_tokenizer(base_path: Path):
    """Load and cache tokenizers only (lightweight)."""
    key = str(base_path.resolve()) + "_tokenizers"
    
    if key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[key]
    
    tokenizer_path = base_path / "tokenizer"
    tokenizer_2_path = base_path / "tokenizer_2"
    
    # Check if folder structure exists (Diffusers format)
    use_local_tokenizers = False
    try:
        use_local_tokenizers = tokenizer_path.exists() and tokenizer_2_path.exists()
    except OSError:
        # Windows cache access issues
        use_local_tokenizers = False
    
    if use_local_tokenizers:
        try:
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
            tokenizer_2 = CLIPTokenizer.from_pretrained(tokenizer_2_path)
        except (TypeError, OSError, ValueError) as exc:
            # Tokenizer folder exists but is broken (e.g., missing vocab_file, symlink issues)
            log.warning(
                "Failed to load model tokenizers from %s (%s), falling back to standard SDXL tokenizers",
                base_path, exc,
            )
            use_local_tokenizers = False
    
    if not use_local_tokenizers:
        # Fallback for single-file checkpoints, missing folders, or broken tokenizers: use standard SDXL tokenizers
        # These are small downloads and cached
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            pad_token="!", # SDXL uses '!' padding for openclip
        )
    
    manage_cache(_TOKENIZER_CACHE, key)
    _TOKENIZER_CACHE[key] = (tokenizer, tokenizer_2)
    return tokenizer, tokenizer_2


def get_text_components(base_path: Path):
    """Load and cache text encoders and tokenizers."""
    key = str(base_path.resolve())

    if key in _TEXT_COMPONENT_CACHE:
        return _TEXT_COMPONENT_CACHE[key]

    try:
        tokenizer = CLIPTokenizer.from_pretrained(base_path / "tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(base_path / "tokenizer_2")

        text_encoder = CLIPTextModel.from_pretrained(
            base_path / "text_encoder",
            torch_dtype=DTYPE,
        ).to(DEVICE)
        text_encoder.config.output_hidden_states = True
        text_encoder.eval()

        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            base_path / "text_encoder_2",
            torch_dtype=DTYPE,
        ).to(DEVICE)
        text_encoder_2.eval()
    except (TypeError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to load text components from {base_path}. "
            f"This may be due to a corrupted HuggingFace cache or Windows symlink issues. "
            f"Try enabling Windows Developer Mode or clearing the HF cache. "
            f"Original error: {exc}"
        ) from exc

    manage_cache(_TEXT_COMPONENT_CACHE, key)
    _TEXT_COMPONENT_CACHE[key] = (tokenizer, tokenizer_2, text_encoder, text_encoder_2)
    return _TEXT_COMPONENT_CACHE[key]


def get_vae():
    """Load and cache VAE."""
    key = "sdxl"

    if key not in _VAE_CACHE:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=DTYPE,
        ).to(DEVICE)
        vae.enable_tiling()
        vae.enable_slicing()
        vae.eval()
        _VAE_CACHE[key] = vae

    return _VAE_CACHE[key]


def get_scheduler(base_path: Path):
    """Load and cache scheduler."""
    key = str(base_path.resolve())

    if key not in _SCHEDULER_CACHE:
        try:
            scheduler = UniPCMultistepScheduler.from_pretrained(
                base_path / "scheduler"
            )
        except (TypeError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to load scheduler from {base_path / 'scheduler'}. "
                f"This may be due to a corrupted HuggingFace cache or Windows symlink issues. "
                f"Original error: {exc}"
            ) from exc
        scheduler.config.sigma_min = 0.029
        scheduler.config.use_karras_sigmas = True
        _SCHEDULER_CACHE[key] = scheduler

    return _SCHEDULER_CACHE[key]


def get_cached_unet(base_path: Path) -> UNet2DConditionModel:
    """Load and cache UNet."""
    key = str(base_path.resolve())
    manage_cache(_UNET_CACHE, key)

    if key not in _UNET_CACHE:
        unet = load_unet_auto(base_path / "unet")
        unet.to(DEVICE, dtype=DTYPE)
        _UNET_CACHE[key] = unet

    return _UNET_CACHE[key]


def build_pipeline(
    base_model_name: str,
    second_pass_model_name: str | None = None,
    lora_names: list[str] | None = None,
    load_progress_callback=None,
):
    """Build SDXL pipeline with optional second pass model and LoRAs."""
    base_entry = MODEL_REGISTRY[base_model_name]
    if callable(load_progress_callback):
        load_progress_callback("Loading VAE", 1)
    vae = get_vae()

    # Single-file checkpoint
    if base_entry["type"] == "single":
        config_repo = base_entry.get("config_repo", "stabilityai/stable-diffusion-xl-base-1.0")
        single_file_kwargs = {
            "torch_dtype": DTYPE,
            "use_safetensors": True,
        }

        try:
            config_path = _get_single_file_config_path(config_repo)
            if config_path:
                single_file_kwargs["config"] = config_path
                single_file_kwargs["local_files_only"] = True
        except Exception as exc:
            log.warning("Could not prepare local single-file config (%s): %s", config_repo, exc)

        if callable(load_progress_callback):
            load_progress_callback("Loading checkpoint", 2)
        try:
            pipe = StableDiffusionXLPipeline.from_single_file(
                base_entry["path"],
                **single_file_kwargs,
            ).to(DEVICE)
        except OSError as exc:
            # Windows symlink privilege issue from HF cache internals.
            if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
                raise RuntimeError(
                    "Hugging Face cache symlink permission error on Windows. "
                    "Enable Windows Developer Mode or run once as Administrator."
                ) from exc
            raise

        if callable(load_progress_callback):
            load_progress_callback("Attaching VAE", 3)
        pipe.vae = vae
        pipe.vae.to(DEVICE, dtype=DTYPE)

        trigger_phrase = ""
        if lora_names:
            trigger_phrase = apply_loras(pipe, lora_names)

        return pipe, None, trigger_phrase

    # Diffusers checkpoint
    base_path: Path = base_entry["path"]

    if callable(load_progress_callback):
        load_progress_callback("Loading UNet", 2)
    unet = get_cached_unet(base_path)
    if callable(load_progress_callback):
        load_progress_callback("Loading text encoders", 3)
    tokenizer, tokenizer_2, text_encoder, text_encoder_2 = get_text_components(base_path)
    if callable(load_progress_callback):
        load_progress_callback("Loading scheduler", 4)
    scheduler = get_scheduler(base_path)

    if callable(load_progress_callback):
        load_progress_callback("Assembling pipeline", 5)
    pipe = StableDiffusionXLPipeline(
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        scheduler=scheduler,
    )

    pipe.to(DEVICE)
    if _ATTENTION_SLICING:
        pipe.enable_attention_slicing(_ATTENTION_SLICING)
    set_inference_mode(pipe)

    trigger_phrase = ""
    if lora_names:
        trigger_phrase = apply_loras(pipe, lora_names)

    # Optional second pass model
    img2img = None
    if second_pass_model_name:
        model_entry = MODEL_REGISTRY[second_pass_model_name]
        model_path: Path = model_entry["path"]

        second_pass_unet = load_unet_auto(model_path / "unet").to(DEVICE, DTYPE)

        img2img = StableDiffusionXLImg2ImgPipeline(
            vae=vae,
            unet=second_pass_unet,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            scheduler=scheduler,
        )

        img2img.to("cpu")

    return pipe, img2img, trigger_phrase


def destroy_pipeline(pipe, img2img):
    """Safely destroy pipeline and free VRAM."""
    try:
        if img2img:
            img2img.to("cpu")
    except Exception:
        pass

    del pipe
    del img2img

    gc.collect()
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class PipelineManager:
    """Manages SDXL pipeline lifecycle and model switching."""
    
    def __init__(self):
        self.pipe = None
        self.img2img = None
        self.base_img2img = None
        self.base_inpaint = None
        self.shared = {}
        self.trigger_phrase = ""
        self.key = None
        self.scheduler_name = None
        self.base_scheduler_config = None
        self.last_used = 0
        self.lock = threading.Lock()

        self.current_second_pass_model = None
        self.current_loras = {}
        self.current_embeddings = {}

    def _ensure_not_cancelled(self, cancel_event):
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Generation cancelled while loading model components")

    def _offload_pipeline(self, pipe_obj):
        if pipe_obj is None:
            return
        try:
            pipe_obj.to("cpu")
        except Exception:
            pass
        for attr in ("unet", "vae", "text_encoder", "text_encoder_2"):
            comp = getattr(pipe_obj, attr, None)
            if comp is not None and hasattr(comp, "to"):
                try:
                    comp.to("cpu")
                except Exception:
                    pass

    def _clear_component_caches(self):
        caches = (
            _UNET_CACHE,
            _REFINER_UNET_CACHE,
            _TEXT_COMPONENT_CACHE,
            _VAE_CACHE,
            _SCHEDULER_CACHE,
            _TOKENIZER_CACHE,
        )
        for cache in caches:
            for v in list(cache.values()):
                _release_cached_obj(v)
            cache.clear()

    def _unload_all_locked(self):
        self._offload_pipeline(self.img2img)
        self._offload_pipeline(self.base_img2img)
        self._offload_pipeline(self.base_inpaint)
        self._offload_pipeline(self.pipe)

        self.pipe = None
        self.img2img = None
        self.base_img2img = None
        self.base_inpaint = None
        self.shared = {}
        self.trigger_phrase = ""
        self.key = None
        self.scheduler_name = None
        self.base_scheduler_config = None
        self.current_second_pass_model = None
        self.current_loras = {}
        self.current_embeddings = {}

        self._clear_component_caches()
        gc.collect()
        if DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def unload_all(self):
        """Unload all loaded model resources and clear caches."""
        with self.lock:
            self._unload_all_locked()

    def attach_second_pass_model(self, model_name):
        """Attach or update second pass model (refiner or img2img)."""
        if self.current_second_pass_model == model_name:
            return

        if self.img2img:
            del self.img2img
            if DEVICE == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.img2img = None
        self.current_second_pass_model = None

        if not model_name or model_name == "None":
            return

        # Ensure base model is offloaded before loading new one
        if self.pipe:
            self.pipe.unet.to("cpu")
            self.pipe.vae.to("cpu")
            if DEVICE == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        model_entry = MODEL_REGISTRY[model_name]
        model_path = model_entry["path"]

        self.img2img = build_second_pass_pipeline(model_path, self.shared)
        self.current_second_pass_model = model_name

    def clear_loras(self):
        """Remove all loaded LoRAs."""
        try:
            self.pipe.unload_lora_weights()
        except Exception:
            pass
        self.current_loras = {}
        self.trigger_phrase = ""

    def _expand_multivector_tokens(self, tokens: list[str]) -> list[str]:
        """Expand multi-vector embedding tokens to include sub-tokens."""
        if not tokens:
            return tokens
        expanded = list(tokens)
        for tok in list(tokens):
            for i in range(1, 32):
                sub = f"{tok}_{i}"
                if sub not in expanded:
                    expanded.append(sub)
        return expanded

    def clear_embeddings(self):
        """Remove all loaded textual-inversion embeddings."""
        base_tokens = []
        for value in self.current_embeddings.values():
            tok = str(value or "").strip()
            if tok and tok not in base_tokens:
                base_tokens.append(tok)

        tokens = self._expand_multivector_tokens(base_tokens)

        try:
            if self.pipe is not None and hasattr(self.pipe, "unload_textual_inversion"):
                self.pipe.unload_textual_inversion(
                    tokens=tokens or None,
                    tokenizer=getattr(self.pipe, "tokenizer", None),
                    text_encoder=getattr(self.pipe, "text_encoder", None),
                )
        except Exception:
            pass

        try:
            if (
                self.pipe is not None
                and hasattr(self.pipe, "unload_textual_inversion")
                and hasattr(self.pipe, "tokenizer_2")
                and hasattr(self.pipe, "text_encoder_2")
            ):
                self.pipe.unload_textual_inversion(
                    tokens=tokens or None,
                    tokenizer=getattr(self.pipe, "tokenizer_2", None),
                    text_encoder=getattr(self.pipe, "text_encoder_2", None),
                )
        except Exception:
            pass
        self.current_embeddings = {}

    def apply_loras(self, loras):
        """Load and configure LoRA adapters."""
        with stage_timer("lora_apply_seconds"):
            desired = {}

            for l in loras:
                name = l["name"]
                reg = LORA_REGISTRY[name]

                weight = float(l.get("weight", reg.get("weight", 1.0)))
                weight = min(weight, 1.25)

                desired[name] = weight

            if desired == self.current_loras:
                return

            self.clear_loras()

            trigger_phrases = []

            for name, weight in desired.items():
                lora = LORA_REGISTRY[name]
                adapter = sanitize_adapter_name(name)
                self.pipe.load_lora_weights(
                    lora["path"],
                    adapter_name=adapter,
                )
                trigger = lora.get("trigger")
                if trigger:
                    trigger_phrases.append(f"({trigger}:{weight})")

            if desired:
                self.pipe.set_adapters(
                    [sanitize_adapter_name(n) for n in desired.keys()],
                    adapter_weights=list(desired.values())
                )

            self.current_loras = desired
            self.trigger_phrase = ", ".join(trigger_phrases)

    def apply_embeddings(self, embeddings):
        """Load textual inversion embeddings for the active pipeline."""
        with stage_timer("embedding_apply_seconds"):
            desired = {}
            model_arch = MODEL_REGISTRY.get(self.key, {}).get("arch")
            for item in embeddings or []:
                if isinstance(item, str):
                    name = item
                    token = None
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("model")
                    token = item.get("token")
                else:
                    continue

                if not name or name not in EMBEDDING_REGISTRY:
                    continue

                reg = EMBEDDING_REGISTRY[name]
                if model_arch and reg.get("arch") and reg.get("arch") != model_arch:
                    continue
                desired[name] = str(token or reg.get("token") or name)

            if desired == self.current_embeddings:
                return

            self.clear_embeddings()
            if not desired:
                return
            if self.pipe is None or not hasattr(self.pipe, "load_textual_inversion"):
                return

            failed: list[str] = []
            for name, token in desired.items():
                reg = EMBEDDING_REGISTRY[name]
                path = Path(reg["path"])
                try:
                    loaded_dual = _load_sdxl_dual_clip_embedding(self.pipe, path, token)
                    if not loaded_dual:
                        load_kwargs = {"token": token} if token else {}
                        self.pipe.load_textual_inversion(
                            str(path),
                            **load_kwargs,
                        )
                except Exception as exc:
                    log.warning("Skipping embedding '%s' — %s", name, exc)
                    failed.append(name)

            self.current_embeddings = {k: v for k, v in desired.items() if k not in failed}

    def apply_identity_adapter(self, adapter_cfg: dict | None) -> dict:
        """Load and apply IP-Adapter FaceID configuration."""
        if not adapter_cfg or str(adapter_cfg.get("enabled", "false")).lower() != "true":
            if hasattr(self.pipe, "unload_ip_adapter"):
                self.pipe.unload_ip_adapter()
            if hasattr(self.pipe, "peft_config") and "faceid_lora" in getattr(self.pipe, "peft_config", {}):
                self.pipe.delete_adapters(["faceid_lora"])
            return {}

        repo = str(adapter_cfg.get("repo", "h94/IP-Adapter-FaceID"))
        weight = str(adapter_cfg.get("adapter_weight", "ip-adapter-faceid-plusv2_sdxl.bin"))
        lora_weight = adapter_cfg.get("lora_weight")
        if lora_weight:
            lora_weight = str(lora_weight)
        adapter_scale = float(adapter_cfg.get("adapter_scale", 0.55))
        lora_scale = float(adapter_cfg.get("lora_scale", 0.60))
        refs = adapter_cfg.get("reference_images", [])

        if not refs:
            return {}

        # 1. Diffusers IP-Adapter loading
        self.pipe.load_ip_adapter(repo, subfolder=None, weight_name=weight)
        self.pipe.set_ip_adapter_scale(adapter_scale)

        if lora_weight:
            self.pipe.load_lora_weights(repo, weight_name=lora_weight, adapter_name="faceid_lora")
            # Must re-set all adapters since load_lora_weights activates only the new one by default
            current_adapters = [sanitize_adapter_name(n) for n in self.current_loras.keys()]
            current_weights = list(self.current_loras.values())
            current_adapters.append("faceid_lora")
            current_weights.append(lora_scale)
            self.pipe.set_adapters(current_adapters, adapter_weights=current_weights)

        # 2. Extract Face Embeddings using InsightFace
        import cv2
        import torch
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align
        
        embedder_name = str(adapter_cfg.get("embedder", "buffalo_l"))
        app = FaceAnalysis(name=embedder_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        face_images = []
        face_embeds = []

        for ref_path in refs:
            img = cv2.imread(str(ref_path))
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            # Select the largest face bounding box
            face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))[-1]

            embed = torch.from_numpy(face.normed_embedding).unsqueeze(0)
            face_embeds.append(embed)

            face_img = face_align.norm_crop(img, landmark=face.kps, image_size=256)
            face_images.append(face_img)

        if not face_images:
            return {}

        # 3. Calculate mean embedding for multiple reference images
        avg_embed = torch.mean(torch.stack(face_embeds), dim=0)

        # 4. For FaceID Plus v2, diffusers expects ip_adapter_image and ip_adapter_embeds
        return {
            "ip_adapter_image": [face_images], # Diffusers sometimes expects nested lists for multiple scale conditions
            "ip_adapter_embeds": [avg_embed],
        }

    def set_active_unet(self, which: str):
        """Swap between base and second pass UNet on GPU."""
        assert which in ("base", "second_pass"), f"Invalid UNet target: {which}"

        if which == "base":
            if self.img2img:
                self.img2img.unet.to("cpu")
                self.img2img.vae.to("cpu")
            self.pipe.unet.to(DEVICE)
            self.pipe.vae.to(DEVICE)

        else:
            assert self.img2img is not None, "Second pass model not loaded"
            self.pipe.unet.to("cpu")
            self.pipe.vae.to("cpu")
            self.img2img.unet.to(DEVICE)
            self.img2img.vae.to(DEVICE)

    def get(self, base_model, second_pass_model=None, loras=None, embeddings=None, scheduler_name="UniPC", cancel_event=None, identity_adapter=None):
        """Get or create pipeline with specified configuration."""
        with self.lock:
            if self.key != base_model:
                from server.state import update_stage, update_progress
                load_total_steps = 7

                def mark_load(stage: str, step: int):
                    self._ensure_not_cancelled(cancel_event)
                    # Loading should stay below the denoising band so runtime percentages feel linear.
                    progress = 0.04 + (min(step, load_total_steps) / load_total_steps) * 0.34
                    update_stage(f"Loading pipeline components... ({step}/{load_total_steps}) - {stage}")
                    update_progress(progress, step=step, total_steps=load_total_steps)

                mark_load("Preparing", 0)
                self._unload_all_locked()
                self._ensure_not_cancelled(cancel_event)

                self.pipe, _, _ = build_pipeline(
                    base_model_name=base_model,
                    second_pass_model_name=None,
                    lora_names=None,
                    load_progress_callback=mark_load,
                )
                self._ensure_not_cancelled(cancel_event)
                
                mark_load("Building img2img helper", 6)
                self.base_img2img = StableDiffusionXLImg2ImgPipeline(
                    **self.pipe.components
                )
                self._ensure_not_cancelled(cancel_event)

                mark_load("Building inpaint helper", 7)
                self.base_inpaint = StableDiffusionXLInpaintPipeline(
                    **self.pipe.components
                )
                self._ensure_not_cancelled(cancel_event)

                self.shared = {
                    "vae": self.pipe.vae,
                    "tokenizer": self.pipe.tokenizer,
                    "tokenizer_2": self.pipe.tokenizer_2,
                    "text_encoder": self.pipe.text_encoder,
                    "text_encoder_2": self.pipe.text_encoder_2,
                    "scheduler": self.pipe.scheduler,
                }

                self.key = base_model
                self.current_second_pass_model = None
                self.current_loras = {}
                self.current_embeddings = {}
                self.trigger_phrase = ""
                self.scheduler_name = None # Reset scheduler tracking
                self.base_scheduler_config = self.pipe.scheduler.config
            
            # Switch scheduler if needed
            if scheduler_name and scheduler_name != self.scheduler_name:
                self._ensure_not_cancelled(cancel_event)
                from server.state import update_stage # Redundant if already imported but safe
                update_stage(f"Switching scheduler to {scheduler_name}")
                
                # Always use the base config to prevent drift
                config_source = self.base_scheduler_config or self.pipe.scheduler.config
                new_scheduler = create_scheduler(scheduler_name, config_source)
                
                self.pipe.scheduler = new_scheduler
                self.base_img2img.scheduler = new_scheduler
                self.base_inpaint.scheduler = new_scheduler
                if self.img2img:
                    self.img2img.scheduler = new_scheduler
                self.scheduler_name = scheduler_name

            from server.state import update_stage, update_progress
            update_stage("Attaching second pass model")
            update_progress(0.15)
            self._ensure_not_cancelled(cancel_event)
            self.attach_second_pass_model(second_pass_model)

            update_stage("Loading LoRAs")
            update_progress(0.25)
            self._ensure_not_cancelled(cancel_event)
            self.apply_loras(loras or [])

            update_stage("Loading embeddings")
            update_progress(0.30)
            self._ensure_not_cancelled(cancel_event)
            self.apply_embeddings(embeddings or [])

            update_stage("Loading identity adapter")
            self._ensure_not_cancelled(cancel_event)
            faceid_kwargs = self.apply_identity_adapter(identity_adapter)

            update_stage("Generating")
            update_progress(0.35)

            # Ensure we start with base model on GPU
            self._ensure_not_cancelled(cancel_event)
            self.set_active_unet("base")

            if DEVICE == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            self.last_used = time.time()

            return self.pipe, self.img2img, self.base_img2img, self.base_inpaint, self.trigger_phrase, faceid_kwargs


pipeline_manager = PipelineManager()


def get_runtime_profile_dict() -> dict:
    """Expose selected runtime profile for diagnostics/health endpoints."""
    return RUNTIME_PROFILE.to_dict()
