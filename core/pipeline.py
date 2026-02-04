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

from safetensors.torch import load_file
from webbduck.models.registry import MODEL_REGISTRY, LORA_REGISTRY

from webbduck.core.schedulers import create_scheduler

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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.enable_cudnn_sdp(False)
torch.set_float32_matmul_precision("high")

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Component caches
_UNET_CACHE = {}
_TEXT_CACHE = {}
_VAE_CACHE = {}
_SCHEDULER_CACHE = {}
_REFINER_UNET_CACHE = {}

MAX_CACHE_SIZE = 1

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
        del cache[oldest_key]
        gc.collect()


def sanitize_adapter_name(name: str) -> str:
    """Convert LoRA name to valid adapter identifier."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def load_unet_auto(unet_dir) -> UNet2DConditionModel:
    """Load UNet from diffusers directory with automatic shard handling."""
    unet_dir = Path(unet_dir)
    config_path = unet_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Missing UNet config: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    unet = UNet2DConditionModel.from_config(config, torch_dtype=DTYPE)

    safe_index = unet_dir / "diffusion_pytorch_model.safetensors.index.json"
    safe_file = unet_dir / "diffusion_pytorch_model.safetensors"
    bin_file = unet_dir / "diffusion_pytorch_model.bin"

    if safe_index.exists():
        state_dict = {}
        with open(safe_index) as f:
            index = json.load(f)
        for shard in set(index["weight_map"].values()):
            state_dict.update(load_file(unet_dir / shard, device=DEVICE))

    elif safe_file.exists():
        state_dict = load_file(safe_file, device=DEVICE)

    elif bin_file.exists():
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

def get_text_components(base_path: Path):
    """Load and cache text encoders and tokenizers."""
    key = str(base_path.resolve())

    if key in _TEXT_CACHE:
        return _TEXT_CACHE[key]

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

    _TEXT_CACHE[key] = (tokenizer, tokenizer_2, text_encoder, text_encoder_2)
    return _TEXT_CACHE[key]


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
        scheduler = UniPCMultistepScheduler.from_pretrained(
            base_path / "scheduler"
        )
        scheduler.config.sigma_min = 0.029
        scheduler.config.use_karras_sigmas = True
        _SCHEDULER_CACHE[key] = scheduler

    return _SCHEDULER_CACHE[key]


def get_cached_unet(base_path: Path) -> UNet2DConditionModel:
    """Load and cache UNet."""
    key = str(base_path.resolve())

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
):
    """Build SDXL pipeline with optional second pass model and LoRAs."""
    base_entry = MODEL_REGISTRY[base_model_name]
    vae = get_vae()

    # Single-file checkpoint
    if base_entry["type"] == "single":
        pipe = StableDiffusionXLPipeline.from_single_file(
            base_entry["path"],
            torch_dtype=DTYPE,
            use_safetensors=True,
        ).to(DEVICE)

        pipe.vae = vae
        pipe.vae.to(DEVICE, dtype=DTYPE)

        trigger_phrase = ""
        if lora_names:
            trigger_phrase = apply_loras(pipe, lora_names)

        return pipe, None, trigger_phrase

    # Diffusers checkpoint
    base_path: Path = base_entry["path"]

    unet = get_cached_unet(base_path)
    tokenizer, tokenizer_2, text_encoder, text_encoder_2 = get_text_components(base_path)
    scheduler = get_scheduler(base_path)

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
    pipe.enable_attention_slicing("max")
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
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


class PipelineManager:
    """Manages SDXL pipeline lifecycle and model switching."""
    
    def __init__(self):
        self.pipe = None
        self.img2img = None
        self.base_img2img = None
        self.base_inpaint = None
        self.trigger_phrase = ""
        self.key = None
        self.scheduler_name = None
        self.base_scheduler_config = None
        self.last_used = 0
        self.lock = threading.Lock()

        self.current_second_pass_model = None
        self.current_loras = {}

    def attach_second_pass_model(self, model_name):
        """Attach or update second pass model (refiner or img2img)."""
        if self.current_second_pass_model == model_name:
            return

        if self.img2img:
            del self.img2img
            torch.cuda.empty_cache()

        self.img2img = None
        self.current_second_pass_model = None

        if not model_name or model_name == "None":
            return

        # Ensure base model is offloaded before loading new one
        if self.pipe:
            self.pipe.unet.to("cpu")
            self.pipe.vae.to("cpu")
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

    def apply_loras(self, loras):
        """Load and configure LoRA adapters."""
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

    def set_active_unet(self, which: str):
        """Swap between base and second pass UNet on GPU."""
        assert which in ("base", "second_pass"), f"Invalid UNet target: {which}"

        if which == "base":
            if self.img2img:
                self.img2img.unet.to("cpu")
                self.img2img.vae.to("cpu")
            self.pipe.unet.to("cuda")
            self.pipe.vae.to("cuda")

        else:
            assert self.img2img is not None, "Second pass model not loaded"
            self.pipe.unet.to("cpu")
            self.pipe.vae.to("cpu")
            self.img2img.unet.to("cuda")
            self.img2img.vae.to("cuda")

            self.img2img.unet.to("cuda")
            self.img2img.vae.to("cuda")

    def get(self, base_model, second_pass_model=None, loras=None, scheduler_name="UniPC"):
        """Get or create pipeline with specified configuration."""
        with self.lock:
            if self.key != base_model:
                from webbduck.server.state import update_stage, update_progress
                update_stage("Loading base model")
                update_progress(0.05)

                if self.pipe:
                    # Move to CPU first to help allocator
                    try:
                        self.pipe.to("cpu")
                        if self.base_img2img:
                            self.base_img2img.to("cpu")
                    except Exception:
                        pass
                        
                    destroy_pipeline(self.pipe, self.img2img)
                    self.base_img2img = None
                    self.pipe = None
                    self.img2img = None

                    # Force clearing of global caches for heavy components to ensure clean switch
                    global _UNET_CACHE, _REFINER_UNET_CACHE
                    _UNET_CACHE.clear()
                    _REFINER_UNET_CACHE.clear()
                    
                    gc.collect()
                    torch.cuda.empty_cache()

                self.pipe, _, _ = build_pipeline(
                    base_model_name=base_model,
                    second_pass_model_name=None,
                    lora_names=None,
                )
                
                self.base_img2img = StableDiffusionXLImg2ImgPipeline(
                    **self.pipe.components
                )

                self.base_inpaint = StableDiffusionXLInpaintPipeline(
                    **self.pipe.components
                )

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
                self.trigger_phrase = ""
                self.scheduler_name = None # Reset scheduler tracking
                self.base_scheduler_config = self.pipe.scheduler.config
            
            # Switch scheduler if needed
            if scheduler_name and scheduler_name != self.scheduler_name:
                from webbduck.server.state import update_stage # Redundant if already imported but safe
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

            from webbduck.server.state import update_stage, update_progress
            update_stage("Attaching second pass model")
            update_progress(0.15)
            self.attach_second_pass_model(second_pass_model)

            update_stage("Loading LoRAs")
            update_progress(0.25)
            self.apply_loras(loras or [])

            update_stage("Generating")
            update_progress(0.35)

            # Ensure we start with base model on GPU
            self.set_active_unet("base")

            torch.cuda.synchronize()
            self.last_used = time.time()

            return self.pipe, self.img2img, self.base_img2img, self.base_inpaint, self.trigger_phrase


pipeline_manager = PipelineManager()
