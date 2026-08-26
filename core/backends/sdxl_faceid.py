import sys
import copy
from pathlib import Path
import logging
import torch

log = logging.getLogger(__name__)

def run_official_faceid_sdxl(settings, pipe, generator, cancel_event):
    """Run generation using the official h94 IPAdapterFaceIDXL wrapper."""
    ip_adapter_path = Path("third_party/IP-Adapter")
    if str(ip_adapter_path) not in sys.path:
        sys.path.append(str(ip_adapter_path))

    try:
        from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL
    except ImportError as exc:
        raise RuntimeError(
            "Could not import official IPAdapterFaceIDXL. Please ensure third_party/IP-Adapter is present."
        ) from exc

    from core.backends.sdxl_pipeline import pipeline_manager
    adapter_cfg = settings.get("identity_adapter", {})
    refs = adapter_cfg.get("reference_images", [])
    if not refs:
        raise RuntimeError("No reference images provided for official FaceID SDXL.")

    # Ensure identity_adapter_debug exists for runtime metadata capture
    debug = settings.setdefault("identity_adapter_debug", {})
    debug["reference_images_requested"] = len(refs)

    log.info("Extracting FaceID embeddings via InsightFace")
    face_embeds, _ = pipeline_manager._extract_identity_faces(refs, adapter_cfg.get("embedder", "buffalo_l"))
    debug["reference_faces_detected"] = len(face_embeds)
    if not face_embeds:
        raise RuntimeError("No faces detected in reference images.")

    ref_mode = adapter_cfg.get("reference_mode", "primary_only")
    if ref_mode == "average":
        avg_embed = torch.mean(torch.stack(face_embeds), dim=0)
    else:
        avg_embed = face_embeds[0]

    # Use unet.dtype as a safe fallback — pipe.dtype may not exist on all diffusers versions
    try:
        pipe_dtype = pipe.dtype
    except AttributeError:
        pipe_dtype = pipe.unet.dtype
    faceid_embeds = avg_embed.to(device=pipe.device, dtype=pipe_dtype)

    # Download or find the IP-Adapter weights
    from huggingface_hub import hf_hub_download
    repo = adapter_cfg.get("repo", "h94/IP-Adapter-FaceID")
    weight_name = adapter_cfg.get("adapter_weight", "ip-adapter-faceid_sdxl.bin")

    log.info("Downloading/verifying official IP-Adapter weights")
    ip_ckpt = hf_hub_download(repo_id=repo, filename=weight_name)

    log.info("Instantiating IPAdapterFaceIDXL wrapper")
    pipeline_manager.set_active_unet("base")

    # Save original UNet state so we can restore it after generation.
    # The official wrapper's __init__ replaces ALL attention processors.
    original_attn_processors = copy.deepcopy(pipe.unet.attn_processors)
    original_encoder_hid_proj = getattr(pipe.unet, "encoder_hid_proj", None)

    ip_model = IPAdapterFaceIDXL(
        pipe, ip_ckpt, pipe.device, torch_dtype=pipe_dtype,
    )

    scale = float(adapter_cfg.get("adapter_scale", 1.0))
    prompt = settings.get("prompt", "")
    negative_prompt = settings.get("negative_prompt", "")
    width = settings.get("width", 1024)
    height = settings.get("height", 1024)
    steps = settings.get("steps", 30)
    guidance_scale = settings.get("cfg", 7.0)
    num_samples = settings.get("num_images", 1)

    debug["adapter_scale"] = scale
    debug["adapter_weight"] = weight_name
    debug["repo"] = repo
    debug["preset_name"] = adapter_cfg.get("preset_name", "") or "custom"

    log.info(f"Generating with official IPAdapterFaceIDXL (scale={scale})")

    images = ip_model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        faceid_embeds=faceid_embeds,
        scale=scale,
        num_samples=num_samples,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=generator.initial_seed() if generator else None,
    )

    # Restore original UNet state so subsequent generations (even without FaceID)
    # start clean. Using set_attn_processor avoids stale PEFT/IP-Adapter state.
    pipe.unet.set_attn_processor(original_attn_processors)
    if original_encoder_hid_proj is not None:
        pipe.unet.encoder_hid_proj = original_encoder_hid_proj
    elif hasattr(pipe.unet, "encoder_hid_proj"):
        pipe.unet.encoder_hid_proj = None

    return images, generator.initial_seed() if generator else 0
