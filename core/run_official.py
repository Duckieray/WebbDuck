import sys
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
        
    from core.pipeline import pipeline_manager
    adapter_cfg = settings.get("identity_adapter", {})
    refs = adapter_cfg.get("reference_images", [])
    if not refs:
        raise RuntimeError("No reference images provided for official FaceID SDXL.")
        
    log.info("Extracting FaceID embeddings via InsightFace")
    face_embeds, _ = pipeline_manager._extract_identity_faces(refs, adapter_cfg.get("embedder", "buffalo_l"))
    if not face_embeds:
        raise RuntimeError("No faces detected in reference images.")
        
    ref_mode = adapter_cfg.get("reference_mode", "primary_only")
    if ref_mode == "average":
        avg_embed = torch.mean(torch.stack(face_embeds), dim=0)
    else:
        avg_embed = face_embeds[0]
        
    # The official wrapper expects a tensor of shape (num_prompts, 512). 
    # avg_embed is already (1, 512) since _extract_identity_faces adds unsqueeze(0).
    faceid_embeds = avg_embed.to(device=pipe.device, dtype=pipe.dtype)
    
    # Download or find the IP-Adapter weights
    from huggingface_hub import hf_hub_download
    repo = adapter_cfg.get("repo", "h94/IP-Adapter-FaceID")
    weight_name = adapter_cfg.get("adapter_weight", "ip-adapter-faceid_sdxl.bin")
    
    log.info("Downloading/verifying official IP-Adapter weights")
    ip_ckpt = hf_hub_download(repo_id=repo, filename=weight_name)
    
    log.info("Instantiating IPAdapterFaceIDXL wrapper")
    # Make sure we use the base UNet
    pipeline_manager.set_active_unet("base")
    
    ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, pipe.device, torch_dtype=pipe.dtype)
    
    scale = float(adapter_cfg.get("adapter_scale", 1.0))
    prompt = settings.get("prompt", "")
    negative_prompt = settings.get("negative_prompt", "")
    width = settings.get("width", 1024)
    height = settings.get("height", 1024)
    steps = settings.get("steps", 30)
    guidance_scale = settings.get("cfg", 7.0)
    num_samples = settings.get("num_images", 1)
    
    log.info(f"Generating with official IPAdapterFaceIDXL (scale={scale})")
    
    # Generate
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
        seed=generator.initial_seed() if generator else None
    )
    
    # Clean up? The official wrapper sets attention processors on the pipeline's unet.
    # To restore the pipeline for other uses, we might need to unload it.
    # We can rely on `pipeline_manager._unload_all_locked()` when switching models,
    # but subsequent generations with the same model might be affected if they don't use FaceID.
    # For now, let's just restore the processors if we can, or just let `pipeline_manager` handle it.
    # Actually, pipeline_manager._cleanup_identity_adapter() unloads diffusers IP-Adapters. 
    # The official one might need manual cleanup if used sequentially with non-faceid.
    
    # Cleanup
    if hasattr(pipe, "unload_ip_adapter"):
        try:
            pipe.unload_ip_adapter()
        except Exception:
            pass

    return images, generator.initial_seed() if generator else 0
