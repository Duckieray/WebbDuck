import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder="", weight_name="ip-adapter-faceid_sdxl.bin", image_encoder_folder=None)

embed = torch.randn(1, 512, device="cuda", dtype=torch.float16)
combined = torch.cat([torch.zeros_like(embed), embed], dim=0).unsqueeze(1)

pipe(prompt="test", ip_adapter_image_embeds=[combined], num_inference_steps=2)
