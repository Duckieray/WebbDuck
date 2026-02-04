"""Real-ESRGAN upscaling integration."""

import torch
from pathlib import Path
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

_UPSAMPLERS = {}


def get_upsampler(scale: int = 2):
    """Load Real-ESRGAN once per scale and reuse it."""
    if scale in _UPSAMPLERS:
        return _UPSAMPLERS[scale]

    model_path = Path(f"weights/RealESRGAN_x{scale}plus.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing ESRGAN weights: {model_path}")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )

    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(model_path),
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=10,
        half=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    _UPSAMPLERS[scale] = upsampler
    return upsampler