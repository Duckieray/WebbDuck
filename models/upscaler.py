"""Real-ESRGAN upscaling integration."""

import os
import sys
import types
import os
import torch
from pathlib import Path

_UPSAMPLERS = {}

APP_ROOT = Path(__file__).resolve().parent.parent


def _resolve_weights_root() -> Path:
    explicit = str(os.getenv("WEBBDUCK_WEIGHTS_DIR") or "").strip()
    if explicit:
        return Path(explicit).expanduser()

    models_root = str(os.getenv("WEBBDUCK_MODELS_DIR") or "").strip()
    if models_root:
        return (Path(models_root).expanduser() / "weights")

    return APP_ROOT / "weights"


def _ensure_torchvision_compat() -> None:
    """Provide backwards-compatible alias expected by older BasicSR releases.

    Some BasicSR versions import `torchvision.transforms.functional_tensor`,
    which was removed in newer torchvision builds. Create a minimal module
    alias so importing BasicSR does not fail.
    """
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return

    try:
        from torchvision.transforms import functional as functional_mod
    except Exception:
        return

    alias = types.ModuleType("torchvision.transforms.functional_tensor")
    for name in ("rgb_to_grayscale",):
        func = getattr(functional_mod, name, None)
        if callable(func):
            setattr(alias, name, func)

    sys.modules["torchvision.transforms.functional_tensor"] = alias


def _import_upscaler_components():
    _ensure_torchvision_compat()
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Real-ESRGAN dependencies are unavailable. Ensure `basicsr`, "
            "`realesrgan`, and a compatible `torchvision` are installed."
        ) from exc
    return RRDBNet, RealESRGANer


def get_upsampler(scale: int = 2):
    """Load Real-ESRGAN once per scale and reuse it."""
    if scale in _UPSAMPLERS:
        return _UPSAMPLERS[scale]

    RRDBNet, RealESRGANer = _import_upscaler_components()

    weights_root = _resolve_weights_root()
    model_path = weights_root / f"RealESRGAN_x{scale}plus.pth"
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
