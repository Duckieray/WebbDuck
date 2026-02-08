"""Model and LoRA registry with auto-discovery."""

from safetensors.torch import safe_open
from pathlib import Path
import json
import threading

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent

HF_CACHE = Path.home() / ".cache/huggingface/hub"
CHECKPOINT_ROOT = ROOT / "checkpoint/sdxl"
LORA_ROOT = ROOT / "lora"
LORA_ROOT = ROOT / "lora"
LORA_FILE = LORA_ROOT / "loras.json"
MODELS_FILE = CHECKPOINT_ROOT / "models.json"

KNOWN_DEFAULTS = {
    "RealVisXL_V5.0": {"steps": 30, "cfg": 6.0},
    "Juggernaut-XI-v11": {"steps": 30, "cfg": 5.0},
    "biglustydonutmixNSFW_v12": {"steps": 30, "cfg": 7.5},
    "stable-diffusion-xl-refiner-1.0": {"steps": 20, "cfg": 5.0},
    "stable-diffusion-xl-base-1.0": {"steps": 30, "cfg": 7.0},
}

_registry_lock = threading.Lock()


def detect_arch(path: Path) -> str | None:
    """Detect model architecture from directory structure."""
    if (path / "text_encoder_2").exists():
        return "sdxl"

    if (path / "model_index.json").exists():
        data = json.loads((path / "model_index.json").read_text())
        cls = data.get("_class_name", "").lower()
        if "xl" in cls:
            return "sdxl"
        if "flux" in cls:
            return "flux"
        return "sd15"

    return None


def detect_lora_arch(lora_path: Path) -> str | None:
    """Detect LoRA architecture from safetensors keys."""
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception:
        return None

    joined = " ".join(keys)

    if "transformer.blocks" in joined or "flux" in joined:
        return "flux"

    if "lora_te2_" in joined:
        return "sdxl"

    if "lora_te_" in joined:
        return "sd15"

    return None


def scan_hf_cache():
    """Scan HuggingFace cache for models and LoRAs."""
    models = {}
    loras = {}

    if not HF_CACHE.exists():
        return models, loras

    for repo in HF_CACHE.glob("models--*"):
        snap_root = repo / "snapshots"
        if not snap_root.exists():
            continue

        for snap in snap_root.iterdir():
            # Diffusers checkpoint
            if (snap / "unet").exists():
                arch = detect_arch(snap)
                if arch:
                    name = repo.name.replace("models--", "").replace("--", "/")
                    models[name] = {
                        "type": "diffusers",
                        "arch": arch,
                        "path": snap,
                        "source": "hf_cache",
                    }

            # LoRAs
            for f in snap.glob("*.safetensors"):
                if f.stat().st_size < 1.5 * 1024**3 and not (f.parent / "unet").exists():
                    arch = detect_lora_arch(f)
                    if arch:
                        loras[f.stem] = {
                            "path": f,
                            "arch": arch,
                            "trigger": None,
                            "weight": 1.0,
                            "description": f"HF cache LoRA ({arch})",
                            "source": "hf_cache",
                        }

    return models, loras


def discover_local_models():
    """Discover locally stored models."""
    models = {}

    if not CHECKPOINT_ROOT.exists():
        return models

    for item in CHECKPOINT_ROOT.iterdir():
        # Single-file checkpoint
        if item.is_file() and item.suffix == ".safetensors":
            models[item.stem] = {
                "type": "single",
                "arch": "sdxl",
                "path": item,
                "source": "local",
            }

        # Diffusers folder
        elif item.is_dir():
            unet = item / "unet"
            if (unet / "config.json").exists():
                models[item.name] = {
                    "type": "diffusers",
                    "arch": detect_arch(item),
                    "path": item,
                    "source": "local",
                }

    return models


def ensure_lora_registry():
    """Create initial LoRA registry if missing."""
    LORA_ROOT.mkdir(exist_ok=True)

    if LORA_FILE.exists():
        return

    registry = {}

    for f in LORA_ROOT.glob("*.safetensors"):
        arch = detect_lora_arch(f)
        if arch:
            registry[f.stem] = {
                "file": f.name,
                "trigger": None,
                "weight": 1.0,
                "description": "",
            }

    if registry:
        LORA_FILE.write_text(json.dumps(registry, indent=2))


def sync_lora_registry_file():
    """Ensure new local .safetensors files are reflected in loras.json."""
    LORA_ROOT.mkdir(exist_ok=True)
    if LORA_FILE.exists():
        try:
            data = json.loads(LORA_FILE.read_text())
        except Exception:
            data = {}
    else:
        data = {}

    changed = False
    for f in sorted(LORA_ROOT.glob("*.safetensors"), key=lambda p: p.name.lower()):
        key = f.stem
        if key in data:
            continue
        arch = detect_lora_arch(f)
        if not arch:
            continue
        data[key] = {
            "file": f.name,
            "trigger": None,
            "weight": 1.0,
            "description": "",
        }
        changed = True

    if changed:
        LORA_FILE.write_text(json.dumps(data, indent=2))


def load_lora_registry():
    """Load local LoRA registry."""
    if not LORA_FILE.exists():
        return {}

    data = json.loads(LORA_FILE.read_text())
    registry = {}

    for name, cfg in data.items():
        file_path = LORA_ROOT / cfg["file"]

        if not file_path.exists():
            raise FileNotFoundError(f"LoRA file not found for '{name}': {file_path}")

        registry[name] = {
            "path": file_path,
            "arch": detect_lora_arch(file_path) or "sdxl",
            "trigger": cfg.get("trigger"),
            "weight": float(cfg.get("weight", 1.0)),
            "description": cfg.get("description", ""),
            "source": "local",
        }

    return registry


# Initialize registries
# Initialize registries
ensure_lora_registry()

def ensure_model_registry():
    """Create initial model registry if missing and sync with disk."""
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    disk_models = discover_local_models()
    hf_models, _ = scan_hf_cache()
    return merge_model_registry(disk_models, hf_models, persist=True)


def merge_model_registry(disk_models: dict, hf_models: dict, persist: bool = True) -> dict:
    """Merge discovered models with persisted defaults."""
    all_discovered = {**disk_models, **hf_models}

    if MODELS_FILE.exists():
        saved_data = json.loads(MODELS_FILE.read_text())
    else:
        saved_data = {}

    final_registry = {}
    for name, info in all_discovered.items():
        merged = dict(info)
        if name in saved_data:
            merged["defaults"] = saved_data[name].get("defaults", {})
        else:
            defaults = {}
            for key, val in KNOWN_DEFAULTS.items():
                if key in name:
                    defaults = val
                    break
            merged["defaults"] = defaults
        final_registry[name] = merged

    if persist:
        to_save = {
            name: {"defaults": info.get("defaults", {})}
            for name, info in final_registry.items()
        }
        existing_save = {}
        if MODELS_FILE.exists():
            try:
                existing_save = json.loads(MODELS_FILE.read_text())
            except Exception:
                existing_save = {}
        if to_save != existing_save:
            MODELS_FILE.write_text(json.dumps(to_save, indent=2))

    return final_registry


def refresh_registries() -> bool:
    """Refresh model/LoRA registries in place; returns True when changed."""
    with _registry_lock:
        ensure_lora_registry()
        sync_lora_registry_file()
        local_models = discover_local_models()
        new_models = merge_model_registry(local_models, _HF_MODELS, persist=True)
        new_loras = {**load_lora_registry(), **_HF_LORAS}

        models_changed = _registry_signature(MODEL_REGISTRY) != _registry_signature(new_models)
        loras_changed = _registry_signature(LORA_REGISTRY) != _registry_signature(new_loras)
        changed = models_changed or loras_changed

        if changed:
            MODEL_REGISTRY.clear()
            MODEL_REGISTRY.update(new_models)
            LORA_REGISTRY.clear()
            LORA_REGISTRY.update(new_loras)

        return changed


def _registry_signature(data: dict):
    """Stable signature for change detection."""
    return tuple(
        sorted((k, repr(v)) for k, v in data.items())
    )

_HF_MODELS, _HF_LORAS = scan_hf_cache()
MODEL_REGISTRY = merge_model_registry(discover_local_models(), _HF_MODELS, persist=True)
sync_lora_registry_file()
LORA_REGISTRY = {**load_lora_registry(), **_HF_LORAS}
