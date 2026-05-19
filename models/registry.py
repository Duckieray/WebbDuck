"""Model, LoRA, and embedding registries with auto-discovery."""

from safetensors.torch import safe_open
from pathlib import Path
import json
import os
import threading

# Paths
ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = Path(os.getenv("WEBBDUCK_MODELS_DIR", str(ROOT))).expanduser()


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_hf_cache_root() -> Path:
    explicit = os.getenv("WEBBDUCK_HF_CACHE_DIR")
    if explicit:
        value = Path(explicit).expanduser()
        if value.name.lower() == "hub":
            return value
        return value / "hub"

    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        raw = os.getenv(key)
        if raw:
            value = Path(raw).expanduser()
            if value.name.lower() == "hub":
                return value
            return value / "hub"

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def _resolve_checkpoint_root() -> Path:
    explicit = os.getenv("WEBBDUCK_CHECKPOINT_DIR")
    if explicit:
        return Path(explicit).expanduser()

    models_root = MODELS_ROOT
    candidates = [
        models_root / "checkpoint" / "sdxl",
        models_root / "checkpoints" / "sdxl",
        models_root / "checkpoint",
        models_root / "checkpoints",
        models_root / "sdxl",
    ]
    return _first_existing(candidates)


def _resolve_lora_root() -> Path:
    explicit = os.getenv("WEBBDUCK_LORA_DIR")
    if explicit:
        return Path(explicit).expanduser()
    models_root = MODELS_ROOT
    candidates = [models_root / "lora", models_root / "loras"]
    return _first_existing(candidates)


def _resolve_embedding_root() -> Path:
    explicit = os.getenv("WEBBDUCK_EMBEDDING_DIR")
    if explicit:
        return Path(explicit).expanduser()
    models_root = MODELS_ROOT
    candidates = [models_root / "embeddings", models_root / "embedding"]
    return _first_existing(candidates)


HF_CACHE = _resolve_hf_cache_root()
CHECKPOINT_ROOT = _resolve_checkpoint_root()
LORA_ROOT = _resolve_lora_root()
LORA_FILE = LORA_ROOT / "loras.json"
EMBEDDING_ROOT = _resolve_embedding_root()
EMBEDDING_FILE = EMBEDDING_ROOT / "embeddings.json"
CHECKPOINTS_FILE = CHECKPOINT_ROOT / "checkpoints.json"

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


def detect_embedding_arch(embedding_path: Path) -> str | None:
    """Detect embedding architecture from tensor names.

    Falls back to SDXL when keys are inconclusive.
    """
    if embedding_path.suffix.lower() != ".safetensors":
        # Most modern embeddings in this app are SDXL-focused; allow non-safetensors
        # files to participate with SDXL default unless explicitly changed in JSON.
        return "sdxl"

    try:
        with safe_open(embedding_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception:
        return "sdxl"

    joined = " ".join(keys).lower()

    # Common SD1.x style textual inversion key names.
    if "cond_stage_model" in joined:
        return "sd15"

    # SDXL textual inversion exports frequently include clip_l / clip_g style keys
    # or generic learned-embedding maps that are used with SDXL tokenizers.
    if "clip_g" in joined or "clip_l" in joined or "string_to_param" in joined:
        return "sdxl"

    return "sdxl"


def _is_embedding_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".safetensors", ".pt", ".bin"}


def _is_lora_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".safetensors"


def scan_hf_cache():
    """Scan HuggingFace cache for models and LoRAs."""
    models = {}
    loras = {}

    if not HF_CACHE.exists():
        return models, loras

    for repo in HF_CACHE.glob("models--*"):
        snap_root = repo / "snapshots"
        try:
            has_snapshots = snap_root.exists()
        except OSError:
            continue
        if not has_snapshots:
            continue

        try:
            snapshots = list(snap_root.iterdir())
        except OSError:
            continue

        for snap in snapshots:
            # Diffusers checkpoint
            try:
                is_diffusers = (snap / "unet").exists()
            except OSError:
                is_diffusers = False
            if is_diffusers:
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
            try:
                safetensors_files = list(snap.glob("*.safetensors"))
            except OSError:
                safetensors_files = []

            for f in safetensors_files:
                try:
                    size_ok = f.stat().st_size < 1.5 * 1024**3
                    is_unet_dir = (f.parent / "unet").exists()
                except OSError:
                    continue
                if size_ok and not is_unet_dir:
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

    def _walk_checkpoints(current_dir: Path):
        for item in current_dir.iterdir():
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
                else:
                    _walk_checkpoints(item)

    _walk_checkpoints(CHECKPOINT_ROOT)

    return models


def ensure_lora_registry():
    """Create initial LoRA registry if missing."""
    LORA_ROOT.mkdir(exist_ok=True, parents=True)

    if LORA_FILE.exists():
        return

    registry = {}

    for f in sorted(LORA_ROOT.rglob("*"), key=lambda p: p.name.lower()):
        if not _is_lora_file(f):
            continue
        registry[f.stem] = {
            "file": f.relative_to(LORA_ROOT).as_posix(),
            "trigger": None,
            "weight": 1.0,
            "description": "",
        }

    if registry:
        LORA_FILE.write_text(json.dumps(registry, indent=2))


def ensure_embedding_registry():
    """Create initial embedding registry if missing."""
    EMBEDDING_ROOT.mkdir(exist_ok=True, parents=True)

    if EMBEDDING_FILE.exists():
        return

    registry = {}
    for f in sorted(EMBEDDING_ROOT.rglob("*"), key=lambda p: p.name.lower()):
        if not _is_embedding_file(f):
            continue
        arch = detect_embedding_arch(f)
        if not arch:
            continue
        registry[f.stem] = {
            "file": f.relative_to(EMBEDDING_ROOT).as_posix(),
            "token": f.stem,
            "description": "",
        }

    if registry:
        EMBEDDING_FILE.write_text(json.dumps(registry, indent=2))


def sync_lora_registry_file():
    """Ensure new local .safetensors files are reflected in loras.json."""
    LORA_ROOT.mkdir(exist_ok=True, parents=True)
    if LORA_FILE.exists():
        try:
            data = json.loads(LORA_FILE.read_text())
        except Exception:
            data = {}
    else:
        data = {}

    changed = False
    for f in sorted(LORA_ROOT.rglob("*"), key=lambda p: p.name.lower()):
        if not _is_lora_file(f):
            continue
        key = f.stem
        if key in data:
            continue
        data[key] = {
            "file": f.relative_to(LORA_ROOT).as_posix(),
            "trigger": None,
            "weight": 1.0,
            "description": "",
        }
        changed = True

    if changed:
        LORA_FILE.write_text(json.dumps(data, indent=2))


def sync_embedding_registry_file():
    """Ensure new local embedding files are reflected in embeddings.json."""
    EMBEDDING_ROOT.mkdir(exist_ok=True, parents=True)
    if EMBEDDING_FILE.exists():
        try:
            data = json.loads(EMBEDDING_FILE.read_text())
        except Exception:
            data = {}
    else:
        data = {}

    changed = False
    for f in sorted(EMBEDDING_ROOT.rglob("*"), key=lambda p: p.name.lower()):
        if not _is_embedding_file(f):
            continue
        key = f.stem
        if key in data:
            continue
        arch = detect_embedding_arch(f)
        if not arch:
            continue
        data[key] = {
            "file": f.relative_to(EMBEDDING_ROOT).as_posix(),
            "token": key,
            "description": "",
        }
        changed = True

    if changed:
        EMBEDDING_FILE.write_text(json.dumps(data, indent=2))


def load_lora_registry():
    """Load local LoRA registry."""
    if not LORA_FILE.exists():
        return {}

    try:
        data = json.loads(LORA_FILE.read_text())
    except Exception:
        data = {}
    registry = {}

    for name, cfg in data.items():
        file_value = cfg.get("file")
        if not file_value:
            continue
        file_path = LORA_ROOT / file_value

        if not file_path.exists():
            continue

        registry[name] = {
            "path": file_path,
            "arch": detect_lora_arch(file_path) or "sdxl",
            "trigger": cfg.get("trigger"),
            "weight": float(cfg.get("weight", 1.0)),
            "description": cfg.get("description", ""),
            "source": "local",
        }

    return registry


def load_embedding_registry():
    """Load local embedding registry."""
    if not EMBEDDING_FILE.exists():
        return {}

    data = json.loads(EMBEDDING_FILE.read_text())
    registry = {}

    for name, cfg in data.items():
        file_path = EMBEDDING_ROOT / cfg["file"]
        if not file_path.exists():
            raise FileNotFoundError(f"Embedding file not found for '{name}': {file_path}")

        registry[name] = {
            "path": file_path,
            "arch": cfg.get("arch") or detect_embedding_arch(file_path) or "sdxl",
            "token": cfg.get("token") or name,
            "description": cfg.get("description", ""),
            "source": "local",
        }

    return registry


ensure_lora_registry()
ensure_embedding_registry()

def ensure_model_registry():
    """Create initial model registry if missing and sync with disk."""
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    disk_models = discover_local_models()
    hf_models, _ = scan_hf_cache()
    return merge_model_registry(disk_models, hf_models, persist=True)


def merge_model_registry(disk_models: dict, hf_models: dict, persist: bool = True) -> dict:
    """Merge discovered models with persisted defaults."""
    all_discovered = {**disk_models, **hf_models}

    if CHECKPOINTS_FILE.exists():
        saved_data = json.loads(CHECKPOINTS_FILE.read_text())
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
        if CHECKPOINTS_FILE.exists():
            try:
                existing_save = json.loads(CHECKPOINTS_FILE.read_text())
            except Exception:
                existing_save = {}
        if to_save != existing_save:
            CHECKPOINTS_FILE.write_text(json.dumps(to_save, indent=2))

    return final_registry


def refresh_registries() -> bool:
    """Refresh model/LoRA/embedding registries in place; returns True when changed."""
    with _registry_lock:
        ensure_lora_registry()
        ensure_embedding_registry()
        sync_lora_registry_file()
        sync_embedding_registry_file()
        local_models = discover_local_models()
        new_models = merge_model_registry(local_models, _HF_MODELS, persist=True)
        new_loras = {**load_lora_registry(), **_HF_LORAS}
        new_embeddings = {**load_embedding_registry(), **_HF_EMBEDDINGS}

        models_changed = _registry_signature(MODEL_REGISTRY) != _registry_signature(new_models)
        loras_changed = _registry_signature(LORA_REGISTRY) != _registry_signature(new_loras)
        embeddings_changed = _registry_signature(EMBEDDING_REGISTRY) != _registry_signature(new_embeddings)
        changed = models_changed or loras_changed or embeddings_changed

        if changed:
            MODEL_REGISTRY.clear()
            MODEL_REGISTRY.update(new_models)
            LORA_REGISTRY.clear()
            LORA_REGISTRY.update(new_loras)
            EMBEDDING_REGISTRY.clear()
            EMBEDDING_REGISTRY.update(new_embeddings)

        return changed


def _registry_signature(data: dict):
    """Stable signature for change detection."""
    return tuple(
        sorted((k, repr(v)) for k, v in data.items())
    )

_HF_MODELS, _HF_LORAS = scan_hf_cache()
_HF_EMBEDDINGS = {}
MODEL_REGISTRY = merge_model_registry(discover_local_models(), _HF_MODELS, persist=True)
sync_lora_registry_file()
sync_embedding_registry_file()
LORA_REGISTRY = {**load_lora_registry(), **_HF_LORAS}
EMBEDDING_REGISTRY = {**load_embedding_registry(), **_HF_EMBEDDINGS}
