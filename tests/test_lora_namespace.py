from pathlib import Path

from models import registry


def test_ltx_subdirectory_classifies_lora_without_tensor_key_guessing(tmp_path, monkeypatch):
    root = tmp_path / "lora"
    path = root / "ltx" / "OmniNFT.safetensors"
    path.parent.mkdir(parents=True)
    # Deliberately not a valid safetensors payload: namespace classification must
    # happen before key inspection so video-only adapters cannot fall back to SDXL.
    path.write_bytes(b"not-safetensors")
    monkeypatch.setattr(registry, "LORA_ROOT", root)

    assert registry.detect_lora_arch(path) == "ltx25"


def test_lora_namespace_only_applies_inside_shared_root(tmp_path, monkeypatch):
    root = tmp_path / "lora"
    outside = tmp_path / "other" / "ltx" / "OmniNFT.safetensors"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"not-safetensors")
    monkeypatch.setattr(registry, "LORA_ROOT", root)

    assert registry._lora_namespace_arch(outside) is None
