from __future__ import annotations

from pathlib import Path

from runtime_hardware import HostHardwareProfile, memory_tier
from tools.prepare_model_runtimes import _profile_requirements, _torch_install_command


def test_pytorch_channel_tracks_accelerator_and_nvidia_generation():
    assert HostHardwareProfile("cuda", "nvidia", "Blackwell", compute_capability=(12, 0)).torch_index == "cu130"
    assert HostHardwareProfile("cuda", "nvidia", "Volta", compute_capability=(7, 0)).torch_index == "cu126"
    assert HostHardwareProfile("rocm", "amd", "Radeon").torch_index == "rocm7.2"
    assert HostHardwareProfile("cpu", "cpu", "CPU").torch_index == "cpu"
    assert HostHardwareProfile("mps", "apple", "Apple GPU").torch_index is None


def test_memory_tiers_are_capacity_driven():
    assert memory_tier(6.0) == "tiny"
    assert memory_tier(10.0) == "low"
    assert memory_tier(16.0) == "constrained"
    assert memory_tier(24.0) == "mid"
    assert memory_tier(32.0) == "high"
    assert memory_tier(48.0) == "ultra"


def test_torch_install_command_uses_profile_channel():
    command = _torch_install_command(
        "/runtime/python",
        torch_version="2.12.1",
        torchvision_version="0.27.1",
        torch_index="rocm7.2",
    )
    assert command[:4] == ["/runtime/python", "-m", "pip", "install"]
    assert "torch==2.12.1" in command
    assert "torchvision==0.27.1" in command
    assert command[-2:] == ["--index-url", "https://download.pytorch.org/whl/rocm7.2"]


def test_torch_install_command_allows_platform_default_without_index():
    command = _torch_install_command(
        "/runtime/python",
        torch_version="2.12.1",
        torchvision_version="0.27.1",
        torch_index=None,
    )
    assert "--index-url" not in command


def test_profile_dependency_overlay_is_convention_based(monkeypatch, tmp_path):
    # The helper only installs overlays that exist, so adding a future
    # krea2.cuda.txt / krea2.nvidia.txt is enough to extend the runtime without
    # hard-coding a package list in the preparer.
    import tools.prepare_model_runtimes as preparer

    requirements = tmp_path / "runtime_requirements"
    requirements.mkdir()
    cuda_overlay = requirements / "krea2.cuda.txt"
    vendor_overlay = requirements / "krea2.nvidia.txt"
    cuda_overlay.write_text("example-cuda-package\n", encoding="utf-8")
    vendor_overlay.write_text("example-nvidia-package\n", encoding="utf-8")
    monkeypatch.setattr(preparer, "ROOT", tmp_path)

    profile = HostHardwareProfile("cuda", "nvidia", "Test GPU")
    assert _profile_requirements("krea2", profile) == [cuda_overlay, vendor_overlay]
