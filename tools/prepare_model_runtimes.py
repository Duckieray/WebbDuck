#!/usr/bin/env python3
"""Create/update WebbDuck's isolated image runtime environments.

This script installs runtime libraries only. It never downloads model weights.
Run it from any Python environment that can create venvs. On NixOS,
``nix develop`` is one convenient entry point; ordinary Python works on Bazzite
and other conventional Linux hosts.

Hardware selection is automatic by default. The preparer detects NVIDIA CUDA,
AMD ROCm, Apple MPS, or CPU and installs the matching PyTorch build. Backend-
specific profile requirement files may add accelerator-only dependencies without
changing the public generation API.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_hardware import HostHardwareProfile, detect_host_hardware


RUNTIMES = {
    "sdxl": ("WEBBDUCK_SDXL_PYTHON", ROOT / "runtime_requirements" / "sdxl.txt"),
    "flux": ("WEBBDUCK_FLUX_PYTHON", ROOT / "runtime_requirements" / "flux.txt"),
    "krea2": ("WEBBDUCK_KREA2_PYTHON", ROOT / "runtime_requirements" / "krea2.txt"),
    "qwen_image": ("WEBBDUCK_QWEN_IMAGE_PYTHON", ROOT / "runtime_requirements" / "qwen_image.txt"),
}


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(shlex.quote(str(part)) for part in cmd))
    if not dry_run:
        subprocess.check_call(cmd)


def python_in_venv(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _profile_requirements(runtime: str, profile: HostHardwareProfile) -> list[Path]:
    """Return optional dependency overlays for the detected hardware profile.

    A backend can opt into accelerator-specific packages simply by adding, for
    example, ``runtime_requirements/krea2.cuda.txt`` or ``krea2.nvidia.txt``.
    The base requirements remain portable and are always installed first.
    """
    candidates = [
        ROOT / "runtime_requirements" / f"{runtime}.{profile.accelerator}.txt",
        ROOT / "runtime_requirements" / f"{runtime}.{profile.vendor}.txt",
    ]
    result: list[Path] = []
    for path in candidates:
        if path.exists() and path not in result:
            result.append(path)
    return result


def _torch_install_command(
    runtime_python: str,
    *,
    torch_version: str,
    torchvision_version: str,
    torch_index: str | None,
) -> list[str]:
    command = [
        runtime_python,
        "-m",
        "pip",
        "install",
        f"torch=={torch_version}",
        f"torchvision=={torchvision_version}",
    ]
    if torch_index:
        command.extend(["--index-url", f"https://download.pytorch.org/whl/{torch_index}"])
    return command


def prepare(
    runtime: str,
    *,
    root: Path,
    base_python: str,
    torch_version: str,
    torchvision_version: str,
    torch_index: str | None,
    profile: HostHardwareProfile,
    dry_run: bool,
) -> tuple[str, Path]:
    env_var, requirements = RUNTIMES[runtime]
    env_root = root / runtime
    python_path = python_in_venv(env_root)
    if not python_path.exists() or dry_run:
        env_root.parent.mkdir(parents=True, exist_ok=True)
        run([base_python, "-m", "venv", str(env_root)], dry_run=dry_run)

    runtime_python = str(python_path)
    run([runtime_python, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], dry_run=dry_run)
    run(
        _torch_install_command(
            runtime_python,
            torch_version=torch_version,
            torchvision_version=torchvision_version,
            torch_index=torch_index,
        ),
        dry_run=dry_run,
    )
    run([runtime_python, "-m", "pip", "install", "-r", str(requirements)], dry_run=dry_run)
    for overlay in _profile_requirements(runtime, profile):
        print(f"  hardware dependency overlay: {overlay.relative_to(ROOT)}")
        run([runtime_python, "-m", "pip", "install", "-r", str(overlay)], dry_run=dry_run)
    return env_var, python_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", choices=["all", *RUNTIMES])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.getenv("WEBBDUCK_RUNTIME_HOME", "~/.local/share/webbduck/runtimes")).expanduser(),
        help="Directory that owns isolated virtual environments.",
    )
    parser.add_argument("--python", default=sys.executable, help="Base Python used to create venvs.")
    parser.add_argument("--torch-version", default="2.12.1")
    parser.add_argument("--torchvision-version", default="0.27.1")
    parser.add_argument(
        "--torch-index",
        default="auto",
        help="PyTorch wheel channel. Default 'auto' selects CUDA/ROCm/CPU from detected hardware.",
    )
    parser.add_argument(
        "--accelerator",
        default="auto",
        choices=["auto", "cuda", "nvidia", "rocm", "amd", "mps", "apple", "cpu"],
        help="Override hardware detection for runtime preparation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = detect_host_hardware(None if args.accelerator == "auto" else args.accelerator)
    torch_index = profile.torch_index if str(args.torch_index).lower() == "auto" else str(args.torch_index)
    if str(torch_index).lower() in {"none", "pypi", ""}:
        torch_index = None

    print("Detected WebbDuck runtime hardware:")
    print(f"  accelerator: {profile.accelerator}")
    print(f"  vendor:      {profile.vendor}")
    print(f"  device:      {profile.name}")
    if profile.total_vram_gb:
        print(f"  VRAM:        {profile.total_vram_gb:.2f} GiB")
    if profile.compute_capability:
        print(
            "  capability:  "
            f"{profile.compute_capability[0]}.{profile.compute_capability[1]}"
        )
    print(f"  torch index: {torch_index or 'PyPI/default'}")

    selected = list(RUNTIMES) if args.runtime == "all" else [args.runtime]
    exports: list[tuple[str, Path]] = []
    for runtime in selected:
        print(f"\n== {runtime} ==")
        exports.append(
            prepare(
                runtime,
                root=args.root.expanduser(),
                base_python=args.python,
                torch_version=args.torch_version,
                torchvision_version=args.torchvision_version,
                torch_index=torch_index,
                profile=profile,
                dry_run=args.dry_run,
            )
        )

    print("\nAdd these to the WebbDuck launch environment:")
    for env_var, python_path in exports:
        print(f"export {env_var}={shlex.quote(str(python_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
