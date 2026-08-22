#!/usr/bin/env python3
"""Create/update WebbDuck's isolated image runtime environments.

This script installs runtime libraries only. It never downloads model weights.
Run it from a Python environment that can create venvs (for NixOS, `nix develop`
is the intended entry point).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNTIMES = {
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


def prepare(
    runtime: str,
    *,
    root: Path,
    base_python: str,
    torch_version: str,
    torchvision_version: str,
    torch_index: str,
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
        [
            runtime_python,
            "-m",
            "pip",
            "install",
            f"torch=={torch_version}",
            f"torchvision=={torchvision_version}",
            "--index-url",
            f"https://download.pytorch.org/whl/{torch_index}",
        ],
        dry_run=dry_run,
    )
    run([runtime_python, "-m", "pip", "install", "-r", str(requirements)], dry_run=dry_run)
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
    parser.add_argument("--torch-index", default="cu130", help="PyTorch wheel channel, e.g. cu130 or cu132.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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
                torch_index=args.torch_index,
                dry_run=args.dry_run,
            )
        )

    print("\nAdd these to the WebbDuck launch environment:")
    for env_var, python_path in exports:
        print(f"export {env_var}={shlex.quote(str(python_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
