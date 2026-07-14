#!/usr/bin/env python3
"""WebbDuck - SDXL Generation Interface"""

import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent

# Ensure both direct module imports and package imports ("webbduck.*") resolve
# regardless of whether run.py is invoked from repo root or elsewhere.
for path in (PARENT, ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

import argparse
import os

# Disable progress bars to prevent BrokenPipeError in background execution
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# os.environ["TQDM_DISABLE"] = "1"

try:
    RLIMIT_AS_GB = int(os.getenv("WEBBDUCK_RLIMIT_AS_GB", "56"))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit_bytes = RLIMIT_AS_GB * 1024 ** 3
    if hard == resource.RLIM_INFINITY or limit_bytes < hard:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard if hard != resource.RLIM_INFINITY else limit_bytes))
        logging.info("Applied RLIMIT_AS = %d GB", RLIMIT_AS_GB)
    else:
        logging.info("Hard RLIMIT_AS (%d GB) is lower than requested; keeping existing limit", hard // 1024**3)
except Exception as exc:
    logging.warning("Could not set RLIMIT_AS: %s", exc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebbDuck SDXL Server")
    parser.add_argument("--output", type=str, help="Custom output directory for generated images")
    parser.add_argument(
        "--models",
        type=str,
        help=(
            "Model library root directory. WebbDuck will resolve checkpoints/loras/embeddings from this root "
            "(for example: <root>/checkpoint or <root>/checkpoints, <root>/lora, <root>/embeddings)."
        ),
    )
    parser.add_argument(
        "--hf-cache",
        "--hf_cache",
        dest="hf_cache",
        type=str,
        help=(
            "Hugging Face cache root override (for example /mnt/f/.cache/huggingface). "
            "WebbDuck will set HF_HOME and HF_HUB_CACHE accordingly."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="Port for the web server (default: 8010)",
    )
    args = parser.parse_args()

    if args.output:
        out_path = Path(args.output).resolve()
        logging.info(f"Setting output directory to: {out_path}")
        os.environ["WEBBDUCK_OUTPUT_DIR"] = str(out_path)
    if args.models:
        models_path = Path(args.models).expanduser().resolve()
        logging.info(f"Setting models root directory to: {models_path}")
        os.environ["WEBBDUCK_MODELS_DIR"] = str(models_path)
    if args.hf_cache:
        hf_raw = Path(args.hf_cache).expanduser().resolve()
        # Accept either HF_HOME root or explicit hub path.
        if hf_raw.name.lower() == "hub":
            hf_home = hf_raw.parent
            hf_hub = hf_raw
        else:
            hf_home = hf_raw
            hf_hub = hf_raw / "hub"
        logging.info(f"Setting Hugging Face cache root to: {hf_home}")
        logging.info(f"Setting Hugging Face hub cache to: {hf_hub}")
        os.environ["WEBBDUCK_HF_CACHE_DIR"] = str(hf_hub)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HF_HUB_CACHE"] = str(hf_hub)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_hub)

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=max(1, min(65535, int(args.port))),
        reload=False,
    )
