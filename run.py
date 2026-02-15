#!/usr/bin/env python3
"""WebbDuck - SDXL Generation Interface"""

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebbDuck SDXL Server")
    parser.add_argument("--output", type=str, help="Custom output directory for generated images")
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

    uvicorn.run(
        "webbduck.server.app:app",
        host="0.0.0.0",
        port=max(1, min(65535, int(args.port))),
        reload=False,
    )
