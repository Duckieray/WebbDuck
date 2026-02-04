#!/usr/bin/env python3
"""WebbDuck - SDXL Generation Interface"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebbDuck SDXL Server")
    parser.add_argument("--output",type=str, help="Custom output directory for generated images")
    args = parser.parse_args()

    if args.output:
        out_path = Path(args.output).resolve()
        logging.info(f"Setting output directory to: {out_path}")
        os.environ["WEBBDUCK_OUTPUT_DIR"] = str(out_path)

    uvicorn.run(
        "webbduck.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
