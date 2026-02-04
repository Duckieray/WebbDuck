# 🦆 WebbDuck

**WebbDuck** is a lightweight, model-agnostic diffusion UI and workflow engine built on Hugging Face Diffusers.

## Requirements

- Python 3.10+
- CUDA-capable GPU (12GB+ VRAM recommended)
- Linux (tested on Ubuntu 24)

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/webbduck.git
cd webbduck
```

Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create necessary directories:
```bash
mkdir -p checkpoint/sdxl lora outputs weights
```

## Quick Start

Start the server:
```bash
python server.py
```

Open browser to http://localhost:8000

## Project Goals

- Compete with tools like Automatic1111 and ComfyUI on **clarity and efficiency**
- Stay **Diffusers-first**
- Make experimental features explicit, optional, and safe to ignore
- Avoid background compute usage that interferes with generation

## Two-Pass Generation

WebbDuck supports optional two-pass generation:

- **Pass 1**: Base model establishes global composition
- **Pass 2**: Optional refinement using a second model

The second pass may use either:
- A **true SDXL refiner** (special conditioning path), or
- A **generic SDXL img2img model**

### Second Pass Modes

- **Auto** (default): Inspects the model at runtime and selects correct behavior
- **Force Refiner**: Requires a true SDXL refiner checkpoint
- **Force Img2Img**: Treats the second-pass model as standard img2img pipeline

## Features

- ✅ Full control over models (no shipped models or forced filters)
- ✅ Token-aware prompt handling with explicit truncation
- ✅ LoRA support with per-model compatibility checking
- ✅ Real-time VRAM monitoring
- ✅ Optional experimental prompt compression (disabled by default)
- ✅ Built-in upscaling with Real-ESRGAN

## Project Structure

```text
├── core/                  # Generation logic, pipelines, prompt routing
├── modes/                 # Generation strategies (text2img, img2img, two-pass)
├── experimental_prompt/   # Opt-in experimental features
├── webui/                 # Web interface
└── docs/                  # Design documents and philosophy
```

## Status

⚠️ **Early development**  
WebbDuck is under active construction. APIs and workflows may change rapidly.

Experimental features are clearly labeled and disabled by default.

## License

Apache-2.0