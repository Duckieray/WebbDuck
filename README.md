# 🦆 WebbDuck

**WebbDuck** is a user-friendly, fast, and private AI image generator. It runs on your own computer, giving you unlimited images without subscriptions or cloud delays.

Designed for simplicity, it hides the complex math of "nodes" and "tensors" while giving you powerful tools like Inpainting (fixing faces, changing backgrounds) and Image-to-Image transformation.

## Key Features

*   **Simple & Clean**: A straightforward interface. Type a prompt, get an image.
*   **Smart & Efficient**: Automatically manages your computer's memory (VRAM) so you can do other things while finding your next masterpiece.
*   **Modern Web UI**: A fast, zero-build interface with modular architecture. [Read more](ui/README.md).
*   **Powerful Editing**:
    *   **Inpainting**: drawing masks to fix or change specific parts of an image.
    *   **Image-to-Image**: Use an existing image as a guide.
    *   **Two-Pass Generation**: Automatically refine images for sharper details.
    *   **Smart Regeneration**: Easily retry prompts with randomized seeds and seamless settings transfer.
*   **JoyCaption Integration**: (Optional) Use advanced AI to describe your existing images for you.
*   **Private**: Everything runs locally. Your prompts and images never leave your machine.

## Prerequisites

*   **Operating System**: Windows 10/11 or Linux.
*   **Graphics Card (GPU)**: NVIDIA GPU with at least 12GB VRAM recommended.
*   **Python**: Version 3.10 or higher.
*   **Storage**: Enough space for AI models (typically 10GB+).

## Installation

### 1. Download WebbDuck
Open a terminal (Command Prompt or PowerShell on Windows) and run:
```bash
git clone https://github.com/Duckieray/webbduck.git
cd webbduck
```

### 2. Set up the environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Create necessary folders
WebbDuck needs a few places to store your models and images.

**Windows (PowerShell):**
```powershell
mkdir checkpoint\sdxl, lora, outputs, weights
```

**Linux:**
```bash
mkdir -p checkpoint/sdxl lora outputs weights
```

## How to Run

1.  **Activate your environment** (if not already active):
    *   Windows: `.\.venv\Scripts\Activate.ps1`
    *   Linux: `source .venv/bin/activate`

2.  **Start the app**:
    ```bash
    python run.py
    ```

3.  **Open in Browser**:
    Visit [http://localhost:8000](http://localhost:8000)

## Documentation

*   [**Plugins Guide**](docs/PLUGINS.md): How to add the JoyCaption describer.
*   [**Architecture**](docs/architecture.md): How WebbDuck works under the hood.
*   [**Experimental Features**](docs/experimental.md): Try out bleeding-edge features.
*   [**Vision**](docs/vision.md): Our design philosophy.

## License

Apache-2.0