# JoyCaption Plugin

Image captioning plugin for Webbduck using the [JoyCaption Alpha 2](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava) model.

## Requirements

- **VRAM**: ~22GB (automatically managed/offloaded by WebbDuck to fit alongside generation)
- **Dependencies**: Already included in webbduck's requirements (transformers, torch, PIL)

## How It Works

1. When you upload an image and click "🔍 Caption", this plugin generates a description
2. The model downloads automatically from HuggingFace on first use (~15GB download)
3. The caption populates your prompt field, which you can then edit

## Caption Styles

| Style | Best For |
|-------|----------|
| Detailed | Full descriptions for img2img that preserves image content |
| SD Prompt | Stable Diffusion-optimized prompts |
| Short | Quick summaries |
| MidJourney | MidJourney-style prompts |
| Tags | Booru-style tag lists |

## Credits

- Model: [fancyfeast/llama-joycaption-alpha-two-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava)
- Original batch script: [MNeMoNiCuZ/joy-caption-batch](https://github.com/MNeMoNiCuZ/joy-caption-batch)
