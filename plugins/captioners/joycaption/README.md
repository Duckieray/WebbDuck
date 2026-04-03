# JoyCaption Plugin

Image captioning plugin for WebbDuck using the [JoyCaption Alpha 2](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava) model.

## Requirements

- **VRAM**: Large captioning workloads can use substantial GPU memory; WebbDuck unloads generation models before captioning to make room.
- **Dependencies**: Uses dependencies already present in the main WebbDuck environment, including `transformers`, `torch`, and Pillow.

## How It Works

1. When you upload an image and click `Caption`, this plugin generates a description.
2. The model downloads automatically from Hugging Face on first use.
3. The caption populates your prompt field, which you can then edit.

## Caption Styles

| Style | Best For |
|-------|----------|
| Detailed | Full descriptions for img2img that preserves image content |
| SD Prompt | Stable Diffusion-optimized prompts |
| Short | Quick summaries |
| MidJourney | MidJourney-style prompts |
| Booru | Booru-style tag lists |

## Credits

- Model: [fancyfeast/llama-joycaption-alpha-two-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava)
- Original batch script: [MNeMoNiCuZ/joy-caption-batch](https://github.com/MNeMoNiCuZ/joy-caption-batch)
