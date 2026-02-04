# Webbduck Captioning Plugins

This guide explains how to set up and use image captioning plugins with Webbduck.

## Overview

Webbduck supports optional image captioning through plugins. When a captioner is installed, a "Caption" button appears in the img2img settings, allowing you to generate prompts from uploaded images.

## Plugin Location

## Plugin Location
WebbDuck automatically looks for plugins in these locations (in order):
1. Environment variable `WEBBDUCK_PLUGINS_DIR`
2. The `plugins/` directory inside the WebbDuck folder (e.g., `webbduck/plugins/`)
3. `~/.webbduck/plugins/` (User home directory)

On Windows, the home directory is typically `C:\Users\<username>\.webbduck\plugins\`.


## Plugin Structure

Each captioner must have this structure:
```
captioners/
└── joycaption/            # Folder name = captioner name
    └── captioner.py       # Required - must contain generate_caption function
```

### Required Interface

The `captioner.py` file must implement:

```python
def generate_caption(
    image_path: Path,
    prompt: str,
    max_tokens: int = 300,
) -> str:
    """
    Generate a caption for the image.
    
    Args:
        image_path: Path to the image file
        prompt: The captioning prompt/instruction
        max_tokens: Maximum tokens to generate
        
    Returns:
        The generated caption string
    """
    pass
```

## Installing JoyCaption

JoyCaption is an excellent open-source image captioner. To use it:

1. **Create the plugins directory:**
   ```bash
   mkdir -p ~/.webbduck/plugins/captioners/joycaption
   ```

2. **Create `captioner.py`** in that folder (see template below)

3. **Download the model** on first run (automatic via HuggingFace)

### JoyCaption Template

Create `~/.webbduck/plugins/captioners/joycaption/captioner.py`:

```python
"""JoyCaption plugin for Webbduck."""

import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TVF
from transformers import AutoTokenizer, LlavaForConditionalGeneration

MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"

# Lazy load model
_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        print("Loading JoyCaption model...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        _model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype="bfloat16",
            device_map="auto"
        )
        print("JoyCaption model loaded.")
    return _model, _tokenizer


def generate_caption(
    image_path: Path,
    prompt: str,
    max_tokens: int = 300,
) -> str:
    """Generate a caption for the image using JoyCaption."""
    
    model, tokenizer = _load_model()
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((384, 384), Image.LANCZOS)
    pixel_values = TVF.pil_to_tensor(image).unsqueeze(0)
    pixel_values = pixel_values / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
    
    # Get device and dtype from model
    vision_dtype = model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
    vision_device = model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
    language_device = model.language_model.get_input_embeddings().weight.device
    
    pixel_values = pixel_values.to(vision_device, dtype=vision_dtype)
    
    # Build conversation
    convo = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user", "content": prompt},
    ]
    
    convo_string = tokenizer.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )
    
    # Tokenize
    convo_tokens = tokenizer.encode(convo_string, add_special_tokens=False)
    
    # Handle image tokens
    image_token_id = model.config.image_token_index
    input_tokens = []
    for token in convo_tokens:
        if token == image_token_id:
            input_tokens.append(token)
        else:
            input_tokens.append(token)
    
    input_ids = torch.tensor([input_tokens], dtype=torch.long).to(language_device)
    attention_mask = torch.ones_like(input_ids)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.5,
            top_k=10,
            top_p=0.9,
            suppress_tokens=None,
            use_cache=True,
        )
    
    # Decode
    generated_ids = output_ids[0][len(input_ids[0]):]
    caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return caption.strip()
```

## VRAM Requirements

- **JoyCaption Alpha 2**: ~22GB VRAM
- Webbduck will automatically offload generation pipelines before captioning

## Caption Styles

Available styles in the UI:
- **Detailed**: Full descriptive caption
- **SD Prompt**: Optimized for Stable Diffusion
- **Short**: Brief description
- **MidJourney**: MidJourney-style prompt
- **Tags**: Booru-style tags

## Troubleshooting

### Caption button doesn't appear
- Check that the plugin folder exists
- Verify `captioner.py` is present
- Restart the server

### Out of memory errors
- JoyCaption requires ~22GB VRAM
- Close other GPU applications
- Consider using a quantized model

### Model download fails
- Check internet connection
- Set HuggingFace cache directory if needed:
  ```
  HF_HOME=/path/to/cache
  ```
