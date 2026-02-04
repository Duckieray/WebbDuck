"""JoyCaption plugin for Webbduck.

This is a minimal adapter that wraps the JoyCaption model for use
with webbduck's captioning system.
"""

import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TVF
from transformers import AutoTokenizer, LlavaForConditionalGeneration

MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"

# Lazy-loaded model and tokenizer
_model = None
_tokenizer = None


def _load_model():
    """Load the JoyCaption model (lazy, on first use)."""
    global _model, _tokenizer
    
    if _model is None:
        print("[JoyCaption] Loading model... (this may take a moment)")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        _model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("[JoyCaption] Model loaded successfully.")
    
    return _model, _tokenizer


def unload_model():
    """Unload the JoyCaption model to free VRAM."""
    global _model, _tokenizer
    
    if _model is not None:
        print("[JoyCaption] Unloading model...")
        del _model
        _model = None
        _tokenizer = None
        
        # Clear CUDA cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("[JoyCaption] Model unloaded.")


def generate_caption(
    image_path: Path,
    prompt: str,
    max_tokens: int = 150,
) -> str:
    """Generate a caption for the image using JoyCaption.
    
    Args:
        image_path: Path to the image file
        prompt: The captioning prompt/instruction
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated caption string
    """
    model, tokenizer = _load_model()
    
    # Use inference mode for faster execution (no gradient tracking)
    with torch.inference_mode():
        # Load and preprocess image to 384x384
        image = Image.open(image_path).convert("RGB")
        image = image.resize((384, 384), Image.LANCZOS)
        pixel_values = TVF.pil_to_tensor(image).unsqueeze(0)
        
        # Get device and dtype from model
        vision_dtype = model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        vision_device = model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
        language_device = model.language_model.get_input_embeddings().weight.device
        
        # Normalize image - do on GPU for speed
        pixel_values = pixel_values.to(vision_device)
        pixel_values = pixel_values / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(vision_dtype)
        
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
        
        # The model expects image tokens to be expanded
        image_token_id = model.config.image_token_index
        
        # Get image sequence length (729 for 384x384 with 14x14 patches)
        image_seq_length = getattr(model.config, 'image_seq_length', 729)
        
        input_tokens = []
        for token in convo_tokens:
            if token == image_token_id:
                input_tokens.extend([image_token_id] * image_seq_length)
            else:
                input_tokens.append(token)
        
        input_ids = torch.tensor([input_tokens], dtype=torch.long).to(language_device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate caption with optimized settings for single image
        output_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy decoding is faster
            use_cache=True,   # KV cache for faster generation
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # Decode - skip the input tokens
        generated_ids = output_ids[0][len(input_ids[0]):]
        caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return caption.strip()
