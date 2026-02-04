"""Image captioning module for generating prompts from images.

This module provides optional image captioning functionality that can
dynamically generate prompts from uploaded images. Captioners are loaded
as plugins from the user's plugins directory.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Optional, Callable
import torch

from webbduck.core.captioning_config import (
    list_available_captioners,
    get_captioner_module_path,
)


logger = logging.getLogger(__name__)


# Prompt templates for different caption styles
CAPTION_PROMPTS = {
    "detailed": "Please provide a detailed description of the image.",
    "short": "Write a short description of the image.",
    "sd_prompt": "Write a stable diffusion prompt for this image.",
    "midjourney": "Write a MidJourney prompt for this image.",
    "booru": "Write a list of Booru-like tags for this image.",
}


class CaptionerManager:
    """Manages captioner plugin loading and execution."""
    
    def __init__(self):
        self._loaded_captioners: dict[str, Callable] = {}
        self._loaded_modules: dict[str, object] = {}  # Store modules for unloading
        self._current_captioner: Optional[str] = None
        self._model = None
        self._tokenizer = None
    
    def get_available(self) -> list[str]:
        """Get list of available captioner plugins."""
        return list_available_captioners()
    
    def is_available(self) -> bool:
        """Check if any captioner is available."""
        return len(self.get_available()) > 0
    
    def _load_captioner(self, name: str) -> Callable:
        """Load a captioner plugin module."""
        if name in self._loaded_captioners:
            return self._loaded_captioners[name]
        
        module_path = get_captioner_module_path(name)
        if not module_path:
            raise ValueError(f"Captioner '{name}' not found")
        
        spec = importlib.util.spec_from_file_location(f"captioner_{name}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load captioner module: {module_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, "generate_caption"):
            raise ImportError(
                f"Captioner '{name}' does not have a generate_caption function"
            )
        
        self._loaded_captioners[name] = module.generate_caption
        self._loaded_modules[name] = module  # Store for unloading
        return module.generate_caption
    
    def unload_all(self):
        """Unload all captioner models to free VRAM."""
        import gc
        
        for name, module in self._loaded_modules.items():
            if hasattr(module, 'unload_model'):
                try:
                    logger.info(f"Unloading captioner: {name}")
                    module.unload_model()
                except Exception as e:
                    logger.warning(f"Failed to unload captioner {name}: {e}")
        
        # Clear references
        self._loaded_captioners.clear()
        self._loaded_modules.clear()
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
    
    def generate_caption(
        self,
        image_path: Path,
        style: str = "detailed",
        captioner_name: Optional[str] = None,
        max_tokens: int = 300,
    ) -> str:
        """Generate a caption for the given image.
        
        Args:
            image_path: Path to the image file
            style: Caption style (detailed, short, sd_prompt, midjourney, booru)
            captioner_name: Name of captioner to use (auto-selects if None)
            max_tokens: Maximum tokens in generated caption
            
        Returns:
            Generated caption string
            
        Raises:
            ValueError: If no captioner is available
        """
        available = self.get_available()
        if not available:
            raise ValueError(
                "No captioner plugins available. "
                "Please install a captioner in ~/.webbduck/plugins/captioners/"
            )
        
        # Use first available if not specified
        if captioner_name is None:
            captioner_name = available[0]
        
        if captioner_name not in available:
            raise ValueError(f"Captioner '{captioner_name}' is not available")
        
        # Get prompt for style
        prompt = CAPTION_PROMPTS.get(style, CAPTION_PROMPTS["detailed"])
        
        # Load and run captioner
        caption_fn = self._load_captioner(captioner_name)
        caption = caption_fn(
            image_path=image_path,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        return caption.strip()


# Global captioner manager instance
captioner_manager = CaptionerManager()


def generate_caption(
    image_path: Path,
    style: str = "detailed",
    captioner_name: Optional[str] = None,
    max_tokens: int = 300,
) -> str:
    """Convenience function to generate a caption.
    
    See CaptionerManager.generate_caption for full documentation.
    """
    return captioner_manager.generate_caption(
        image_path=image_path,
        style=style, 
        captioner_name=captioner_name,
        max_tokens=max_tokens,
    )


def is_captioning_available() -> bool:
    """Check if captioning is available."""
    return captioner_manager.is_available()


def list_captioners() -> list[str]:
    """List available captioner plugins."""
    return captioner_manager.get_available()


def get_caption_styles() -> dict[str, str]:
    """Get available caption styles and their prompts."""
    return CAPTION_PROMPTS.copy()


def unload_captioners():
    """Unload all captioner models to free VRAM.
    
    Call this before loading generation pipelines if a captioner
    was used during the session.
    """
    captioner_manager.unload_all()
