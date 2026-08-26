"""Lightweight host facade for the isolated SDXL runtime.

WebbDuck's service process must not import the mature SDXL Diffusers stack. The
standalone SDXL worker sets ``WEBBDUCK_SDXL_WORKER=1`` before importing this
module, which exposes the backend-owned implementation only inside that worker.

A small host facade remains for legacy internal health/idle/tokenization calls
until those endpoints are fully backend-generic. It intentionally owns no model
weights or CUDA pipeline state.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


if str(os.getenv("WEBBDUCK_SDXL_WORKER", "")).strip().lower() in {"1", "true", "yes", "on"}:
    from core.backends.sdxl_pipeline import *  # noqa: F401,F403
else:
    from server.storage import resolve_web_path

    class _IsolatedPipelineManager:
        """Host-side compatibility view with no in-process SDXL state."""

        def __init__(self) -> None:
            self.pipe = None
            self.key = None
            self.current_second_pass_model = None
            self.current_loras: dict = {}
            self.current_embeddings: dict = {}
            self.last_used = time.time()

        def get(self, *args, **kwargs):
            raise RuntimeError(
                "SDXL generation is isolated. Route generation through "
                "SDXLDiffusersBackend instead of core.pipeline.pipeline_manager."
            )

        def unload_all(self, *args, **kwargs) -> None:
            # SDXL workers are currently job-scoped processes, so there is no
            # host-resident pipeline to release here.
            self.pipe = None
            self.key = None
            self.current_second_pass_model = None
            self.current_loras.clear()
            self.current_embeddings.clear()
            self.last_used = time.time()

    pipeline_manager = _IsolatedPipelineManager()

    def get_runtime_profile_dict() -> dict:
        """Return the WebbDuck host runtime profile, not SDXL worker state."""
        from core.runtime import resolve_runtime_profile

        return resolve_runtime_profile().to_dict()

    def get_tokenizer(base_path: Path):
        """Load only SDXL tokenizers in the host process.

        Tokenizers are CPU-only metadata helpers and do not construct Diffusers
        pipelines or load model weights. Generation itself remains isolated.
        """
        from transformers import CLIPTokenizer

        base_path = Path(base_path)
        tokenizer_path = base_path / "tokenizer"
        tokenizer_2_path = base_path / "tokenizer_2"
        if base_path.is_dir() and tokenizer_path.exists() and tokenizer_2_path.exists():
            try:
                return (
                    CLIPTokenizer.from_pretrained(tokenizer_path),
                    CLIPTokenizer.from_pretrained(tokenizer_2_path),
                )
            except (TypeError, OSError, ValueError):
                pass

        return (
            CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
            CLIPTokenizer.from_pretrained(
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                pad_token="!",
            ),
        )
