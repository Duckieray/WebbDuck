# modes/base.py

from abc import ABC, abstractmethod

class GenerationMode(ABC):
    """
    A single generation strategy.
    """

    @abstractmethod
    def can_run(self, settings, pipe, img2img, base_img2img, base_inpaint=None) -> bool:
        pass

    @abstractmethod
    def run(self, *, settings, pipe, img2img, base_img2img, base_inpaint, generator, callback=None):
        pass
