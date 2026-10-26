"""Reserved seam for Krea 2 masked editing.

Krea2Pipeline exposes text-to-image only. Do not route mask requests here until
a Krea-trained masked-edit pipeline is available; latent compositing alone would
not provide the model-native quality promised by the Studio workflow.
"""

from __future__ import annotations


class Krea2MaskedEditUnavailable(RuntimeError):
    """Raised if a future caller attempts to enable unimplemented Krea masking."""


def prepare_krea2_masked_edit() -> None:
    raise Krea2MaskedEditUnavailable(
        "Krea 2 does not yet provide a trained masked inpaint/outpaint runtime."
    )
