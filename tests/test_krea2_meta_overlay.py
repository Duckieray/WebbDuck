from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from core.backends.krea2_worker import overlay_single_file_transformer


def _quant_tensor(payload: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(payload).encode("utf-8")), dtype=torch.uint8)


class _MetaTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img_in = torch.nn.Linear(2, 2, bias=False, device="meta")


def test_overlay_materializes_meta_parameter_directly_from_checkpoint(tmp_path):
    path = tmp_path / "krea_fp8.safetensors"
    save_file(
        {
            "first.weight": torch.full((2, 2), 2.0, dtype=torch.bfloat16),
            "first.weight_scale": torch.tensor(1.5, dtype=torch.float32),
            "first.comfy_quant": _quant_tensor({"format": "float8_e4m3fn"}),
        },
        str(path),
    )
    transformer = _MetaTransformer()
    assert transformer.img_in.weight.is_meta

    result = overlay_single_file_transformer(
        transformer,
        path,
        target_dtype=torch.bfloat16,
    )

    assert result["mapped_tensors"] == 1
    assert transformer.img_in.weight.device.type == "cpu"
    assert transformer.img_in.weight.dtype == torch.bfloat16
    assert torch.equal(
        transformer.img_in.weight.detach(),
        torch.full((2, 2), 3.0, dtype=torch.bfloat16),
    )
