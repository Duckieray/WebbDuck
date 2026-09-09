from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from core.backends import krea2_worker_impl as worker_impl
from core.backends.krea2_lora import (
    _install_stacking_scaled_fp8,
    convert_krea2_lora_state,
    inject_lora_trigger,
    is_krea2_lora_entry,
)


def test_krea_lora_trigger_is_plain_qwen_text_and_not_duplicated():
    assert inject_lora_trigger("cinematic portrait", "my_character") == "cinematic portrait, my_character"
    assert inject_lora_trigger("cinematic MY_CHARACTER portrait", "my_character") == "cinematic MY_CHARACTER portrait"


def test_krea_native_peft_lora_keys_become_transformer_relative():
    state = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(2, 4),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(6, 2),
    }
    converted = convert_krea2_lora_state(state)
    assert set(converted) == {
        "transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer_blocks.0.attn.to_q.lora_B.weight",
    }


def test_krea_original_comfy_keys_map_to_diffusers_modules():
    state = {
        "diffusion_model.blocks.3.attn.wq.lora_A.weight": torch.zeros(2, 4),
        "diffusion_model.blocks.3.attn.wq.lora_B.weight": torch.zeros(6, 2),
    }
    converted = convert_krea2_lora_state(state)
    assert "transformer_blocks.3.attn.to_q.lora_A.weight" in converted
    assert "transformer_blocks.3.attn.to_q.lora_B.weight" in converted


def test_krea_ai_toolkit_standalone_modules_follow_upstream_mapping():
    state = {
        "base_model.model.first.lora_A.weight": torch.zeros(2, 4),
        "base_model.model.first.lora_B.weight": torch.zeros(6, 2),
        "base_model.model.tmlp.0.lora_A.weight": torch.zeros(2, 4),
        "base_model.model.tmlp.0.lora_B.weight": torch.zeros(6, 2),
        "base_model.model.txtfusion.projector.lora_A.weight": torch.zeros(2, 4),
        "base_model.model.txtfusion.projector.lora_B.weight": torch.zeros(6, 2),
    }
    converted = convert_krea2_lora_state(state)
    assert "img_in.lora_A.weight" in converted
    assert "img_in.lora_B.weight" in converted
    assert "time_embed.linear_1.lora_A.weight" in converted
    assert "time_embed.linear_1.lora_B.weight" in converted
    assert "text_fusion.projector.lora_A.weight" in converted
    assert "text_fusion.projector.lora_B.weight" in converted


def test_krea_kohya_suffixes_and_alpha_are_normalized():
    state = {
        "diffusion_model.blocks.1.attn.wv.lora_down.weight": torch.zeros(2, 4),
        "diffusion_model.blocks.1.attn.wv.lora_up.weight": torch.zeros(6, 2),
        "diffusion_model.blocks.1.attn.wv.alpha": torch.tensor(1.0),
    }
    converted = convert_krea2_lora_state(state)
    prefix = "transformer_blocks.1.attn.to_v"
    assert prefix + ".lora_A.weight" in converted
    assert prefix + ".lora_B.weight" in converted
    assert prefix + ".lora_alpha" in converted


def test_krea_lora_namespace_is_authoritative_even_for_old_registry_arch(tmp_path: Path):
    path = tmp_path / "loras" / "krea2" / "style.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"placeholder")
    assert is_krea2_lora_entry({"arch": "unknown", "path": str(path)}) is True


def test_native_diffusers_krea_lora_is_recognized_by_6144_hidden_size(tmp_path: Path):
    path = tmp_path / "style.safetensors"
    save_file(
        {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(1, 6144),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(6144, 1),
        },
        str(path),
    )
    # Older generic discovery could label this as flux2 because both families
    # use transformer.transformer_blocks. The Krea 6144 hidden width resolves it.
    assert is_krea2_lora_entry({"arch": "flux2", "path": str(path)}) is True


def test_scaled_fp8_multiple_loras_are_added_not_overwritten():
    _install_stacking_scaled_fp8(worker_impl)

    qweight = torch.tensor(
        [[1.0, -2.0], [0.5, 3.0]],
        dtype=torch.float8_e4m3fn,
    )
    layer = worker_impl.ScaledFP8Linear(
        2,
        2,
        qweight=qweight,
        scale=torch.tensor(0.5, dtype=torch.float32),
    )
    x = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)

    a1 = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)
    b1 = torch.tensor([[2.0], [-1.0]], dtype=torch.bfloat16)
    a2 = torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16)
    b2 = torch.tensor([[1.5], [0.5]], dtype=torch.bfloat16)

    layer.install_lora(a1, b1, torch.tensor(1.0), 0.5)
    layer.install_lora(a2, b2, torch.tensor(1.0), 0.25)

    base = F.linear(x, qweight.to(torch.bfloat16) * torch.tensor(0.5, dtype=torch.bfloat16))
    expected = (
        base
        + 0.5 * F.linear(F.linear(x, a1), b1)
        + 0.25 * F.linear(F.linear(x, a2), b2)
    )
    torch.testing.assert_close(layer(x), expected, rtol=0.02, atol=0.02)
    assert layer.lora_rank == 2


def test_krea_host_passes_resolved_loras_to_worker_and_injects_trigger():
    source = Path("core/backends/krea2_host_impl.py").read_text(encoding="utf-8")
    assert "user_loras = resolve_krea2_loras(settings.get(\"loras\"))" in source
    assert "prompt = inject_lora_trigger(prompt, lora_trigger_phrase(user_loras))" in source
    assert 'payload["loras"] = user_loras' in source


def test_krea_worker_installs_user_loras_before_offload_configuration():
    source = Path("core/backends/krea2_worker.py").read_text(encoding="utf-8")
    assert "install_worker_lora_patch(_impl)" in source
    lora_source = Path("core/backends/krea2_lora.py").read_text(encoding="utf-8")
    assert "base_module._load_pipeline = load_pipeline" in lora_source
    assert "Installing Krea LoRAs" in lora_source
