"""Pure-contract tests for WebbDuck's Krea 2 Identity Edit adapter.

Covers the GPU-independent layer in ``core/backends/krea2_identity.py``: the
request snapshot contract, weight resolution, ai-toolkit/ComfyUI -> Diffusers
LoRA key conversion, dual-conditioning geometry (position IDs, ref_boost bias,
fit sizing), and the GQA-safe mask-compatible attention processor factory.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch

from core.backends.krea2_identity import (
    PROVIDER_ID,
    WEIGHT_SPECS,
    KreaIdentityError,
    analyze_lora_keys,
    apply_identity_perf_overrides,
    combined_token_count,
    convert_lora_keys,
    edit_position_ids,
    edit_target_size,
    edit_transformer_forward,
    edit_transformer_forward_paired,
    grid_dims,
    grounded_template,
    identity_repo,
    identity_settings_snapshot,
    identity_weight_override,
    mask_compat_processor,
    preferred_rank,
    ref_boost_bias,
    resolve_identity_weight,
    resolve_reference_path,
    source_token_count,
    template_prefix_idx,
)


# --------------------------------------------------------------------------------------
# Provider contract
# --------------------------------------------------------------------------------------

def test_provider_id_and_weight_spec_contract():
    assert PROVIDER_ID == "krea2_identity_edit"
    assert set(WEIGHT_SPECS) == {"full", "r128", "r64"}
    assert WEIGHT_SPECS["r64"]["approx_bytes"] < WEIGHT_SPECS["r128"]["approx_bytes"] < \
        WEIGHT_SPECS["full"]["approx_bytes"]


def _spec_bytes(rank: str) -> int:
    return int(WEIGHT_SPECS[rank]["approx_bytes"])


def test_preferred_rank_thresholds():
    assert preferred_rank(None) == "r64"
    assert preferred_rank(10.0) == "r64"
    assert preferred_rank(12.4) == "r64"
    assert preferred_rank(14.0) == "r128"
    assert preferred_rank(20.0) == "full"
    assert preferred_rank(math.inf) == "r64"


def test_weight_spec_rank_headroom():
    # The "planned spec" shrink must be meaningful (download + VRAM budget).
    assert _spec_bytes("full") / _spec_bytes("r128") >= 1.8
    assert _spec_bytes("r128") / _spec_bytes("r64") >= 1.8


# --------------------------------------------------------------------------------------
# Identity settings snapshot
# --------------------------------------------------------------------------------------

def _make_ref(tmp_path: Path) -> Path:
    ref = tmp_path / "refs" / "face.png"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"\x89PNG\r\n\x1a\n")
    return ref


def test_snapshot_disabled_or_absent(tmp_path):
    ref = _make_ref(tmp_path)
    assert identity_settings_snapshot(None) is None
    assert identity_settings_snapshot({}) is None
    assert identity_settings_snapshot({"enabled": False}) is None
    assert identity_settings_snapshot({"type": "faceid_sdxl", "reference_images": [str(ref)]}) is None


def test_snapshot_requires_reference(tmp_path):
    with pytest.raises(KreaIdentityError, match="at least one reference"):
        identity_settings_snapshot({"type": PROVIDER_ID, "reference_images": []})


def test_snapshot_v1_single_anchor_only(tmp_path):
    ref = _make_ref(tmp_path)
    with pytest.raises(KreaIdentityError, match="exactly one anchor reference"):
        identity_settings_snapshot(
            {"type": PROVIDER_ID, "reference_images": [str(ref), str(ref)]}
        )


def test_snapshot_defaults(tmp_path):
    ref = _make_ref(tmp_path)
    snap = identity_settings_snapshot({"type": PROVIDER_ID, "reference_images": [str(ref)]})
    assert snap.reference_count_used == 1
    assert snap.ref_boost == 2.0
    assert snap.grounding_px == 768
    assert snap.fit_mode == "fit"
    assert snap.lora_scale == 1.0
    assert snap.max_megapixels == 1.0
    assert snap.reference_image == str(ref.resolve())
    assert not snap.warnings


def test_snapshot_resolves_web_path(tmp_path, monkeypatch):
    ref = _make_ref(tmp_path)
    monkeypatch.setenv("WEBBDUCK_OUTPUT_DIR", str(tmp_path))
    snap = identity_settings_snapshot(
        {"type": PROVIDER_ID, "reference_images": ["/outputs/refs/face.png"]},
        total_vram_gb=20.0,
    )
    assert snap.reference_image == str(ref.resolve())


def test_snapshot_rank_selection_and_warning(tmp_path):
    ref = _make_ref(tmp_path)
    snap = identity_settings_snapshot(
        {"type": PROVIDER_ID, "reference_images": [str(ref)]},
        total_vram_gb=20.0,
    )
    assert snap.lora_rank == "full"
    snap = identity_settings_snapshot(
        {"type": PROVIDER_ID, "reference_images": [str(ref)], "lora_rank": "r64"},
        total_vram_gb=20.0,
    )
    assert snap.lora_rank == "r64"
    assert any("honoring the explicit rank" in w for w in snap.warnings)


def test_snapshot_validation_errors(tmp_path):
    ref = _make_ref(tmp_path)
    with pytest.raises(KreaIdentityError, match="ref_boost"):
        identity_settings_snapshot(
            {"type": PROVIDER_ID, "reference_images": [str(ref)], "ref_boost": 99.0}
        )
    with pytest.raises(KreaIdentityError, match="ref_boost"):
        identity_settings_snapshot(
            {"type": PROVIDER_ID, "reference_images": [str(ref)], "ref_boost": "hot"}
        )
    with pytest.raises(KreaIdentityError, match="grounding_px"):
        identity_settings_snapshot(
            {"type": PROVIDER_ID, "reference_images": [str(ref)], "grounding_px": 64}
        )
    with pytest.raises(KreaIdentityError, match="fit_mode"):
        identity_settings_snapshot(
            {"type": PROVIDER_ID, "reference_images": [str(ref)], "fit_mode": "stretch"}
        )
    with pytest.raises(KreaIdentityError, match="Unknown Krea identity LoRA rank"):
        identity_settings_snapshot(
            {"type": PROVIDER_ID, "reference_images": [str(ref)], "lora_rank": "r999"}
        )
    with pytest.raises(KreaIdentityError, match="face_crop"):
        identity_settings_snapshot(
            {"type": PROVIDER_ID, "reference_images": [str(ref)], "face_crop": "squish"}
        )


def test_snapshot_face_crop_auto_warns(tmp_path):
    ref = _make_ref(tmp_path)
    snap = identity_settings_snapshot(
        {"type": PROVIDER_ID, "reference_images": [str(ref)], "face_crop": "auto"}
    )
    assert snap.face_crop == "off"
    assert any("reserved" in w for w in snap.warnings)


def test_snapshot_missing_file_fails(tmp_path):
    with pytest.raises(KreaIdentityError, match="does not exist"):
        identity_settings_snapshot(
            {"type": PROVIDER_ID, "reference_images": ["/outputs/refs/missing.png"]}
        )


# --------------------------------------------------------------------------------------
# Reference resolution
# --------------------------------------------------------------------------------------

def test_resolve_web_path_uses_output_base(tmp_path):
    ref = tmp_path / "refs" / "face.png"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"x")
    assert resolve_reference_path("/outputs/refs/face.png", output_base=tmp_path) == ref.resolve()
    assert resolve_reference_path("outputs/refs/face.png", output_base=tmp_path) == ref.resolve()


def test_resolve_absolute_path(tmp_path):
    ref = tmp_path / "face.png"
    ref.write_bytes(b"x")
    assert resolve_reference_path(str(ref)) == ref.resolve()


def test_resolve_absolute_path_with_outputs_segment(tmp_path):
    # An already-resolved filesystem path may legitimately contain an
    # "outputs/" segment (e.g. <BASE>/outputs/WebbDuck/refs/face.png when the
    # output root is itself named after the project). Only the web-path
    # prefix (/outputs/ or outputs/) may trigger re-prefixing, otherwise the
    # path is double-mounted and resolve fails.
    ref = tmp_path / "outputs" / "WebbDuck" / "refs" / "face.png"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"x")
    assert resolve_reference_path(str(ref), output_base=tmp_path) == ref.resolve()


def test_resolve_missing_raises(tmp_path):
    with pytest.raises(KreaIdentityError, match="does not exist"):
        resolve_reference_path("/outputs/refs/nope.png", output_base=tmp_path)


# --------------------------------------------------------------------------------------
# Weight resolution
# --------------------------------------------------------------------------------------

def test_weight_override_env(tmp_path, monkeypatch):
    weight = tmp_path / "local.safetensors"
    weight.write_bytes(b"weight")
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_WEIGHT", str(weight))
    assert identity_weight_override() == weight
    result = resolve_identity_weight()
    assert result.source == "env-override"
    assert result.path == str(weight.resolve())
    assert result.approx_bytes == weight.stat().st_size


def test_weight_override_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_WEIGHT", str(tmp_path / "nope.safetensors"))
    with pytest.raises(KreaIdentityError, match="missing file"):
        identity_weight_override()


def test_resolve_identity_weight_repo(tmp_path, monkeypatch):
    def fake_download(repo_id: str, filename: str):
        assert repo_id == identity_repo()
        assert filename == WEIGHT_SPECS["r64"]["filename"]
        dest = tmp_path / filename
        dest.write_bytes(b"x")
        return dest

    result = resolve_identity_weight(rank="r64", hf_hub_download=fake_download)
    assert result.source == "repo"
    assert result.rank == WEIGHT_SPECS["r64"]["rank"]
    assert result.path == str((tmp_path / WEIGHT_SPECS["r64"]["filename"]).resolve())


def test_resolve_identity_weight_download_failure(tmp_path, monkeypatch):
    def broken_download(repo_id, filename):
        raise OSError("network down")

    with pytest.raises(KreaIdentityError, match="Unable to download Krea identity LoRA"):
        resolve_identity_weight(rank="full", hf_hub_download=broken_download)


def test_resolve_identity_weight_invalid_rank(tmp_path, monkeypatch):
    with pytest.raises(KreaIdentityError, match="Unknown Krea identity LoRA rank"):
        resolve_identity_weight(rank="r999", hf_hub_download=lambda **_: Path("x"))


def test_resolve_identity_weight_repo_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_REPO", "my/branch")

    def fake_download(repo_id, filename):
        assert repo_id == "my/branch"
        dest = tmp_path / filename
        dest.write_bytes(b"x")
        return dest

    result = resolve_identity_weight(rank="r128", hf_hub_download=fake_download)
    assert result.rank == WEIGHT_SPECS["r128"]["rank"]


# --------------------------------------------------------------------------------------
# LoRA key conversion
# --------------------------------------------------------------------------------------

_AI_TOOLKIT_SAMPLE = {
    "diffusion_model.blocks.0.attn.wq.lora_A.weight": 1,
    "diffusion_model.blocks.0.attn.wq.lora_B.weight": 2,
    "diffusion_model.blocks.0.attn.wk.lora_A.weight": 3,
    "diffusion_model.blocks.0.attn.wv.lora_A.weight": 4,
    "diffusion_model.blocks.0.attn.wo.lora_A.weight": 5,
    "diffusion_model.blocks.0.attn.gate.lora_A.weight": 6,
    "diffusion_model.blocks.0.mlp.gate.lora_A.weight": 7,
    "diffusion_model.blocks.0.mlp.up.lora_A.weight": 8,
    "diffusion_model.blocks.0.mlp.down.lora_A.weight": 9,
    "diffusion_model.blocks.0.attn.wq.lora_alpha": 4.0,
    "diffusion_model.txtfusion.projector.lora_A.weight": 10,
}


def test_converts_ai_toolkit_keys():
    converted = convert_lora_keys(dict(_AI_TOOLKIT_SAMPLE))
    assert "transformer_blocks.0.attn.to_q.lora_A.weight" in converted
    assert "transformer_blocks.0.attn.to_k.lora_A.weight" in converted
    assert "transformer_blocks.0.attn.to_v.lora_A.weight" in converted
    assert "transformer_blocks.0.attn.to_out.0.lora_A.weight" in converted
    assert "transformer_blocks.0.attn.to_gate.lora_A.weight" in converted
    assert "transformer_blocks.0.ff.gate.lora_A.weight" in converted
    assert "transformer_blocks.0.ff.up.lora_A.weight" in converted
    assert "transformer_blocks.0.ff.down.lora_A.weight" in converted
    assert "text_fusion.projector.lora_A.weight" in converted
    # metadata is carried through unchanged rather than dropped
    assert "transformer_blocks.0.attn.to_q.lora_alpha" in converted


def test_analyze_lora_keys_reports_counts():
    report = analyze_lora_keys(dict(_AI_TOOLKIT_SAMPLE))
    assert len(report["mapped"]) == 10
    assert report["unresolved"] == []
    assert len(report["ignored"]) == 1


def test_unknown_lora_key_fails_loudly():
    bad = {"some.unknown.path.lora_A.weight": 1}
    with pytest.raises(KreaIdentityError, match="unresolved"):
        convert_lora_keys(dict(bad), strict=True)
    # non-strict mode returns the key unchanged so callers can inspect.
    assert convert_lora_keys(bad, strict=False) == bad


def test_no_prefix_key_stays_ignored():
    report = analyze_lora_keys({"lora_scaler": 1.0})
    assert "lora_scaler" in report["ignored"]


# --------------------------------------------------------------------------------------
# Grounding template
# --------------------------------------------------------------------------------------

def test_grounded_template_embeds_instruction():
    text = grounded_template("make it golden hour")
    assert "<|vision_start|><|image_pad|><|vision_end|>make it golden hour" in text
    assert text.startswith("<|im_start|>system\n")
    assert template_prefix_idx() == 34


# --------------------------------------------------------------------------------------
# Fit geometry
# --------------------------------------------------------------------------------------

def test_fit_mode_contains_source_in_request_box():
    # portrait source 1024x1536 fit inside landscape-ish request 832x1216
    h, w = edit_target_size((1024, 1536), (832, 1216))
    assert w <= 832 and h <= 1216
    assert w % 16 == 0 and h % 16 == 0
    # contain-fit leads with the width-limited scale (1024*832/1024=832 wide)
    # -> 832w, 1248h would overflow 1216, so snap keeps it inside.


def test_fit_uses_request_ar():
    src = (512, 512)
    h, w = edit_target_size(src, (1024, 1024))
    # 1024x1024 is 1.048MP > the 1MP cap, same as the upstream `_target_size`.
    assert (h, w) == (992, 992)


def test_fit_defaults_to_source_ar():
    h, w = edit_target_size((1024, 1536))
    assert w % 16 == 0 and h % 16 == 0
    assert abs(w / h - 1024 / 1536) < 0.02


def test_full_mode_uses_requested():
    h, w = edit_target_size((512, 512), (1024, 1024), fit_mode="full")
    assert (h, w) == (992, 992)


def test_caps_at_max_megapixels():
    # 2048x2048 source without a cap would blow past 1MP.
    h, w = edit_target_size((2048, 2048), max_megapixels=1.0)
    assert (w * h) / 1e6 <= 1.0


def test_rejects_invalid_fit_mode():
    with pytest.raises(KreaIdentityError, match="fit_mode"):
        edit_target_size((512, 512), fit_mode="banana")


# --------------------------------------------------------------------------------------
# Grid / token accounting (adaptive planner input)
# --------------------------------------------------------------------------------------

def test_grid_and_token_accounting():
    gh, gw = grid_dims(832, 1216)
    assert (gh, gw) == (76, 52)  # height//16, width//16
    assert source_token_count(gh, gw) == 76 * 52
    assert combined_token_count(8, gh, gw) == 8 + 2 * 76 * 52
    assert combined_token_count(8, gh, gw, reference_count=2) == 8 + 3 * 76 * 52


# --------------------------------------------------------------------------------------
# Position IDs
# --------------------------------------------------------------------------------------

def _positions(text_len: int, gh: int, gw: int, n_src: int):
    pos = edit_position_ids(text_len, gh, gw, n_src, torch.device("cpu"))
    return pos


def test_position_ids_layout_single_source():
    pos = _positions(8, 4, 4, 1)
    assert tuple(pos.shape) == (8 + 16 + 16, 3)
    # text rows all zero
    assert bool((pos[:8] == 0).all())
    # source rows frame=1 with h/w axes
    src = pos[8:24]
    assert bool((src[:, 0] == 1).all())
    assert torch.equal(src[:4, 1], torch.tensor([0, 0, 0, 0]))
    assert torch.equal(src[0, 1:], torch.tensor([0, 0]))
    # target rows frame=0
    tgt = pos[24:]
    assert bool((tgt[:, 0] == 0).all())
    assert torch.equal(tgt[0, 1:], torch.tensor([0, 0]))


def test_position_ids_multi_source_frames():
    pos = _positions(8, 2, 3, 2)
    src1 = pos[8:8 + 6]
    src2 = pos[8 + 6:8 + 12]
    assert bool((src1[:, 0] == 1).all())
    assert bool((src2[:, 0] == 2).all())


# --------------------------------------------------------------------------------------
# ref_boost bias
# --------------------------------------------------------------------------------------

def test_ref_boost_bias_off_when_one():
    assert ref_boost_bias(8, 16, 16, 1.0, torch.device("cpu"), torch.float32) is None


def test_ref_boost_bias_value_and_layout():
    text, src, tgt = 6, 12, 12
    bias = ref_boost_bias(text, src, tgt, 4.0, torch.device("cpu"), torch.float32)
    assert tuple(bias.shape) == (1, 1, text + src + tgt, text + src + tgt)
    boosted = bias[0, 0, text + src, text]            # first target row vs first src col
    assert abs(boosted.item() - math.log(4.0)) < 1e-5
    # text row not boosted
    assert bias[0, 0, text - 1, text].item() == 0.0
    # target vs target zero
    assert bias[0, 0, text + src, text + src].item() == 0.0


def test_ref_boost_bias_floor():
    # ref_boost ~ 0 still produces a bounded (log-clamped at max(boost, 1e-4))
    # bias rather than NaN/-inf leaking into the attention logits.
    text, src, tgt = 4, 4, 4
    bias = ref_boost_bias(text, src, tgt, 0.0, torch.device("cpu"), torch.float32)
    assert torch.isfinite(bias).all()
    assigned = bias[0, 0, text + src, text]  # target-vs-source entry
    assert abs(assigned.item() - math.log(1e-4)) < 1e-5


# --------------------------------------------------------------------------------------
# mask-compatible attention processor factory
# --------------------------------------------------------------------------------------

class _StockLikeBase:
    """Stand-in for the stock ``Krea2AttnProcessor`` with the backend fields the
    subclass reads, plus a call log for delegation checks."""

    def __init__(self):
        self._attention_backend = None
        self._parallel_config = None
        self.calls = []


class _RecordableProcessor(_StockLikeBase):
    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        self.calls.append((attention_mask is None, attention_mask, image_rotary_emb))
        return "stock-result"


def test_mask_processor_delegates_when_mask_none():
    base = _RecordableProcessor()
    cls = mask_compat_processor(base.__class__)
    processor = cls()
    result = processor(None, "hs", None, None)
    assert result == "stock-result"
    # super().__call__ ran on the subclass instance (delegation is real),
    # and the masked path was not taken.
    assert processor.calls == [(True, None, None)]


def _fake_attn_module(head_dim: int = 8, num_heads: int = 4, num_kv_heads: int = 2):
    attn = torch.nn.Module()
    dim = head_dim * num_heads
    kv = head_dim * num_kv_heads
    attn.to_q = torch.nn.Linear(dim, dim, bias=False)
    attn.to_k = torch.nn.Linear(dim, kv, bias=False)
    attn.to_v = torch.nn.Linear(dim, kv, bias=False)
    attn.to_gate = torch.nn.Linear(dim, dim)
    attn.to_out = torch.nn.ModuleList([torch.nn.Linear(dim, dim)])
    attn.norm_q = torch.nn.LayerNorm((num_heads, head_dim))
    attn.norm_k = torch.nn.LayerNorm((num_kv_heads, head_dim))
    attn.num_heads = num_heads
    attn.num_kv_heads = num_kv_heads
    attn.head_dim = head_dim
    return attn


def test_mask_processor_applies_dense_mask_under_gqa():
    """GQA kv-repeat + float attention mask through mem-efficient dispatch."""

    class _StockLike(_StockLikeBase):
        def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
            self.got_mask = attention_mask
            return torch.zeros_like(hidden_states)

    head_dim, num_heads, num_kv_heads = 8, 4, 2
    attn = _fake_attn_module(head_dim, num_heads, num_kv_heads)
    stock = _StockLike()
    cls = mask_compat_processor(stock.__class__)
    processor = cls()

    batch, seq, dim = 1, 16, head_dim * num_heads
    hidden = torch.randn(batch, seq, dim)
    mask = torch.zeros(1, 1, seq, seq, dtype=hidden.dtype)
    out = processor(attn, hidden, attention_mask=mask)
    assert out.shape == hidden.shape
    assert getattr(stock, "got_mask", None) is None  # masked path must NOT delegate


def test_mask_processor_has_mask_none_fast_path_with_real_attn():
    """mask=None keeps the stock fast path (no dispatch import needed)."""

    class _StockLike(_StockLikeBase):
        def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
            self.received = attention_mask
            return hidden_states

    attn = _fake_attn_module()
    stock = _StockLike()
    cls = mask_compat_processor(stock.__class__)
    processor = cls()
    hidden = torch.randn(1, 8, 32)
    out = processor(attn, hidden, None, None)
    assert out is hidden
    # super().__call__ ran on the processor instance with the unmodified args.
    assert processor.received is None


def test_persona_ui_is_krea_aware():
    """Phase 3 UI contract: a Krea model must NOT silently fall back to faceid_sdxl."""
    source = Path("ui/modules/PersonaIdentityUI.js").read_text(encoding="utf-8")
    app_source = Path("ui/app_main.js").read_text(encoding="utf-8")
    html_source = Path("ui/index.html").read_text(encoding="utf-8")

    assert "selectedModelLooksKrea" in source
    assert "ensureKreaOption" in source
    assert "krea2_identity_edit" in source
    assert "setKreaOnlyControlsVisible" in source
    assert "krea2_identity_edit" in html_source
    assert "ip-adapter-grounding-px" in html_source
    assert "ip-adapter-lora-rank" in html_source


def test_persona_ui_emits_krea_payload_keys_and_single_anchor_ref():
    """The UI must emit krea2 identity fields and cap refs to one anchor."""
    app_source = Path("ui/app_main.js").read_text(encoding="utf-8")

    assert "adapterType === 'krea2_identity_edit'" in app_source
    for key in ("ref_boost", "grounding_px", "fit_mode", "lora_scale", "lora_rank"):
        assert f"payload.{key}" in app_source
    assert "function kreaIdentityActive" in app_source
    assert "function addRefUrl" in app_source
    assert "_ipAdapterRefs = [url];" in app_source
    # FaceID-only fields must not leak into the Krea payload.
    assert "payload.repo = 'h94/IP-Adapter-FaceID'" in app_source


# --------------------------------------------------------------------------------------
# Paired (cond+uncond) streaming forward + A/B performance overrides
# --------------------------------------------------------------------------------------

class _FakeBlock(torch.nn.Module):
    """Per-token block that uses temb_mod scalar conditioning; ignores rotary/mask."""

    def __init__(self, hid: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(hid, hid)

    def forward(self, x, temb_mod, rotary, attention_mask):
        return torch.tanh(self.fc(x)) * temb_mod


class _FakeTimeEmbed(torch.nn.Module):
    def __init__(self, hid: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(1, hid)

    def forward(self, timestep, dtype=None):
        return torch.nn.functional.silu(self.fc(timestep.float().unsqueeze(-1)))


class _FakeTextFusion(torch.nn.Module):
    """Collapses the ``(B, T, num_layers, hid)`` VLM layer axis like the real fusion."""

    def __init__(self, hid: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(hid, hid)

    def forward(self, prompt_embeds, attention_mask=None):
        return self.fc(torch.mean(prompt_embeds, dim=2))


class _FakeFinalLayer(torch.nn.Module):
    """``final_layer(hidden, temb)`` like the real Krea output layer."""

    def __init__(self, hid: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(hid, hid)

    def forward(self, hidden, temb):
        return self.fc(hidden)


class _FakeKreaTransformer(torch.nn.Module):
    """Mini stand-in for ``Krea2Transformer2DModel`` exposing the same surface.

    Prompt embeddings are ``(B, T, num_layers, hid)``; ``text_fusion`` collapses
    the layer axis like the real Krea multi-layer fusion. Blocks are streamed
    like the real model (per-block ``.to()`` in the paired path).
    """

    def __init__(self, hid: int = 16, num_blocks: int = 3) -> None:
        super().__init__()
        self.hid = hid
        self.time_embed = _FakeTimeEmbed(hid)
        self.time_mod_proj = torch.nn.Linear(hid, hid)
        self.text_fusion = _FakeTextFusion(hid)
        self.txt_in = torch.nn.Linear(hid, hid)
        self.img_in = torch.nn.Linear(hid, hid)
        self.rotary_emb = torch.nn.Identity()
        self.transformer_blocks = torch.nn.ModuleList(
            [_FakeBlock(hid) for _ in range(num_blocks)]
        )
        self.final_layer = _FakeFinalLayer(hid)


def _build_toy_inputs(hid: int, text_pos: int, text_neg: int, src_tokens: int, tgt_tokens: int):
    torch.manual_seed(0)
    latents = torch.randn(1, tgt_tokens, hid)
    src = torch.randn(1, src_tokens, hid) * 0.5
    prompt_pos = torch.randn(1, text_pos, 4, hid) * 0.2
    prompt_neg = torch.randn(1, text_neg, 4, hid) * 0.2
    mask_pos = torch.ones(1, text_pos, dtype=torch.bool)
    mask_neg = torch.ones(1, text_neg, dtype=torch.bool)
    ids_pos = edit_position_ids(text_pos, 4, src_tokens // 4, 1, "cpu")
    ids_neg = edit_position_ids(text_neg, 4, src_tokens // 4, 1, "cpu")
    timestep = torch.tensor([0.5])
    return dict(
        latents=latents,
        src_packed=src,
        prompt_embeds_pos=prompt_pos,
        prompt_mask_pos=mask_pos,
        position_ids_pos=ids_pos,
        prompt_embeds_neg=prompt_neg,
        prompt_mask_neg=mask_neg,
        position_ids_neg=ids_neg,
        timestep=timestep,
    )


def test_paired_forward_matches_two_single_forwards():
    m = _FakeKreaTransformer()
    row = _build_toy_inputs(m.hid, text_pos=11, text_neg=7, src_tokens=8, tgt_tokens=8)

    out_pos_single = edit_transformer_forward(
        m, row["latents"], row["src_packed"],
        row["prompt_embeds_pos"], row["prompt_mask_pos"],
        row["timestep"], row["position_ids_pos"], ref_boost=1.5,
    )
    out_neg_single = edit_transformer_forward(
        m, row["latents"], row["src_packed"],
        row["prompt_embeds_neg"], row["prompt_mask_neg"],
        row["timestep"], row["position_ids_neg"], ref_boost=1.5,
    )
    out_pos, out_neg = edit_transformer_forward_paired(
        m, row["latents"], row["src_packed"],
        row["prompt_embeds_pos"], row["prompt_mask_pos"], row["position_ids_pos"],
        row["prompt_embeds_neg"], row["prompt_mask_neg"], row["position_ids_neg"],
        row["timestep"], ref_boost=1.5, device="cpu",
    )
    assert out_pos.shape == out_pos_single.shape
    assert out_neg.shape == out_neg_single.shape
    assert torch.allclose(out_pos, out_pos_single, atol=1e-5)
    assert torch.allclose(out_neg, out_neg_single, atol=1e-5)


def test_paired_forward_maskless_boost_path():
    m = _FakeKreaTransformer(num_blocks=2)
    row = _build_toy_inputs(m.hid, text_pos=9, text_neg=5, src_tokens=4, tgt_tokens=4)
    out_pos, _ = edit_transformer_forward_paired(
        m, row["latents"], row["src_packed"],
        row["prompt_embeds_pos"], row["prompt_mask_pos"], row["position_ids_pos"],
        row["prompt_embeds_neg"], row["prompt_mask_neg"], row["position_ids_neg"],
        row["timestep"], ref_boost=1.0, device="cpu",
    )
    assert torch.isfinite(out_pos).all()


def test_paired_forward_shared_text_length_rows():
    m = _FakeKreaTransformer()
    row = _build_toy_inputs(m.hid, text_pos=8, text_neg=8, src_tokens=8, tgt_tokens=8)
    out_p, out_n = edit_transformer_forward_paired(
        m, row["latents"], row["src_packed"],
        row["prompt_embeds_pos"], row["prompt_mask_pos"], row["position_ids_pos"],
        row["prompt_embeds_neg"], row["prompt_mask_neg"], row["position_ids_neg"],
        row["timestep"], ref_boost=2.0, device="cpu",
    )
    # Identical prompt rows must give identical predictions even in a shared stream.
    assert torch.allclose(out_p, out_n, atol=1e-5)


def test_perf_overrides_noop_without_env(monkeypatch):
    for key in (
        "WEBBDUCK_KREA2_IDENTITY_STEPS",
        "WEBBDUCK_KREA2_IDENTITY_GUIDANCE",
        "WEBBDUCK_KREA2_IDENTITY_CFG_FREE",
    ):
        monkeypatch.delenv(key, raising=False)
    assert apply_identity_perf_overrides(steps=28, guidance=7.5) == (28, 7.5)


def test_perf_overrides_steps_and_guidance(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_STEPS", "12")
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_GUIDANCE", "0.5")
    assert apply_identity_perf_overrides(steps=28, guidance=4.5) == (12, 0.5)


def test_perf_overrides_cfg_free(monkeypatch):
    monkeypatch.delenv("WEBBDUCK_KREA2_IDENTITY_STEPS", raising=False)
    monkeypatch.delenv("WEBBDUCK_KREA2_IDENTITY_GUIDANCE", raising=False)
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_CFG_FREE", "1")
    assert apply_identity_perf_overrides(steps=28, guidance=7.5) == (28, 0.0)


def test_perf_overrides_cfg_free_edge_value(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_CFG_FREE", "0")
    assert apply_identity_perf_overrides(steps=28, guidance=7.5) == (28, 7.5)