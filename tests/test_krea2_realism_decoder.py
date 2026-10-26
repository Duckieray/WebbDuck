from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from core.backends import krea2_realism_decoder as realism
from core.backends.krea2 import Krea2DiffusersBackend
from models.model_descriptor import ModelDescriptor, capabilities_for_architecture


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "WEBBDUCK_KREA2_DECODER",
        "WEBBDUCK_KREA2_DECODER_OUTPUT",
        "WEBBDUCK_KREA2_REALISM_DECODER",
        "WEBBDUCK_KREA2_REALISM_DECODER_SUBFOLDER",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Request resolution (host)
# ---------------------------------------------------------------------------


def test_stock_decoder_is_default():
    assert realism.resolve_decoder_request({}) is None
    assert realism.resolve_decoder_request({"vae_decoder": ""}) is None
    assert realism.resolve_decoder_request({"vae_decoder": "stock"}) is None


def test_request_selects_realism_with_default_repo():
    payload = realism.resolve_decoder_request({"vae_decoder": "realism"})
    assert payload == {
        "mode": "realism",
        "output": "requested",
        "source": realism.DEFAULT_REPO,
        "subfolder": realism.DEFAULT_SUBFOLDER,
    }


def test_env_opt_in_and_explicit_stock_override(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_KREA2_DECODER", "realism")
    monkeypatch.setenv("WEBBDUCK_KREA2_DECODER_OUTPUT", "2x")
    payload = realism.resolve_decoder_request({})
    assert payload is not None and payload["output"] == "2x"
    # An explicit per-request choice beats the deployment default.
    assert realism.resolve_decoder_request({"vae_decoder": "stock"}) is None
    payload = realism.resolve_decoder_request({"vae_decoder": "realism", "vae_decoder_output": "requested"})
    assert payload["output"] == "requested"


def test_decoder_source_and_subfolder_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("WEBBDUCK_KREA2_REALISM_DECODER", str(tmp_path))
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    payload = realism.resolve_decoder_request({"vae_decoder": "realism"})
    assert payload["source"] == str(tmp_path)
    assert payload["subfolder"] is None

    nested_root = tmp_path / "repo"
    nested = nested_root / realism.DEFAULT_SUBFOLDER
    nested.mkdir(parents=True)
    (nested / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("WEBBDUCK_KREA2_REALISM_DECODER", str(nested_root))
    assert realism.decoder_subfolder() == realism.DEFAULT_SUBFOLDER

    monkeypatch.setenv("WEBBDUCK_KREA2_REALISM_DECODER_SUBFOLDER", ".")
    assert realism.decoder_subfolder() is None
    monkeypatch.setenv("WEBBDUCK_KREA2_REALISM_DECODER_SUBFOLDER", "custom/vae")
    assert realism.decoder_subfolder() == "custom/vae"


def test_fit_realism_output_sizes():
    image = Image.new("RGB", (64, 96))
    resized, used = realism.fit_realism_output(image, requested=(32, 48), output_mode="requested")
    assert resized.size == (32, 48) and used == "lanczos"
    kept, used = realism.fit_realism_output(image, requested=(32, 48), output_mode="2x")
    assert kept.size == (64, 96) and used is None
    same, used = realism.fit_realism_output(image, requested=(64, 96), output_mode="requested")
    assert same is image and used is None


def test_krea2_descriptor_advertises_realism_decoder_only_for_krea():
    assert capabilities_for_architecture("krea2").realism_decoder is True
    assert capabilities_for_architecture("sdxl").realism_decoder is False


# ---------------------------------------------------------------------------
# Worker decode
# ---------------------------------------------------------------------------


def test_pixel_shuffle_first_frame_matches_upscale2x_layout():
    decoded = torch.randn(1, 12, 1, 4, 5)
    image = realism.pixel_shuffle_first_frame(decoded)
    assert image.shape == (1, 3, 8, 10)
    assert torch.equal(image, F.pixel_shuffle(decoded[:, :, 0], 2))

    stock = torch.randn(1, 3, 1, 4, 5)
    assert torch.equal(realism.pixel_shuffle_first_frame(stock), stock[:, :, 0])

    with pytest.raises(ValueError):
        realism.pixel_shuffle_first_frame(torch.randn(1, 7, 1, 4, 4))


class _FakeProcessor:
    @staticmethod
    def postprocess(image, output_type="pil"):
        assert output_type == "pil"
        _b, _c, h, w = image.shape
        return [Image.new("RGB", (w, h))]


class _FakePipe:
    image_processor = _FakeProcessor()

    @staticmethod
    def _unpack_latents(latents, height, width):
        return torch.zeros(1, 16, 1, height // 8, width // 8, dtype=latents.dtype)


def _fake_vae():
    return SimpleNamespace(
        dtype=torch.float32,
        use_tiling=False,
        config=SimpleNamespace(z_dim=16, latents_mean=[0.0] * 16, latents_std=[1.0] * 16),
    )


class _FakeDecoder:
    dtype = torch.float32
    config = SimpleNamespace(z_dim=16)

    def decode(self, z, return_dict=False):
        # Like the real decoder: 8x spatial, 12 channels (2x2 sub-pixels x RGB).
        b, _c, f, h, w = z.shape
        return (torch.zeros(b, 12, f, h * 8, w * 8),)


def _original_decode(calls):
    def _decode(pipe, vae, latents, *, width, height, device):
        calls.append((width, height))
        return [Image.new("RGB", (width, height))]

    return _decode


def test_realism_decode_inactive_uses_original():
    calls = []
    wrapped = realism.make_realism_decode(_original_decode(calls))
    with realism.realism_decode_context({"prompt": "x"}) as note:
        assert note is None
        images = wrapped(_FakePipe(), _fake_vae(), torch.zeros(1, 4, 64), width=64, height=48, device="cpu")
    assert calls == [(64, 48)]
    assert images[0].size == (64, 48)


def test_realism_decode_produces_2x_image(monkeypatch):
    calls = []
    monkeypatch.setattr(realism, "_load_decoder", lambda config, dtype, device: _FakeDecoder())
    wrapped = realism.make_realism_decode(_original_decode(calls))
    request = {"decoder": {"mode": "realism", "output": "requested", "source": "x"}}
    with realism.realism_decode_context(request) as note:
        images = wrapped(_FakePipe(), _fake_vae(), torch.zeros(1, 4, 64), width=64, height=48, device="cpu")
        final_note = dict(note)
    assert calls == []
    assert images[0].size == (128, 96)
    assert final_note["applied"] is True
    assert final_note["decoded"] == [128, 96]
    assert "error" not in final_note
    # The decoder is released when the request finishes.
    assert realism._DECODER is None


def test_realism_decode_falls_back_to_stock_on_load_failure(monkeypatch):
    calls = []

    def _boom(config, dtype, device):
        raise OSError("offline")

    monkeypatch.setattr(realism, "_load_decoder", _boom)
    wrapped = realism.make_realism_decode(_original_decode(calls))
    request = {"decoder": {"mode": "realism", "source": "x"}}
    with realism.realism_decode_context(request) as note:
        first = wrapped(_FakePipe(), _fake_vae(), torch.zeros(1, 4, 64), width=64, height=48, device="cpu")
        second = wrapped(_FakePipe(), _fake_vae(), torch.zeros(1, 4, 64), width=64, height=48, device="cpu")
        final_note = dict(note)
    assert calls == [(64, 48), (64, 48)]
    assert first[0].size == (64, 48) and second[0].size == (64, 48)
    assert final_note["applied"] is False
    assert final_note["fallback"] == "stock-vae"
    assert "OSError: offline" in final_note["error"]


def test_realism_decode_rejects_mismatched_latent_space(monkeypatch):
    class _WrongDecoder(_FakeDecoder):
        config = SimpleNamespace(z_dim=48)

    calls = []
    monkeypatch.setattr(realism, "_load_decoder", lambda config, dtype, device: _WrongDecoder())
    wrapped = realism.make_realism_decode(_original_decode(calls))
    with realism.realism_decode_context({"decoder": {"mode": "realism"}}) as note:
        wrapped(_FakePipe(), _fake_vae(), torch.zeros(1, 4, 64), width=64, height=48, device="cpu")
        final_note = dict(note)
    assert calls == [(64, 48)]
    assert "latent channels" in final_note["error"]


def test_worker_module_installs_realism_patch():
    from core.backends import krea2_worker

    assert getattr(krea2_worker._decode_latents, "_krea_realism_wrapped", False) is True


# ---------------------------------------------------------------------------
# Host end-to-end (fake worker process)
# ---------------------------------------------------------------------------


def _descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        name="krea/Krea-2-Turbo",
        path="/cache/krea2",
        format="diffusers",
        source="hf_cache",
        architecture="krea2",
        backend="krea2_diffusers",
        capabilities=capabilities_for_architecture("krea2"),
        detection={"variant": "turbo"},
    )


def _fake_process_factory(captured, *, image_size, decoder_note):
    class FakeProcess:
        def __init__(self, cmd, **_kwargs):
            request_path = Path(cmd[cmd.index("--request") + 1])
            result_path = Path(cmd[cmd.index("--result") + 1])
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            captured["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
            image_path = output_dir / "krea2_000.png"
            Image.new("RGB", image_size, "white").save(image_path)
            runtime = {
                "adaptive_request": {
                    "requested_width": 64,
                    "requested_height": 64,
                    "effective_width": 48,
                    "effective_height": 48,
                    "resolution_scaled": True,
                    "steps_tuned": False,
                },
            }
            if decoder_note is not None:
                runtime["decoder"] = decoder_note
            result_path.write_text(
                json.dumps({"ok": True, "images": [str(image_path)], "seed": 7, "runtime": runtime}),
                encoding="utf-8",
            )
            self.returncode = 0

        def poll(self):
            return self.returncode

    return FakeProcess


@pytest.mark.parametrize(
    "output_mode, expected_size",
    [("requested", (64, 64)), ("2x", (96, 96))],
)
def test_host_sizes_realism_decode_and_records_metadata(monkeypatch, output_mode, expected_size):
    captured: dict = {}
    note = {"mode": "realism", "decoder": "wan2.1-upscale2x", "applied": True, "load_seconds": 1.5}
    monkeypatch.setattr(
        "core.backends.krea2.subprocess.Popen",
        _fake_process_factory(captured, image_size=(96, 96), decoder_note=note),
    )
    monkeypatch.setenv("WEBBDUCK_KREA2_PYTHON", "/runtime/krea/python")
    settings = {
        "prompt": "portrait in a crowd",
        "width": 64,
        "height": 64,
        "steps": 8,
        "cfg": 0.0,
        "seed": 7,
        "vae_decoder": "realism",
        "vae_decoder_output": output_mode,
    }
    images, _seed = Krea2DiffusersBackend().generate(_descriptor(), settings)

    assert captured["payload"]["decoder"]["mode"] == "realism"
    assert captured["payload"]["decoder"]["output"] == output_mode
    assert images[0].size == expected_size
    assert settings["krea_decoder"]["decoded"] == [96, 96]
    assert settings["krea_decoder"]["to"] == list(expected_size)
    assert (settings["width"], settings["height"]) == expected_size
    assert settings["requested_width"] == 64
    assert settings["performance_timing"]["krea_decoder_load_seconds"] == 1.5
    assert "krea_upscale" not in settings


def test_host_stock_request_sends_no_decoder(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "core.backends.krea2.subprocess.Popen",
        _fake_process_factory(captured, image_size=(48, 48), decoder_note=None),
    )
    monkeypatch.setenv("WEBBDUCK_KREA2_PYTHON", "/runtime/krea/python")
    settings = {"prompt": "duck", "width": 64, "height": 64, "steps": 8, "cfg": 0.0, "seed": 7}
    images, _seed = Krea2DiffusersBackend().generate(_descriptor(), settings)

    assert "decoder" not in captured["payload"]
    assert "krea_decoder" not in settings
    assert images[0].size == (48, 48)


def test_host_records_fallback_note_without_resizing(monkeypatch):
    captured: dict = {}
    note = {"mode": "realism", "applied": False, "fallback": "stock-vae", "error": "OSError: offline"}
    monkeypatch.setattr(
        "core.backends.krea2.subprocess.Popen",
        _fake_process_factory(captured, image_size=(48, 48), decoder_note=note),
    )
    monkeypatch.setenv("WEBBDUCK_KREA2_PYTHON", "/runtime/krea/python")
    settings = {
        "prompt": "duck",
        "width": 64,
        "height": 64,
        "steps": 8,
        "cfg": 0.0,
        "seed": 7,
        "vae_decoder": "realism",
    }
    images, _seed = Krea2DiffusersBackend().generate(_descriptor(), settings)

    assert images[0].size == (48, 48)
    assert settings["krea_decoder"]["error"] == "OSError: offline"
    assert settings["krea_decoder"]["applied"] is False


# ---------------------------------------------------------------------------
# UI contract (static source checks, matching tests/test_capability_driven_ui.py)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]


def test_ui_exposes_decoder_only_for_capable_models():
    html = (_ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    caps_js = (_ROOT / "ui" / "core" / "modelCapabilities.js").read_text(encoding="utf-8")
    app_js = (_ROOT / "ui" / "app_main.js").read_text(encoding="utf-8")
    state_js = (_ROOT / "ui" / "core" / "state.js").read_text(encoding="utf-8")

    assert 'id="vae_decoder"' in html and 'id="vae_decoder_output"' in html
    assert "capabilityEnabled(caps, 'realism_decoder')" in caps_js
    assert "formData.append('vae_decoder'" in app_js
    assert "formData.append('vae_decoder_output'" in app_js
    assert "vaeDecoder:" in state_js and "vaeDecoderOutput:" in state_js


def test_server_forwards_decoder_fields():
    source = (_ROOT / "server" / "app.py").read_text(encoding="utf-8")
    assert source.count('vae_decoder: str = Form("")') == 2
    assert source.count('"vae_decoder": vae_decoder.strip()') == 2
