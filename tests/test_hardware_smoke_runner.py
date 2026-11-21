from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "webbduck_hardware_smoke",
    ROOT / "tools" / "run_hardware_smoke.py",
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_model_matching_uses_public_names_only():
    items = [
        {"name": "black-forest-labs/FLUX.2-klein-4B", "ready": True},
        {"name": "krea/Krea-2-Turbo", "ready": True},
    ]
    assert smoke._find_model(items, None, ("flux.2", "klein"))["name"].endswith("klein-4B")
    assert smoke._find_model(items, "krea/Krea-2-Turbo", ("ignored",))["name"] == "krea/Krea-2-Turbo"


def test_krea_cfg_zero_survives_request_construction():
    fields = smoke._base_fields(
        "krea/Krea-2-Turbo",
        width=1024,
        height=1024,
        steps=8,
        cfg=0.0,
        seed=0,
    )
    assert fields["cfg"] == 0.0
    assert fields["seed"] == 0
    assert fields["num_images"] == 1


def test_output_refs_accept_direct_and_nested_generation_results():
    assert smoke._output_refs({"images": ["/outputs/a.png"]}) == ["/outputs/a.png"]
    assert smoke._output_refs({"result": {"image": "/outputs/b.png"}}) == ["/outputs/b.png"]


def test_runner_requires_explicit_execute_for_generation():
    source = (ROOT / "tools" / "run_hardware_smoke.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--execute", action="store_true"' in source
    assert "if not args.execute:" in source
    assert 'f"{base_url}/runtime-readiness"' in source
    assert '"/generate"' in source
    assert '"/models/unload_all"' in source


def test_runner_readiness_timeout_is_configurable():
    source = (ROOT / "tools" / "run_hardware_smoke.py").read_text(encoding="utf-8")
    assert '"--readiness-timeout"' in source
    assert "timeout=args.readiness_timeout" in source


def _smoke_args(**overrides):
    import argparse

    values = {
        "sdxl_model": None,
        "flux_model": None,
        "krea_model": None,
        "dark_beast_model": None,
        "qwen_model": None,
        "moss_model": None,
        "lora": None,
        "refiner": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_moss_image_row_uses_model_defaults_and_is_optional():
    item = {
        "name": "moss/Dirty-MossV2",
        "ready": True,
        "defaults": {"width": 1024, "height": 1024, "steps": 50, "cfg": 3.0},
    }
    rows = smoke._rows(_smoke_args(), [item], Path("/tmp/source.png"), Path("/tmp/mask.png"))
    moss_rows = [row for row in rows if row["row"] == "moss-image-t2i"]
    assert len(moss_rows) == 1
    assert moss_rows[0]["model"]["name"] == "moss/Dirty-MossV2"
    assert moss_rows[0]["fields"]["steps"] == 50
    assert moss_rows[0]["fields"]["cfg"] == 3.0
    assert moss_rows[0]["optional"] is False

    missing = smoke._rows(_smoke_args(), [], Path("/tmp/source.png"), Path("/tmp/mask.png"))
    missing_rows = [row for row in missing if row["row"] == "moss-image-t2i"]
    assert missing_rows[0]["model"] is None
    assert missing_rows[0]["optional"] is True


def test_moss_lora_row_appears_only_when_named():
    item = {
        "name": "moss/Dirty-MossV2",
        "ready": True,
        "defaults": {"width": 1024, "height": 1024, "steps": 50, "cfg": 3.0},
    }
    rows = smoke._rows(
        _smoke_args(moss_lora="charref_v2"),
        [item],
        Path("/tmp/source.png"),
        Path("/tmp/mask.png"),
    )
    lora_rows = [row for row in rows if row["row"] == "moss-image-lora"]
    assert len(lora_rows) == 1
    assert lora_rows[0]["model"]["name"] == "moss/Dirty-MossV2"
    assert json.loads(lora_rows[0]["fields"]["loras"]) == [
        {"name": "charref_v2", "weight": 1.0}
    ]

    without = smoke._rows(
        _smoke_args(), [item], Path("/tmp/source.png"), Path("/tmp/mask.png")
    )
    assert [row for row in without if row["row"] == "moss-image-lora"] == []
