from __future__ import annotations

import importlib.util
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
    assert '"/runtime-readiness"' in source
    assert '"/generate"' in source
    assert '"/models/unload_all"' in source
