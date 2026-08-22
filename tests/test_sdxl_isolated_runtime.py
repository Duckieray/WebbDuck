from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import core.backends.sdxl as sdxl_backend


def _load_prepare_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools" / "prepare_model_runtimes.py"
    spec = importlib.util.spec_from_file_location("prepare_model_runtimes_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_runtime_catalog_includes_sdxl():
    module = _load_prepare_module()
    env_var, requirements = module.RUNTIMES["sdxl"]
    assert env_var == "WEBBDUCK_SDXL_PYTHON"
    assert requirements.name == "sdxl.txt"
    assert requirements.exists()


def test_sdxl_runtime_has_dedicated_default_path_not_host_python():
    source = inspect.getsource(sdxl_backend._runtime_python)
    assert "WEBBDUCK_SDXL_PYTHON" in source
    assert '"sdxl"' in source
    assert "sys.executable" not in source


def test_sdxl_runtime_requirements_preserve_mature_features():
    root = Path(__file__).resolve().parents[1]
    text = (root / "runtime_requirements" / "sdxl.txt").read_text(encoding="utf-8")
    assert "diffusers==0.36.0" in text
    assert "peft==" in text
    assert "insightface==" in text
    assert "onnxruntime-gpu" in text


def test_worker_preserves_seed_zero_and_mature_metadata():
    root = Path(__file__).resolve().parents[1]
    text = (root / "core" / "backends" / "sdxl_worker.py").read_text(encoding="utf-8")
    assert 'settings.get("seed") != 0' in text
    assert '"lora_trigger_phrase"' in text
    assert '"identity_adapter_debug"' in text
    assert '"performance_timing"' in text
    assert '"smart_extend_source_box"' in text


def test_scheduler_module_has_no_top_level_diffusers_import():
    root = Path(__file__).resolve().parents[1]
    path = root / "core" / "schedulers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert node.module != "diffusers"
        if isinstance(node, ast.Import):
            assert all(alias.name != "diffusers" for alias in node.names)
