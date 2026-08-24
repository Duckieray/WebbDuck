from __future__ import annotations

from server import state as runtime_state


def test_update_vram_uses_device_wide_free_total_memory(monkeypatch):
    gib = 1024**3
    monkeypatch.setattr(runtime_state.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_state.torch.cuda, "mem_get_info", lambda: (4 * gib, 16 * gib))

    runtime_state.update_vram()

    assert runtime_state.state["vram"]["used"] == 12.0
    assert runtime_state.state["vram"]["total"] == 16.0
