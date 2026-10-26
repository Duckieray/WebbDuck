"""Filesystem and browser path conversion coverage for output storage."""

from server import storage


def test_resolve_web_path_preserves_absolute_paths_with_outputs_segment(tmp_path, monkeypatch):
    output_base = tmp_path / "outputs" / "WebbDuck"
    reference = output_base / "refs" / "face.png"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"face")
    monkeypatch.setattr(storage, "BASE", output_base)

    assert storage.resolve_web_path(str(reference)) == reference


def test_resolve_web_path_maps_browser_output_path_to_output_base(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "BASE", tmp_path)

    assert storage.resolve_web_path("/outputs/refs/face.png") == tmp_path / "refs" / "face.png"
    assert storage.resolve_web_path("outputs/refs/face.png") == tmp_path / "refs" / "face.png"
