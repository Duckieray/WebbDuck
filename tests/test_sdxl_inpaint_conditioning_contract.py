from pathlib import Path


def test_smart_extend_conditions_against_the_pipeline_not_faceid_wrapper():
    source = (Path(__file__).resolve().parents[1] / "modes" / "inpaint.py").read_text(
        encoding="utf-8"
    )

    assert "original_base_inpaint = base_inpaint" in source
    assert "active_pipe=base_inpaint" not in source
    assert source.count("active_pipe=original_base_inpaint") == 7
