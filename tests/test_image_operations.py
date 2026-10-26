from PIL import Image

from core.image_operations import prepare_image_operation


def test_prepare_image_operation_inverts_keep_mask_and_resizes():
    source = Image.new("RGB", (8, 8), "white")
    mask = Image.new("L", (8, 8), 0)
    mask.paste(255, (0, 0, 4, 8))

    prepared = prepare_image_operation(
        {
            "input_image": source,
            "mask_image": mask,
            "width": 16,
            "height": 8,
            "inpainting_fill": "keep",
        }
    )

    assert prepared is not None
    assert prepared.image.size == (16, 8)
    assert prepared.mask.getpixel((1, 4)) == 0
    assert prepared.mask.getpixel((14, 4)) == 255


def test_prepare_image_operation_builds_smart_extend_mask():
    prepared = prepare_image_operation(
        {
            "input_image": Image.new("RGB", (16, 16), "blue"),
            "width": 32,
            "height": 16,
            "smart_extend": True,
            "smart_extend_anchor": "left",
        }
    )

    assert prepared is not None
    assert prepared.image.size == (32, 16)
    assert prepared.source_box == {"x": 0, "y": 0, "w": 16, "h": 16}
    assert prepared.mask.getpixel((1, 8)) == 0
    assert prepared.mask.getpixel((31, 8)) == 255
