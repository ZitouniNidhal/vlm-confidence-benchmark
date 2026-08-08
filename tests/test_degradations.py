from PIL import Image

import pytest

from degradations.pipeline import apply_degradations


def test_apply_degradations_executes_named_operations_in_order():
    image = Image.new("RGB", (64, 64), color=(255, 0, 0))

    result = apply_degradations(
        image,
        operations=[("blur", {"radius": 1.0}), ("rotation", {"angle": 15.0})],
    )

    assert isinstance(result, Image.Image)
    assert result.size[0] > 0 and result.size[1] > 0


def test_apply_degradations_rejects_unknown_operations():
    image = Image.new("RGB", (32, 32), color=(0, 255, 0))

    with pytest.raises(ValueError, match="Unknown degradation"):
        apply_degradations(image, operations=[("mystery", {})])


def test_apply_degradations_supports_severity_presets():
    image = Image.new("RGB", (64, 64), color=(255, 255, 255))

    result = apply_degradations(
        image,
        operations=[
            ("blur", "low"),
            ("jpeg", "mid"),
            ("noise", "high"),
            ("fog", "low"),
        ],
    )

    assert isinstance(result, Image.Image)
    assert result.size == image.size


def test_apply_degradations_rejects_unknown_severity():
    image = Image.new("RGB", (32, 32), color=(128, 128, 128))

    with pytest.raises(ValueError, match="Unknown severity level"):
        apply_degradations(image, operations=[("blur", "ultra")])
