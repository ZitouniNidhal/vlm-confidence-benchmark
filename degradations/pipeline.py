from __future__ import annotations

from typing import Any

from PIL import Image

from degradations.blur import apply_blur
from degradations.glare import apply_glare
from degradations.jpeg import apply_jpeg_compression
from degradations.lowlight import apply_lowlight
from degradations.resample import apply_resample
from degradations.rotation import apply_rotation


DEGRADATIONS = {
    "blur": apply_blur,
    "glare": apply_glare,
    "jpeg": apply_jpeg_compression,
    "lowlight": apply_lowlight,
    "resample": apply_resample,
    "rotation": apply_rotation,
}


def apply_degradations(image: Image.Image, operations: list[tuple[str, dict[str, Any]]] | None = None) -> Image.Image:
    """Apply a sequence of degradations to an image in order."""
    current = image.convert("RGB")

    for name, params in operations or []:
        if name not in DEGRADATIONS:
            raise ValueError(f"Unknown degradation: {name}")
        current = DEGRADATIONS[name](current, **params)

    return current
