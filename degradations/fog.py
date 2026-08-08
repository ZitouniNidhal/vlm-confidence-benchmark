import numpy as np
from PIL import Image


def apply_fog(image: Image.Image, opacity: float = 0.3) -> Image.Image:
    """Simulate fog by blending the image with a white haze layer."""
    array = np.asarray(image).astype("float32")
    haze = np.full_like(array, 255.0)
    blended = array * (1.0 - opacity) + haze * opacity
    return Image.fromarray(np.clip(blended, 0.0, 255.0).astype("uint8"))
