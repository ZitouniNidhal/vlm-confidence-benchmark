import numpy as np
from PIL import Image


def apply_noise(image: Image.Image, amount: float = 0.05) -> Image.Image:
    """Add Gaussian pixel noise to an image."""
    array = np.asarray(image).astype("float32") / 255.0
    noise = np.random.normal(loc=0.0, scale=amount, size=array.shape).astype("float32")
    noisy = np.clip(array + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).astype("uint8"))
