from PIL import ImageEnhance


def apply_lowlight(image, factor: float = 0.5):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)
