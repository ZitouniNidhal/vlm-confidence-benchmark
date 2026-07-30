from PIL import ImageEnhance


def apply_glare(image, factor: float = 1.5):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)
