from PIL import ImageFilter


def apply_blur(image, radius: float = 2.0):
    return image.filter(ImageFilter.GaussianBlur(radius))
