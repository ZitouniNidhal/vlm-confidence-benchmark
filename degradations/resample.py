from PIL import Image


def apply_resample(image, scale: float = 0.5):
    width, height = image.size
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, resample=Image.BICUBIC).resize((width, height), resample=Image.BICUBIC)
