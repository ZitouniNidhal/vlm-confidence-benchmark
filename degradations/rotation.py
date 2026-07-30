def apply_rotation(image, angle: float = 15.0):
    return image.rotate(angle, resample=Image.BICUBIC, expand=True)
