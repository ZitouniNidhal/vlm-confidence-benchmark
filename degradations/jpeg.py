import io

from PIL import Image


def apply_jpeg_compression(image, quality: int = 75):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)
