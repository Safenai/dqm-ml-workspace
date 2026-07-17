"""Image generators for visual features stress tests.

Each function returns either a single JPEG ``bytes`` object or a ``list[bytes]``
of related variants.  Callers compose the generators they need.
"""

import io

import numpy as np
from PIL import Image, ImageDraw

from tests.utils.seeds import get_test_seed


def _jpg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _uniform_image(r: int, g: int, b: int, w: int = 100, h: int = 70) -> bytes:
    return _jpg_bytes(Image.new("RGB", (w, h), (r, g, b)))


def uniform_black() -> bytes:
    return _uniform_image(0, 0, 0)


def uniform_white() -> bytes:
    return _uniform_image(255, 255, 255)


def uniform_gray() -> bytes:
    return _uniform_image(128, 128, 128)


def uniform_series() -> list[bytes]:
    return [uniform_black(), uniform_white(), uniform_gray()]


def checkerboard(w: int = 100, h: int = 70, step: int = 4) -> bytes:
    cb = Image.new("L", (w, h), 0)
    pix = cb.load()
    for x in range(w):
        for y in range(h):
            if (x // step + y // step) % 2 == 0:
                pix[x, y] = 255
    return _jpg_bytes(cb.convert("RGB"))


def gaussian_noise_images(rng: np.random.Generator, n: int = 2, w: int = 100, h: int = 70) -> list[bytes]:
    return [_jpg_bytes(Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), mode="RGB")) for _ in range(n)]


def salt_pepper_images(rng: np.random.Generator, n: int = 2, w: int = 100, h: int = 70) -> list[bytes]:
    images: list[bytes] = []
    size = w * h
    for _ in range(n):
        base = np.full((h, w, 3), 128, dtype=np.uint8)
        n_salt = rng.choice(size, size=500, replace=False)
        n_pepper = rng.choice(size, size=500, replace=False)
        base.reshape(-1, 3)[n_salt] = [255, 255, 255]
        base.reshape(-1, 3)[n_pepper] = [0, 0, 0]
        images.append(_jpg_bytes(Image.fromarray(base, mode="RGB")))
    return images


def near_uniform_images(
    rng: np.random.Generator, base_values: tuple[int, ...] = (64, 192), w: int = 100, h: int = 70
) -> list[bytes]:
    images: list[bytes] = []
    for base_val in base_values:
        arr = np.full((h, w, 3), base_val, dtype=np.uint8)
        noise = rng.integers(-1, 2, (h, w, 3), dtype=np.int8)
        arr = np.clip(arr.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        images.append(_jpg_bytes(Image.fromarray(arr, mode="RGB")))
    return images


def jpeg_artifact_images(
    colors: list[tuple[int, int, int]], quality: int = 5, w: int = 100, h: int = 70
) -> list[bytes]:
    images: list[bytes] = []
    for color in colors:
        img = Image.new("RGB", (w, h), color)
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, w - 10, h - 10], fill=(255 - color[0], 255 - color[1], 255 - color[2]))
        images.append(_jpg_bytes(img, quality=quality))
    return images


def thin_image_tall(w: int = 1, h: int = 500) -> bytes:
    return _jpg_bytes(Image.new("RGB", (w, h), (128, 128, 128)))


def thin_image_wide(w: int = 500, h: int = 1) -> bytes:
    return _jpg_bytes(Image.new("RGB", (w, h), (128, 128, 128)))


def tiny_2x2_image() -> bytes:
    return _jpg_bytes(Image.new("RGB", (2, 2), (0, 255, 0)))


def single_pixel_contrast_image(w: int = 100, h: int = 70) -> bytes:
    img = Image.new("RGB", (w, h), (100, 100, 100))
    img.putpixel((w // 2, h // 2), (255, 0, 0))
    return _jpg_bytes(img)


def gradient_images(vertical: bool = True, w: int = 100, h: int = 70) -> list[bytes]:
    grad = Image.new("L", (w, h), 0)
    for x in range(w):
        for y in range(h):
            if vertical:
                grad.putpixel((x, y), int(255 * y / (h - 1)))
            else:
                grad.putpixel((x, y), int(255 * x / (w - 1)))
    return [_jpg_bytes(grad.convert("RGB"))]


def random_rgb_noise_images(rng: np.random.Generator, n: int = 2, w: int = 100, h: int = 70) -> list[bytes]:
    images: list[bytes] = []
    for _ in range(n):
        r_ch = rng.integers(0, 256, (h, w), dtype=np.uint8)
        g_ch = rng.integers(0, 256, (h, w), dtype=np.uint8)
        b_ch = rng.integers(0, 256, (h, w), dtype=np.uint8)
        rgb = np.stack([r_ch, g_ch, b_ch], axis=-1)
        images.append(_jpg_bytes(Image.fromarray(rgb, mode="RGB")))
    return images


def generate_stress_images() -> list[bytes]:
    """Generate 20 stress-test images covering edge cases for visual features.

    Composes the individual generators above with a shared RNG seed.
    """
    rng = np.random.default_rng(get_test_seed())
    return [
        *uniform_series(),
        checkerboard(),
        *gaussian_noise_images(rng, 2),
        *salt_pepper_images(rng, 2),
        *near_uniform_images(rng),
        *jpeg_artifact_images([(0, 0, 0), (255, 0, 0)], quality=5),
        thin_image_tall(),
        thin_image_wide(),
        tiny_2x2_image(),
        single_pixel_contrast_image(),
        *gradient_images(vertical=True),
        *gradient_images(vertical=False),
        *random_rgb_noise_images(rng, 2),
    ]
