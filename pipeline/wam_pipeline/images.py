"""Strict RGB PNG Base64 codec shared by the API, tests, and adapters."""

from __future__ import annotations

import base64
import io
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable

import numpy as np
from PIL import Image

from .profile import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH


class ImageValidationError(ValueError):
    """The API image violates the published PNG/RGB profile."""


def encode_png_base64(image: np.ndarray) -> dict:
    """Encode one HWC uint8 RGB image using exactly the official API envelope."""
    if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS) or image.dtype != np.uint8:
        raise ImageValidationError(
            f"expected uint8 RGB [{IMAGE_HEIGHT}, {IMAGE_WIDTH}, 3], got {image.shape} {image.dtype}"
        )
    output = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(output, format="PNG")
    return {
        "encoding": "png_base64",
        "color_space": "RGB",
        "height": IMAGE_HEIGHT,
        "width": IMAGE_WIDTH,
        "channels": IMAGE_CHANNELS,
        "data": base64.b64encode(output.getvalue()).decode("ascii"),
    }


def decode_png_base64(payload: object) -> np.ndarray:
    """Decode and validate an official image envelope without silently converting modes."""
    if not isinstance(payload, dict):
        raise ImageValidationError("image must be an object")
    expected = {
        "encoding": "png_base64",
        "color_space": "RGB",
        "height": IMAGE_HEIGHT,
        "width": IMAGE_WIDTH,
        "channels": IMAGE_CHANNELS,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ImageValidationError(f"image.{key} must be {value!r}")
    encoded = payload.get("data")
    if not isinstance(encoded, str) or encoded.startswith("data:") or "\n" in encoded or "\r" in encoded:
        raise ImageValidationError("image.data must be one-line standard Base64 without a data prefix")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ImageValidationError("image.data is not valid Base64") from exc
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ImageValidationError("image must be a non-palette RGB PNG without alpha")
            array = np.asarray(image)
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError("image.data is not a decodable PNG") from exc
    if array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS) or array.dtype != np.uint8:
        raise ImageValidationError("decoded PNG does not match the official RGB dimensions")
    return array.copy()


def encode_png_base64_batch(
    images: Iterable[np.ndarray], workers: int = 1
) -> list[dict]:
    """Encode independent frames in stable input order using CPU workers."""
    values = list(images)
    if workers < 1:
        raise ValueError("PNG codec workers must be at least one")
    if workers == 1 or len(values) < 2:
        return [encode_png_base64(value) for value in values]
    with ThreadPoolExecutor(max_workers=min(workers, len(values))) as executor:
        return list(executor.map(encode_png_base64, values))


def decode_png_base64_batch(
    payloads: Iterable[object], workers: int = 1
) -> list[np.ndarray]:
    """Decode independent frames in stable input order using CPU workers."""
    values = list(payloads)
    if workers < 1:
        raise ValueError("PNG codec workers must be at least one")
    if workers == 1 or len(values) < 2:
        return [decode_png_base64(value) for value in values]
    with ThreadPoolExecutor(max_workers=min(workers, len(values))) as executor:
        return list(executor.map(decode_png_base64, values))
