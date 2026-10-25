"""Small deterministic synthetic images for tests (no binary fixtures needed)."""

from __future__ import annotations

import numpy as np

from vistral.core.frame import Frame
from vistral.core.image import Image
from vistral.core.spec import Color, Range


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def bgr_u8(h: int = 48, w: int = 64, seed: int = 0) -> Image:
    return Image.from_array(_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8), color=Color.BGR)


def rgb_u8(h: int = 48, w: int = 64, seed: int = 0) -> Image:
    return Image.from_array(_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8), color=Color.RGB)


def bgra_u8(h: int = 48, w: int = 64, seed: int = 0) -> Image:
    return Image.from_array(
        _rng(seed).integers(0, 256, (h, w, 4), dtype=np.uint8), color=Color.BGRA
    )


def gray_u8(h: int = 48, w: int = 64, seed: int = 0) -> Image:
    return Image.from_array(_rng(seed).integers(0, 256, (h, w), dtype=np.uint8), color=Color.GRAY)


def gray_u16(h: int = 48, w: int = 64, seed: int = 0) -> Image:
    data = _rng(seed).integers(0, 65536, (h, w), dtype=np.uint16)
    return Image.from_array(data, color=Color.GRAY)


def rgb_unit(h: int = 48, w: int = 64, seed: int = 0) -> Image:
    data = _rng(seed).random((h, w, 3), dtype=np.float32)
    return Image.from_array(data, color=Color.RGB, range=Range.UNIT)


def mask(h: int = 48, w: int = 64, seed: int = 0) -> Image:
    return Image.from_array(_rng(seed).random((h, w)) > 0.5, color=Color.MASK)


def gradient_bgr(h: int = 48, w: int = 64) -> Image:
    """Deterministic image whose channels differ, so BGR/RGB swaps are visible."""
    y, x = np.mgrid[0:h, 0:w]
    data = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), np.full_like(x, 40)], -1)
    return Image.from_array(data.astype(np.uint8), color=Color.BGR)


def frame(image: Image | None = None) -> Frame:
    return Frame(image if image is not None else bgr_u8())
