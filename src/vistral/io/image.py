"""Reading images into :class:`vistral.Image`. OpenCV is imported only when reading files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vistral.core.errors import MissingDependencyError
from vistral.core.image import Image
from vistral.core.spec import Color, ImageSpec, Range


def read_image(
    source: Any,
    *,
    color: Color | str | None = None,
    range: Range | str | None = None,
    spec: ImageSpec | None = None,
) -> Image:
    """Load or wrap an image.

    Paths are decoded with OpenCV (``IMREAD_UNCHANGED``, so 16-bit and alpha are preserved;
    color images are BGR). NumPy arrays are wrapped (3 channels assumed BGR unless ``color``
    says otherwise), PIL images are converted, and ``Image`` objects are returned as is.
    """
    if isinstance(source, Image):
        return source
    if isinstance(source, (str, Path)):
        return _read_path(Path(source), color=color, range=range, spec=spec)
    if isinstance(source, np.ndarray) or hasattr(source, "device"):
        return Image.from_array(source, color=color, range=range, spec=spec)
    if hasattr(source, "mode") and hasattr(source, "getbands"):
        return Image.from_pil(source)
    raise TypeError(f"cannot read an image from {type(source).__name__}")


def _read_path(path: Path, *, color: Any, range: Any, spec: ImageSpec | None) -> Image:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - cv2 is a core dependency
        raise MissingDependencyError("opencv-python-headless") from exc

    if not path.is_file():
        raise FileNotFoundError(path)

    data = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise ValueError(f"OpenCV could not decode {path}")

    if color is None and (spec is None or spec.color is None):
        color = {2: Color.GRAY, 3: None}.get(data.ndim)

        if data.ndim == 3:
            color = {1: Color.GRAY, 3: Color.BGR, 4: Color.BGRA}.get(data.shape[-1])

    return Image.from_array(data, color=color, range=range, spec=spec)
