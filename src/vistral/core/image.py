"""``Image``: pixel data plus its spec and the geometric transforms applied since the source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vistral.core.errors import ConversionError
from vistral.core.geometry import GeoTransform
from vistral.core.spec import Color, ImageSpec, Layout, Range, derive_spec, observe


def _default_color(shape: tuple[int, ...], dtype: str) -> Color | None:
    if dtype == "bool":
        return Color.MASK
    if len(shape) == 2 or (len(shape) == 3 and shape[-1] == 1):
        return Color.GRAY
    if len(shape) == 3 and shape[-1] == 3:
        return Color.BGR
    if len(shape) == 3 and shape[-1] == 4:
        return Color.BGRA
    return None


@dataclass(frozen=True, eq=False)
class Image:
    """Immutable image. Stages return new ``Image`` objects; arrays may be shared (views)."""

    data: Any
    spec: ImageSpec
    transforms: tuple[GeoTransform, ...] = ()

    @classmethod
    def from_array(
        cls,
        data: Any,
        *,
        color: Color | str | None = None,
        range: Range | str | None = None,
        layout: Layout | str | None = None,
        spec: ImageSpec | None = None,
    ) -> Image:
        """Wrap an array. A 3-channel NumPy array is assumed BGR (OpenCV order) unless declared.

        ``spec`` supplies declarations; observable fields always come from the data.
        """
        declared: dict[str, Any] = dict(spec.known()) if spec is not None else {}
        for key in ("dtype", "size", "device", "backend"):
            declared.pop(key, None)
        if color is not None:
            declared["color"] = color
        if range is not None:
            declared["range"] = range
        if layout is not None:
            declared["layout"] = layout
        if "color" not in declared:
            obs = observe(data)
            default = _default_color(obs.shape, obs.dtype)
            if default is not None:
                declared["color"] = default
        return cls(data, derive_spec(data, declared=declared, strict=True))

    @classmethod
    def from_pil(cls, img: Any) -> Image:
        mode = img.mode
        if mode in ("I;16", "I;16L", "I;16B"):
            return cls.from_array(np.array(img, dtype=np.uint16), color=Color.GRAY)
        if mode == "1":
            return cls.from_array(np.array(img, dtype=bool), color=Color.MASK)
        if mode == "L":
            return cls.from_array(np.array(img), color=Color.GRAY)
        if mode == "RGBA":
            return cls.from_array(np.array(img), color=Color.RGBA)
        if mode != "RGB":
            img = img.convert("RGB")
        return cls.from_array(np.array(img), color=Color.RGB)

    def to_pil(self) -> Any:
        """Return a PIL image (copies). Requires NumPy uint8/uint16 data in HWC or HW layout."""
        from PIL import Image as PILImage

        data = self.data
        if not isinstance(data, np.ndarray) or self.spec.layout not in (Layout.HW, Layout.HWC):
            raise ConversionError(f"to_pil needs a NumPy HW/HWC image, got {self.spec}")
        color = self.spec.color
        if color in (Color.BGR, Color.BGRA):
            order = [2, 1, 0] if color is Color.BGR else [2, 1, 0, 3]
            data = data[..., order]
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data[..., 0]
        return PILImage.fromarray(np.ascontiguousarray(data))

    def with_data(self, data: Any, **declared: Any) -> Image:
        """New image with ``data``; spec fields re-derived, ``declared`` fields win."""
        return Image(data, derive_spec(data, base=self.spec, declared=declared), self.transforms)

    def with_transform(self, data: Any, transform: GeoTransform, **declared: Any) -> Image:
        """New image after a geometric operation; appends ``transform``."""
        spec = derive_spec(data, base=self.spec, declared=declared)
        return Image(data, spec, (*self.transforms, transform))

    def copy(self) -> Image:
        data = self.data.copy() if isinstance(self.data, np.ndarray) else self.data.clone()
        return Image(data, self.spec, self.transforms)

    @property
    def size(self) -> tuple[int, int] | None:
        return self.spec.size

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    def __repr__(self) -> str:
        geo = f", transforms={len(self.transforms)}" if self.transforms else ""
        return f"Image({self.spec}{geo})"
