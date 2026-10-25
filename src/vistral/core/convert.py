"""Explicit image conversions planned over a graph of spec states.

Each converter is an edge between spec states with a cost and a ``lossy`` flag. Planning
finds the cheapest path from a spec to any spec an ``Accepts`` constraint allows. Automatic
conversion only uses lossless edges unless lossy ones are explicitly allowed (D-019).
Normalization is never a converter: it belongs to model encoders.
"""

from __future__ import annotations

import heapq
import importlib.util
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from vistral.core.errors import ConversionError
from vistral.core.image import Image
from vistral.core.spec import (
    CHANNELS,
    Accepts,
    Backend,
    Color,
    ImageSpec,
    Layout,
    Range,
)
from vistral.core.trace import ConversionRecord, record_conversion


@dataclass(frozen=True)
class Converter:
    name: str
    cost: float
    lossy: bool
    applies: Callable[[ImageSpec], bool]
    result: Callable[[ImageSpec], ImageSpec]
    fn: Callable[[Any, ImageSpec], Any]

    def __str__(self) -> str:
        return f"{self.name}{' (lossy)' if self.lossy else ''}"


def describe_path(path: list[Converter]) -> str:
    return " -> ".join(str(c) for c in path) if path else "(no conversion)"


class ConversionGraph:
    def __init__(self, converters: list[Converter] | None = None) -> None:
        self._converters: list[Converter] = list(converters or [])
        self._torch_registered = False

    def register(self, converter: Converter) -> None:
        self._converters.append(converter)

    @property
    def converters(self) -> list[Converter]:
        self._ensure_torch()
        return list(self._converters)

    def _ensure_torch(self) -> None:
        if not self._torch_registered:
            self._torch_registered = True
            if importlib.util.find_spec("torch") is not None:
                for conv in _torch_converters():
                    self.register(conv)

    def plan(
        self,
        spec: ImageSpec,
        target: Accepts | ImageSpec,
        *,
        allow_lossy: bool = False,
        max_steps: int = 6,
    ) -> list[Converter] | None:
        """Cheapest converter path from ``spec`` to a spec satisfying ``target``.

        Returns ``[]`` if ``spec`` already satisfies it, ``None`` if no path exists (including
        when a constrained field of ``spec`` is unknown).
        """
        accepts = Accepts.coerce(target)
        assert accepts is not None
        violations, unknown = accepts.check(spec)
        if unknown:
            return None
        if not violations:
            return []
        convs = [c for c in self.converters if allow_lossy or not c.lossy]
        return self._search(spec, accepts, convs, max_steps)

    @staticmethod
    def _search(
        spec: ImageSpec, accepts: Accepts, convs: list[Converter], max_steps: int
    ) -> list[Converter] | None:
        counter = itertools.count()
        frontier: list[tuple[float, int, ImageSpec, list[Converter]]] = [(0.0, 0, spec, [])]
        best: dict[ImageSpec, float] = {spec: 0.0}
        while frontier:
            cost, _, state, path = heapq.heappop(frontier)
            if accepts.satisfied_by(state):
                return path
            if len(path) >= max_steps or cost > best.get(state, float("inf")):
                continue
            _expand_frontier(state, cost, path, convs, best, frontier, counter)
        return None

    def apply(self, image: Image, path: list[Converter]) -> Image:
        for conv in path:
            before = image
            target = conv.result(image.spec)
            data = conv.fn(image.data, image.spec)
            declared = {k: getattr(target, k) for k in ("color", "range", "layout")}
            image = Image(data, _respec(data, declared), image.transforms)
            record_conversion(
                ConversionRecord(
                    conv.name,
                    str(before.spec),
                    str(image.spec),
                    copied=_copied(before.data, data),
                    nbytes=_nbytes(data),
                    lossy=conv.lossy,
                )
            )
        return image

    def convert(
        self, image: Image, target: Accepts | ImageSpec | dict, *, allow_lossy: bool = False
    ) -> Image:
        accepts = Accepts.coerce(target)
        assert accepts is not None
        path = self.plan(image.spec, accepts, allow_lossy=allow_lossy)
        if path is None:
            lossy = None if allow_lossy else self.plan(image.spec, accepts, allow_lossy=True)
            hint = (
                f"; a lossy path exists ({describe_path(lossy)}), pass allow_lossy=True"
                if (lossy)
                else ""
            )
            raise ConversionError(f"no conversion from {image.spec} to {accepts}{hint}")
        return self.apply(image, path)


def _respec(data: Any, declared: dict[str, Any]) -> ImageSpec:
    from vistral.core.spec import derive_spec

    return derive_spec(data, declared=declared)


def _expand_frontier(
    state: ImageSpec,
    cost: float,
    path: list[Converter],
    convs: list[Converter],
    best: dict[ImageSpec, float],
    frontier: list[tuple[float, int, ImageSpec, list[Converter]]],
    counter: itertools.count,
) -> None:
    for conv in convs:
        if not conv.applies(state):
            continue
        nxt = conv.result(state)
        ncost = cost + conv.cost
        if ncost >= best.get(nxt, float("inf")):
            continue
        best[nxt] = ncost
        heapq.heappush(frontier, (ncost, next(counter), nxt, [*path, conv]))


def _copied(before: Any, after: Any) -> bool:
    if isinstance(before, np.ndarray) and isinstance(after, np.ndarray):
        return not np.may_share_memory(before, after)
    before_ptr, after_ptr = getattr(before, "data_ptr", None), getattr(after, "data_ptr", None)
    if before_ptr is None or after_ptr is None:
        return True
    return before_ptr() != after_ptr()  # torch


def _nbytes(data: Any) -> int:
    if isinstance(data, np.ndarray):
        return int(data.nbytes)
    try:
        return int(data.element_size() * data.nelement())
    except AttributeError:
        return 0


# -- numpy converters ------------------------------------------------------------------------

_SWAP = {Color.BGR: Color.RGB, Color.RGB: Color.BGR, Color.BGRA: Color.RGBA, Color.RGBA: Color.BGRA}
_NO_ALPHA = {Color.BGRA: Color.BGR, Color.RGBA: Color.RGB}


def _color(spec: ImageSpec) -> Color:
    assert spec.color is not None  # guaranteed by the converter's applies()
    return spec.color


def _np(spec: ImageSpec) -> bool:
    return spec.backend is Backend.NUMPY


def _channel_axis(spec: ImageSpec) -> int:
    return 0 if spec.layout is Layout.CHW else -1


def _swap_rb(data: Any, spec: ImageSpec) -> Any:
    order = [2, 1, 0] if CHANNELS[spec.color] == 3 else [2, 1, 0, 3]  # type: ignore[index]
    return np.take(data, order, axis=_channel_axis(spec))


def _drop_alpha(data: Any, spec: ImageSpec) -> Any:
    return np.ascontiguousarray(np.take(data, [0, 1, 2], axis=_channel_axis(spec)))


def _gray_to_3(data: Any, spec: ImageSpec) -> Any:
    if spec.layout is Layout.HW:
        return np.repeat(data[..., None], 3, axis=-1)
    return np.repeat(data, 3, axis=_channel_axis(spec))


_BGR_WEIGHTS = np.array([0.114, 0.587, 0.299])


def _to_gray(data: Any, spec: ImageSpec) -> Any:
    channels = np.moveaxis(data, _channel_axis(spec), -1)[..., :3].astype(np.float64)
    weights = _BGR_WEIGHTS if spec.color in (Color.BGR, Color.BGRA) else _BGR_WEIGHTS[::-1]
    gray = channels @ weights
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        return np.clip(np.rint(gray), info.min, info.max).astype(data.dtype)
    return gray.astype(data.dtype)


def _scale(factor: float, dtype: Any, *, rounding: bool = False, lo: float = 0, hi: float = 1):
    def fn(data: Any, spec: ImageSpec) -> Any:
        out = data.astype(np.float32) * np.float32(factor) if not rounding else data * factor
        if rounding:
            out = np.rint(np.clip(out, lo, hi))
        return out.astype(dtype)

    return fn


def _layout_is(*layouts: Layout) -> Callable[[ImageSpec], bool]:
    return lambda s: _np(s) and s.layout in layouts


def _numpy_color_layout(*colors: Color) -> Callable[[ImageSpec], bool]:
    return lambda s: _np(s) and s.color in colors and s.layout in (Layout.HWC, Layout.CHW)


def _numpy_range(kind: str, dtype: str) -> Callable[[ImageSpec], bool]:
    return lambda s: _np(s) and s.range is not None and s.range.kind == kind and s.dtype == dtype


def _unit_range(spec: ImageSpec) -> bool:
    return _np(spec) and spec.range is not None and spec.range.kind == "UNIT"


def _hw_single_channel(spec: ImageSpec) -> bool:
    return _np(spec) and spec.layout is Layout.HW and spec.color in (Color.GRAY, Color.MASK)


def _gray_bgr_layout(spec: ImageSpec) -> ImageSpec:
    return spec.replace(
        color=Color.BGR, layout=Layout.CHW if spec.layout is Layout.CHW else Layout.HWC
    )


def _gray_rgb_layout(spec: ImageSpec) -> ImageSpec:
    return spec.replace(
        color=Color.RGB, layout=Layout.CHW if spec.layout is Layout.CHW else Layout.HWC
    )


def _gray_source(spec: ImageSpec) -> bool:
    return _np(spec) and spec.color is Color.GRAY and spec.layout is not None


def _to_gray_source(spec: ImageSpec) -> bool:
    return (
        _np(spec)
        and spec.color in (Color.BGR, Color.RGB, Color.BGRA, Color.RGBA)
        and spec.layout in (Layout.HWC, Layout.CHW)
    )


def _single_channel(spec: ImageSpec) -> bool:
    return (
        _np(spec)
        and spec.layout in (Layout.HWC, Layout.CHW)
        and spec.color in (Color.GRAY, Color.MASK)
    )


def _squeeze_channel(data: Any, spec: ImageSpec) -> Any:
    return data[0] if spec.layout is Layout.CHW else data[..., 0]


def _numpy_converters() -> list[Converter]:
    return [
        Converter(
            "swap_rb",
            1.0,
            False,
            _numpy_color_layout(*_SWAP),
            lambda s: s.replace(color=_SWAP[_color(s)]),
            _swap_rb,
        ),
        Converter(
            "drop_alpha",
            1.0,
            True,
            _numpy_color_layout(*_NO_ALPHA),
            lambda s: s.replace(color=_NO_ALPHA[_color(s)]),
            _drop_alpha,
        ),
        Converter(
            "gray_to_bgr",
            1.0,
            False,
            _gray_source,
            _gray_bgr_layout,
            _gray_to_3,
        ),
        Converter(
            "gray_to_rgb",
            1.0,
            False,
            _gray_source,
            _gray_rgb_layout,
            _gray_to_3,
        ),
        Converter(
            "to_gray",
            2.0,
            True,
            _to_gray_source,
            lambda s: s.replace(color=Color.GRAY, layout=Layout.HW),
            _to_gray,
        ),
        Converter(
            "hw_to_hwc",
            0.01,
            False,
            _hw_single_channel,
            lambda s: s.replace(layout=Layout.HWC),
            lambda d, s: d[..., None],
        ),
        Converter(
            "hw_to_chw",
            0.01,
            False,
            _hw_single_channel,
            lambda s: s.replace(layout=Layout.CHW),
            lambda d, s: d[None],
        ),
        Converter(
            "squeeze_channel",
            0.01,
            False,
            _single_channel,
            lambda s: s.replace(layout=Layout.HW),
            _squeeze_channel,
        ),
        Converter(
            "hwc_to_chw",
            0.1,
            False,
            _layout_is(Layout.HWC),
            lambda s: s.replace(layout=Layout.CHW),
            lambda d, s: d.transpose(2, 0, 1),
        ),
        Converter(
            "chw_to_hwc",
            0.1,
            False,
            _layout_is(Layout.CHW),
            lambda s: s.replace(layout=Layout.HWC),
            lambda d, s: d.transpose(1, 2, 0),
        ),
        Converter(
            "u8_to_unit",
            1.0,
            False,
            _numpy_range("U8", "uint8"),
            lambda s: s.replace(dtype="float32", range=Range.UNIT),
            _scale(1 / 255, np.float32),
        ),
        Converter(
            "u16_to_unit",
            1.0,
            False,
            _numpy_range("U16", "uint16"),
            lambda s: s.replace(dtype="float32", range=Range.UNIT),
            _scale(1 / 65535, np.float32),
        ),
        Converter(
            "unit_to_u8",
            1.0,
            True,
            _unit_range,
            lambda s: s.replace(dtype="uint8", range=Range.U8),
            _scale(255, np.uint8, rounding=True, hi=255),
        ),
        Converter(
            "unit_to_u16",
            1.0,
            True,
            _unit_range,
            lambda s: s.replace(dtype="uint16", range=Range.U16),
            _scale(65535, np.uint16, rounding=True, hi=65535),
        ),
        Converter(
            "u8_to_u16",
            1.0,
            False,
            _numpy_range("U8", "uint8"),
            lambda s: s.replace(dtype="uint16", range=Range.U16),
            lambda d, s: d.astype(np.uint16) * np.uint16(257),
        ),
        Converter(
            "u16_to_u8",
            1.0,
            True,
            _numpy_range("U16", "uint16"),
            lambda s: s.replace(dtype="uint8", range=Range.U8),
            _scale(1 / 257, np.uint8, rounding=True, hi=255),
        ),
        Converter(
            "f16_to_f32",
            0.5,
            False,
            lambda s: _np(s) and s.dtype == "float16",
            lambda s: s.replace(dtype="float32"),
            lambda d, s: d.astype(np.float32),
        ),
        Converter(
            "f64_to_f32",
            0.5,
            True,
            lambda s: _np(s) and s.dtype == "float64",
            lambda s: s.replace(dtype="float32"),
            lambda d, s: d.astype(np.float32),
        ),
        Converter(
            "f32_to_f16",
            0.5,
            True,
            lambda s: _np(s) and s.dtype == "float32",
            lambda s: s.replace(dtype="float16"),
            lambda d, s: d.astype(np.float16),
        ),
    ]


def _torch_converters() -> list[Converter]:
    """Registered only when torch is installed; torch is imported when a conversion runs."""

    def to_torch(data: Any, spec: ImageSpec) -> Any:
        import torch  # pyright: ignore[reportMissingImports]

        if any(s < 0 for s in data.strides):
            data = np.ascontiguousarray(data)
        return torch.from_numpy(data)

    def is_torch(device_prefix: str | None = None) -> Callable[[ImageSpec], bool]:
        def check(s: ImageSpec) -> bool:
            if s.backend is not Backend.TORCH or s.device is None:
                return False
            return device_prefix is None or s.device.startswith(device_prefix)

        return check

    return [
        Converter(
            "numpy_to_torch",
            0.05,
            False,
            _np,
            lambda s: s.replace(backend=Backend.TORCH, device="cpu"),
            to_torch,
        ),
        Converter(
            "torch_to_numpy",
            0.05,
            False,
            is_torch("cpu"),
            lambda s: s.replace(backend=Backend.NUMPY),
            lambda d, s: d.numpy(),
        ),
        Converter(
            "to_cuda",
            5.0,
            False,
            is_torch("cpu"),
            lambda s: s.replace(device="cuda"),
            lambda d, s: d.to("cuda"),
        ),
        Converter(
            "to_cpu",
            5.0,
            False,
            is_torch("cuda"),
            lambda s: s.replace(device="cpu"),
            lambda d, s: d.to("cpu"),
        ),
    ]


DEFAULT_GRAPH = ConversionGraph(_numpy_converters())


def plan(
    spec: ImageSpec, target: Accepts | ImageSpec | dict, *, allow_lossy: bool = False
) -> list[Converter] | None:
    accepts = Accepts.coerce(target)
    assert accepts is not None
    return DEFAULT_GRAPH.plan(spec, accepts, allow_lossy=allow_lossy)


def convert_image(
    image: Image, target: Accepts | ImageSpec | dict, *, allow_lossy: bool = False
) -> Image:
    """Convert ``image`` to satisfy ``target``, choosing the cheapest path."""
    return DEFAULT_GRAPH.convert(image, target, allow_lossy=allow_lossy)
