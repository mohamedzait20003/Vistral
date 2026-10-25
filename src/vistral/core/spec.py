"""Image specifications: what pixel data *is* (``ImageSpec``) and what a stage *accepts*.

Color order and value range cannot be recovered from an array, so they are declared and tracked.
Layout, dtype, size, device and backend are observable and re-derived from data at runtime.
Sizes are always ``(height, width)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np

from vistral.core.errors import ConfigError, SpecMismatchError, Violation


class Layout(StrEnum):
    HWC = "HWC"
    CHW = "CHW"
    HW = "HW"


class Color(StrEnum):
    BGR = "BGR"
    RGB = "RGB"
    GRAY = "GRAY"
    BGRA = "BGRA"
    RGBA = "RGBA"
    MASK = "MASK"


class Backend(StrEnum):
    NUMPY = "numpy"
    TORCH = "torch"


CHANNELS: dict[Color, int] = {
    Color.GRAY: 1,
    Color.MASK: 1,
    Color.BGR: 3,
    Color.RGB: 3,
    Color.BGRA: 4,
    Color.RGBA: 4,
}

DTYPES = frozenset(
    {"uint8", "uint16", "int16", "int32", "int64", "float16", "float32", "float64", "bool"}
)

RANGE_KINDS = ("U8", "U16", "UNIT", "NORMALIZED", "ANY")


@dataclass(frozen=True)
class Range:
    """Value range of pixel data.

    ``U8`` is 0-255, ``U16`` is 0-65535, ``UNIT`` is 0-1, ``NORMALIZED`` is ``(x - mean) / std``
    of unit values, and ``ANY`` is unbounded (gradients, differences).
    """

    kind: str
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None

    U8: ClassVar[Range]
    U16: ClassVar[Range]
    UNIT: ClassVar[Range]
    ANY: ClassVar[Range]

    def __post_init__(self) -> None:
        kind = str(self.kind).upper()
        if kind not in RANGE_KINDS:
            raise ValueError(f"unknown range kind {self.kind!r}; expected one of {RANGE_KINDS}")
        object.__setattr__(self, "kind", kind)
        if kind == "NORMALIZED":
            if self.mean is None or self.std is None:
                raise ValueError("NORMALIZED range requires mean and std")
            object.__setattr__(self, "mean", tuple(float(m) for m in self.mean))
            object.__setattr__(self, "std", tuple(float(s) for s in self.std))
        elif self.mean is not None or self.std is not None:
            raise ValueError(f"mean/std are only valid for NORMALIZED, not {kind}")

    @classmethod
    def normalized(cls, mean: Iterable[float], std: Iterable[float]) -> Range:
        return cls("NORMALIZED", tuple(mean), tuple(std))

    @classmethod
    def coerce(cls, value: Any) -> Range | None:
        if value is None or isinstance(value, Range):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Mapping):
            if "NORMALIZED" in value:
                inner = value["NORMALIZED"]
                return cls.normalized(inner["mean"], inner["std"])
            return cls(**value)
        raise TypeError(f"cannot interpret {value!r} as a Range")

    def to_config(self) -> str | dict[str, Any]:
        if self.kind == "NORMALIZED":
            return {"NORMALIZED": {"mean": list(self.mean or ()), "std": list(self.std or ())}}
        return self.kind

    def __str__(self) -> str:
        if self.kind == "NORMALIZED":
            return f"NORMALIZED(mean={self.mean}, std={self.std})"
        return self.kind


Range.U8 = Range("U8")
Range.U16 = Range("U16")
Range.UNIT = Range("UNIT")
Range.ANY = Range("ANY")


def _norm_dtype(value: Any) -> str | None:
    if value is None:
        return None
    try:
        name = np.dtype(value).name  # dtype objects, numpy scalar types, "uint8"
    except TypeError:
        name = str(value).removeprefix("torch.")  # torch dtypes
    if name not in DTYPES:
        raise ValueError(f"unsupported dtype {value!r}; expected one of {sorted(DTYPES)}")
    return name


def _norm_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    h, w = (int(v) for v in value)
    return (h, w)


SPEC_FIELDS = ("layout", "color", "dtype", "range", "device", "backend", "size")


@dataclass(frozen=True)
class ImageSpec:
    """Describes pixel data. ``None`` means unknown."""

    layout: Layout | None = None
    color: Color | None = None
    dtype: str | None = None
    range: Range | None = None
    device: str | None = None
    backend: Backend | None = None
    size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "layout", None if self.layout is None else Layout(self.layout))
        set_(self, "color", None if self.color is None else Color(self.color))
        set_(self, "dtype", _norm_dtype(self.dtype))
        set_(self, "range", Range.coerce(self.range))
        set_(self, "device", None if self.device is None else str(self.device))
        set_(self, "backend", None if self.backend is None else Backend(self.backend))
        set_(self, "size", _norm_size(self.size))

    # -- constructors -------------------------------------------------------------------------
    @classmethod
    def bgr_u8(cls, size: tuple[int, int] | None = None) -> ImageSpec:
        return cls(Layout.HWC, Color.BGR, "uint8", Range.U8, "cpu", Backend.NUMPY, size)

    @classmethod
    def rgb_u8(cls, size: tuple[int, int] | None = None) -> ImageSpec:
        return cls(Layout.HWC, Color.RGB, "uint8", Range.U8, "cpu", Backend.NUMPY, size)

    @classmethod
    def gray_u8(cls, size: tuple[int, int] | None = None) -> ImageSpec:
        return cls(Layout.HW, Color.GRAY, "uint8", Range.U8, "cpu", Backend.NUMPY, size)

    @classmethod
    def gray_u16(cls, size: tuple[int, int] | None = None) -> ImageSpec:
        return cls(Layout.HW, Color.GRAY, "uint16", Range.U16, "cpu", Backend.NUMPY, size)

    @classmethod
    def rgb_unit_chw(cls, size: tuple[int, int] | None = None) -> ImageSpec:
        return cls(Layout.CHW, Color.RGB, "float32", Range.UNIT, "cpu", Backend.NUMPY, size)

    @classmethod
    def coerce(cls, value: Any) -> ImageSpec | None:
        if value is None or isinstance(value, ImageSpec):
            return value
        if isinstance(value, Mapping):
            unknown = set(value) - set(SPEC_FIELDS)
            if unknown:
                raise ConfigError(f"unknown ImageSpec fields {sorted(unknown)}")
            return cls(**value)
        raise TypeError(f"cannot interpret {value!r} as an ImageSpec")

    # -- helpers ------------------------------------------------------------------------------
    def replace(self, **changes: Any) -> ImageSpec:
        values = {name: getattr(self, name) for name in SPEC_FIELDS}
        values.update(changes)
        return ImageSpec(**values)

    @property
    def channels(self) -> int | None:
        return None if self.color is None else CHANNELS[self.color]

    def known(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in SPEC_FIELDS if getattr(self, f) is not None}

    def to_config(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, value in self.known().items():
            if isinstance(value, Range):
                out[name] = value.to_config()
            elif isinstance(value, tuple):
                out[name] = list(value)
            else:
                out[name] = str(value)
        return out

    def __str__(self) -> str:
        parts = [
            str(self.color or "?color"),
            self.dtype or "?dtype",
            str(self.layout or "?layout"),
            str(self.range or "?range"),
        ]
        if self.size:
            parts.append(f"{self.size[0]}x{self.size[1]}")
        backend = self.backend or "?backend"
        parts.append(f"{backend}/{self.device or '?device'}")
        return " ".join(parts)


def _as_set(value: Any, kind: Any) -> frozenset[Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, Layout, Color, Backend, Range)) or not isinstance(value, Iterable):
        value = [value]
    return frozenset(kind(v) for v in value)


def _range_kind(value: Any) -> str:
    return value.kind if isinstance(value, Range) else Range(str(value)).kind


def _check_constraint(name: str, actual: Any, allowed: Any) -> Violation | None:
    if name == "size":
        if actual == allowed:
            return None
        hint = (
            "the sizes are swapped; Vistral sizes are (height, width)"
            if actual == allowed[::-1]
            else None
        )
        return Violation("size", actual, allowed, hint)
    if name == "range":
        if actual.kind in allowed:
            return None
        hint = (
            "the data is already normalized; remove the earlier normalization"
            if actual.kind == "NORMALIZED"
            else None
        )
        return Violation("range", actual, set(allowed), hint)
    if name == "device":
        matches = any(actual == device or actual.startswith(f"{device}:") for device in allowed)
        return None if matches else Violation("device", actual, set(allowed))
    return None if actual in allowed else Violation(name, actual, set(allowed))


@dataclass(frozen=True, init=False)
class Accepts:
    """A constraint on image specs. Each field is a set of allowed values, or ``None`` (any).

    ``range`` constrains the range *kind* (``U8``, ``UNIT``, ...). ``size`` requires an exact
    ``(height, width)``. ``device`` value ``"cuda"`` also matches ``"cuda:N"``.
    """

    layout: frozenset[Layout] | None
    color: frozenset[Color] | None
    dtype: frozenset[str] | None
    range: frozenset[str] | None
    device: frozenset[str] | None
    backend: frozenset[Backend] | None
    size: tuple[int, int] | None

    FIELDS: ClassVar[tuple[str, ...]] = SPEC_FIELDS

    def __init__(
        self,
        *,
        layout: Any = None,
        color: Any = None,
        dtype: Any = None,
        range: Any = None,
        device: Any = None,
        backend: Any = None,
        size: Any = None,
    ) -> None:
        set_ = object.__setattr__
        set_(self, "layout", _as_set(layout, Layout))
        set_(self, "color", _as_set(color, Color))
        set_(self, "dtype", _as_set(dtype, _norm_dtype))
        set_(self, "range", _as_set(range, _range_kind))
        set_(self, "device", _as_set(device, str))
        set_(self, "backend", _as_set(backend, Backend))
        set_(self, "size", _norm_size(size))

    @classmethod
    def from_spec(cls, spec: ImageSpec) -> Accepts:
        return cls(**spec.known())

    @classmethod
    def coerce(cls, value: Any) -> Accepts | None:
        if value is None or isinstance(value, Accepts):
            return value
        if isinstance(value, ImageSpec):
            return cls.from_spec(value)
        if isinstance(value, Mapping):
            return cls(**value)
        raise TypeError(f"cannot interpret {value!r} as Accepts")

    def constrained(self) -> tuple[str, ...]:
        return tuple(f for f in self.FIELDS if getattr(self, f) is not None)

    def check(self, spec: ImageSpec) -> tuple[list[Violation], list[str]]:
        """Return ``(violations, unknown_fields)`` of ``spec`` against this constraint."""
        violations: list[Violation] = []
        unknown: list[str] = []
        for name in self.constrained():
            allowed = getattr(self, name)
            actual = getattr(spec, name)
            if actual is None:
                unknown.append(name)
                continue
            violation = _check_constraint(name, actual, allowed)
            if violation is not None:
                violations.append(violation)
        return violations, unknown

    def satisfied_by(self, spec: ImageSpec) -> bool:
        violations, unknown = self.check(spec)
        return not violations and not unknown

    def to_config(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.constrained():
            value = getattr(self, name)
            out[name] = list(value) if name == "size" else sorted(str(v) for v in value)
        return out

    def __str__(self) -> str:
        parts = []
        for name in self.constrained():
            value = getattr(self, name)
            shown = value if name == "size" else "|".join(sorted(str(v) for v in value))
            parts.append(f"{name}={shown}")
        return "Accepts(" + ", ".join(parts) + ")"


# -- observation ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Observed:
    backend: Backend
    dtype: str
    device: str
    shape: tuple[int, ...]


def is_torch_tensor(data: Any) -> bool:
    return type(data).__module__.startswith("torch") and hasattr(data, "device")


def observe(data: Any) -> Observed:
    if isinstance(data, np.ndarray):
        return Observed(Backend.NUMPY, _norm_dtype(data.dtype.name) or "", "cpu", data.shape)
    if is_torch_tensor(data):
        return Observed(
            Backend.TORCH, _norm_dtype(str(data.dtype)) or "", str(data.device), tuple(data.shape)
        )
    raise TypeError(f"image data must be a numpy array or torch tensor, got {type(data).__name__}")


def infer_layout(
    shape: tuple[int, ...],
    backend: Backend,
    *,
    color: Color | None = None,
    declared: Layout | None = None,
) -> Layout | None:
    if len(shape) == 2:
        return Layout.HW
    if len(shape) != 3:
        return None
    channels = CHANNELS[color] if color is not None else None
    if declared in (Layout.HWC, Layout.CHW):
        axis = -1 if declared is Layout.HWC else 0
        if channels is None or shape[axis] == channels:
            return declared
    if channels is not None:
        return _layout_from_known_channels(shape, channels)
    return _layout_from_ambiguous_channels(shape, backend)


def _layout_from_known_channels(shape: tuple[int, ...], channels: int) -> Layout | None:
    if shape[-1] == channels:
        return Layout.HWC
    if shape[0] == channels:
        return Layout.CHW
    return None


def _layout_from_ambiguous_channels(shape: tuple[int, ...], backend: Backend) -> Layout | None:
    first, last = shape[0] in (1, 3, 4), shape[-1] in (1, 3, 4)
    preferred = Layout.CHW if backend is Backend.TORCH else Layout.HWC
    secondary = Layout.HWC if backend is Backend.TORCH else Layout.CHW
    if first if backend is Backend.TORCH else last:
        return preferred
    if last if backend is Backend.TORCH else first:
        return secondary
    return None


def size_of(shape: tuple[int, ...], layout: Layout | None) -> tuple[int, int] | None:
    if layout is Layout.HW or layout is Layout.HWC:
        return (int(shape[0]), int(shape[1]))
    if layout is Layout.CHW:
        return (int(shape[1]), int(shape[2]))
    return None


def channels_of(shape: tuple[int, ...], layout: Layout | None) -> int | None:
    if layout is Layout.HW:
        return 1
    if layout is Layout.HWC:
        return int(shape[-1])
    if layout is Layout.CHW:
        return int(shape[0])
    return None


_DTYPE_RANGES = {"uint8": Range.U8, "uint16": Range.U16}


def derive_spec(
    data: Any,
    *,
    base: ImageSpec | None = None,
    declared: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> ImageSpec:
    """Build the spec of ``data``.

    Observable fields (backend, dtype, device, layout, size) come from the data. ``declared``
    fields win over ``base``. A ``base`` color or range is only inherited when it is still
    consistent with the data; otherwise it becomes unknown. With ``strict=True``, a *declared*
    color that contradicts the data raises :class:`SpecMismatchError`.
    """
    declared = dict(declared or {})
    obs = observe(data)
    color, layout = _infer_color_layout(obs, base, declared, strict)
    range_ = _infer_range(obs.dtype, base, declared)

    return ImageSpec(
        layout=layout,
        color=color,
        dtype=obs.dtype,
        range=range_,
        device=obs.device,
        backend=obs.backend,
        size=size_of(obs.shape, layout),
    )


def _infer_color_layout(
    obs: Observed, base: ImageSpec | None, declared: Mapping[str, Any], strict: bool
) -> tuple[Color | None, Layout | None]:
    color_declared = "color" in declared
    color = Color(declared["color"]) if declared.get("color") is not None else None
    if not color_declared and base is not None:
        color = base.color

    layout_hint = declared.get("layout", base.layout if base else None)
    layout = infer_layout(
        obs.shape, obs.backend, color=color, declared=Layout(layout_hint) if layout_hint else None
    )
    if color is not None:
        actual = channels_of(obs.shape, layout)
        if layout is None or actual != CHANNELS[color]:
            if color_declared and strict:
                raise SpecMismatchError(
                    f"declared color {color} needs {CHANNELS[color]} channel(s), "
                    f"but the data has shape {obs.shape}",
                    violations=(Violation("color", color, f"shape {obs.shape}"),),
                )
            color = None
            layout = infer_layout(obs.shape, obs.backend)

    return color, layout


def _infer_range(dtype: str, base: ImageSpec | None, declared: Mapping[str, Any]) -> Range | None:
    if "range" in declared:
        range_ = Range.coerce(declared["range"])
    elif dtype in _DTYPE_RANGES:
        range_ = _DTYPE_RANGES[dtype]
    elif base is not None and base.dtype == dtype:
        range_ = base.range
    elif dtype == "bool":
        range_ = Range.ANY
    else:
        range_ = None
    if range_ is not None and range_.kind in ("U8", "U16") and dtype not in _DTYPE_RANGES:
        range_ = None
    return range_


def consistency_violations(data: Any, spec: ImageSpec) -> list[Violation]:
    """Check declared fields against the actual data (used in check mode)."""
    obs = observe(data)
    out: list[Violation] = []
    _check_observed_fields(obs, spec, out)
    _check_observed_color(obs, spec, out)
    _check_observed_range(data, obs, spec, out)
    return out


def _check_observed_fields(obs: Observed, spec: ImageSpec, out: list[Violation]) -> None:
    if spec.dtype is not None and spec.dtype != obs.dtype:
        out.append(Violation("dtype", obs.dtype, spec.dtype, "declared dtype differs from data"))
    if spec.backend is not None and spec.backend != obs.backend:
        out.append(Violation("backend", obs.backend, spec.backend))
    if spec.layout is not None and infer_layout(
        obs.shape, obs.backend, declared=spec.layout
    ) not in (spec.layout,):
        out.append(Violation("layout", f"shape {obs.shape}", spec.layout))


def _check_observed_color(obs: Observed, spec: ImageSpec, out: list[Violation]) -> None:
    if spec.color is not None:
        actual = channels_of(obs.shape, spec.layout or infer_layout(obs.shape, obs.backend))
        if actual != CHANNELS[spec.color]:
            out.append(Violation("color", f"{actual} channel(s)", spec.color))


def _check_observed_range(data: Any, obs: Observed, spec: ImageSpec, out: list[Violation]) -> None:
    if spec.range is not None and obs.backend is Backend.NUMPY and data.size:
        kind = spec.range.kind
        if kind in ("U8", "U16") and obs.dtype != ("uint8" if kind == "U8" else "uint16"):
            out.append(Violation("range", f"dtype {obs.dtype}", kind))
        elif kind == "UNIT":
            lo, hi = float(np.min(data)), float(np.max(data))
            if lo < -1e-6 or hi > 1 + 1e-6:
                out.append(Violation("range", f"values in [{lo:.4g}, {hi:.4g}]", "UNIT [0, 1]"))
