"""Result types and the provenance/coordinate-space metadata convention.

Detections and instance masks are ``supervision.Detections`` (D-004); Vistral metadata lives in
``detections.metadata["vistral"]``. Other tasks use the dataclasses below. This module never
imports supervision; it recognizes supervision objects by their attributes.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from vistral.core.errors import ConversionError
from vistral.core.geometry import GeoTransform, map_boxes, map_points, warp_array

if TYPE_CHECKING:
    from vistral.core.frame import Frame

META_KEY = "vistral"


@dataclass(frozen=True)
class ResultMeta:
    """Provenance and coordinate space of a result.

    ``coord_space`` is ``"source"`` or ``"stage:<name>"``; ``transforms`` are the geometric
    transforms between the source image and the image the result's coordinates refer to.
    """

    producer: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    coord_space: str = "source"
    image_size: tuple[int, int] | None = None
    transforms: tuple[GeoTransform, ...] = ()
    created_ns: int | None = None
    recipe_hash: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def replace(self, **changes: Any) -> ResultMeta:
        values = {item.name: getattr(self, item.name) for item in dataclasses.fields(self)}
        values.update(changes)
        return ResultMeta(**values)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "producer": self.producer,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "coord_space": self.coord_space,
            "image_size": list(self.image_size) if self.image_size else None,
            "transforms": [t.to_dict() for t in self.transforms],
            "created_ns": self.created_ns,
            "recipe_hash": self.recipe_hash,
        }
        if self.extra:
            out["extra"] = dict(self.extra)
        return {k: v for k, v in out.items() if v not in (None, [])}


def is_sv_detections(obj: Any) -> bool:
    return hasattr(obj, "xyxy") and isinstance(getattr(obj, "metadata", None), dict)


def is_sv_keypoints(obj: Any) -> bool:
    return hasattr(obj, "xy") and hasattr(obj, "keypoint_confidence")


def get_meta(result: Any) -> ResultMeta | None:
    if isinstance(result, Result):
        return result.meta
    if is_sv_detections(result):
        meta = result.metadata.get(META_KEY)
        return meta if isinstance(meta, ResultMeta) else None
    meta = getattr(result, "meta", None)
    return meta if isinstance(meta, ResultMeta) else None


def set_meta(result: Any, meta: ResultMeta) -> Any:
    """Return ``result`` with ``meta`` attached. Never mutates ``result``."""
    if isinstance(result, Result):
        return dataclasses.replace(result, meta=meta)
    if is_sv_detections(result):
        out = copy.copy(result)
        out.metadata = {**result.metadata, META_KEY: meta}
        return out
    return result


def coord_space_for(transforms: tuple[GeoTransform, ...]) -> str:
    if not transforms:
        return "source"
    last = transforms[-1]
    return f"stage:{last.producer or last.name}"


def stamp(
    result: Any,
    frame: Frame,
    *,
    producer: str,
    model_id: str | None = None,
    model_revision: str | None = None,
    **extra: Any,
) -> Any:
    """Attach provenance and the coordinate space of ``frame.image`` to ``result``."""
    transforms = frame.image.transforms
    meta = ResultMeta(
        producer=producer,
        model_id=model_id,
        model_revision=model_revision,
        coord_space=coord_space_for(transforms),
        image_size=frame.image.spec.size,
        transforms=transforms,
        created_ns=time.time_ns(),
        extra=extra,
    )
    return set_meta(result, meta)


# -- Vistral result types --------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Result:
    """Base class for Vistral results. Subclasses add their fields before ``meta``."""

    meta: ResultMeta = field(default_factory=ResultMeta, kw_only=True)

    def transform_coords(self, matrix: np.ndarray, size: tuple[int, int]) -> Self:
        """Return this result with coordinates mapped by ``matrix`` into an image of ``size``."""
        del matrix, size  # Base results have no coordinates to transform.
        return self

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": type(self).__name__}
        for f in dataclasses.fields(self):
            if f.name == "meta":
                continue
            out[f.name] = _jsonable(getattr(self, f.name))
        out["meta"] = self.meta.to_dict()
        return out

    def summary(self) -> str:
        return type(self).__name__


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size > 10_000:
            return {"shape": list(value.shape), "dtype": value.dtype.name}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, StrEnum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return value


@dataclass(frozen=True, eq=False)
class Classification(Result):
    """Top-k classes. ``sv.Classifications`` has no labels or metadata, so Vistral has its own."""

    class_ids: np.ndarray
    scores: np.ndarray
    labels: tuple[str, ...] | None = None

    @property
    def top(self) -> tuple[int, float, str | None]:
        i = int(np.argmax(self.scores))
        label = self.labels[i] if self.labels else None
        return int(self.class_ids[i]), float(self.scores[i]), label

    def to_supervision(self) -> Any:
        import supervision as sv

        return sv.Classifications(class_id=np.asarray(self.class_ids), confidence=self.scores)

    def summary(self) -> str:
        _, score, label = self.top if len(self.scores) else (None, 0.0, None)
        return f"Classification(top={label!r}, score={score:.3f})"


@dataclass(frozen=True, eq=False)
class Keypoints(Result):
    """Keypoints ``xy`` of shape ``(N, K, 2)``. Convert with :meth:`to_supervision`."""

    xy: np.ndarray
    confidence: np.ndarray | None = None
    class_ids: np.ndarray | None = None

    def transform_coords(self, matrix: np.ndarray, size: tuple[int, int]) -> Self:
        return dataclasses.replace(self, xy=map_points(matrix, self.xy).astype(np.float32))

    def to_supervision(self) -> Any:
        import supervision as sv

        return sv.KeyPoints(
            xy=np.asarray(self.xy, dtype=np.float32),
            class_id=self.class_ids,
            keypoint_confidence=self.confidence,
        )

    def summary(self) -> str:
        return f"Keypoints(n={len(self.xy)})"


def _warp_label_map(label_map: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    # Nearest-neighbour keeps label ids and mask values exact; warp_array handles the
    # corner- vs center-based pixel convention (D-009, risk R14).
    return warp_array(label_map, matrix, size, interpolation="nearest")


@dataclass(frozen=True, eq=False)
class SemanticMask(Result):
    """Per-pixel class ids, shape ``(H, W)``."""

    label_map: np.ndarray
    id2label: Mapping[int, str] | None = None

    def transform_coords(self, matrix: np.ndarray, size: tuple[int, int]) -> Self:
        return dataclasses.replace(self, label_map=_warp_label_map(self.label_map, matrix, size))

    def summary(self) -> str:
        return f"SemanticMask({self.label_map.shape[0]}x{self.label_map.shape[1]})"


@dataclass(frozen=True, eq=False)
class Embedding(Result):
    vector: np.ndarray
    model_id: str | None = None
    normalized: bool = False

    def summary(self) -> str:
        return f"Embedding(dim={self.vector.shape[-1]})"


@dataclass(frozen=True, eq=False)
class Text(Result):
    text: str
    scores: np.ndarray | None = None

    def summary(self) -> str:
        return f"Text({self.text[:24]!r})"


@dataclass(frozen=True, eq=False)
class Contours(Result):
    """Contours as ``(N_i, 2)`` point arrays (x, y)."""

    points: tuple[np.ndarray, ...]
    hierarchy: np.ndarray | None = None

    def transform_coords(self, matrix: np.ndarray, size: tuple[int, int]) -> Self:
        pts = tuple(map_points(matrix, p.reshape(-1, 2)) for p in self.points)
        return dataclasses.replace(self, points=pts)

    def summary(self) -> str:
        return f"Contours(n={len(self.points)})"


@dataclass(frozen=True)
class Measurement:
    name: str
    value: float
    unit: str = "px"
    lo: float | None = None
    hi: float | None = None
    target: int | None = None

    @property
    def ok(self) -> bool | None:
        if self.lo is None and self.hi is None:
            return None
        return (self.lo is None or self.value >= self.lo) and (
            self.hi is None or self.value <= self.hi
        )


@dataclass(frozen=True, eq=False)
class Measurements(Result):
    items: tuple[Measurement, ...]

    @property
    def all_ok(self) -> bool:
        return all(m.ok is not False for m in self.items)

    def summary(self) -> str:
        return f"Measurements(n={len(self.items)}, ok={self.all_ok})"


class Status(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class RuleResult:
    rule: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, eq=False)
class Verdict(Result):
    status: Status
    reasons: tuple[str, ...] = ()
    rule_results: tuple[RuleResult, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", Status(self.status))

    def summary(self) -> str:
        return f"Verdict({self.status})"


# -- generic operations ----------------------------------------------------------------------


def transform_result(result: Any, matrix: np.ndarray, size: tuple[int, int]) -> Any:
    """Map the coordinates of ``result`` with ``matrix`` into an image of ``size`` (H, W)."""
    if isinstance(result, Result):
        return result.transform_coords(matrix, size)
    if is_sv_detections(result):
        out = copy.copy(result)
        out.xyxy = map_boxes(matrix, result.xyxy).astype(np.float32)
        if result.mask is not None:
            if not isinstance(result.mask, np.ndarray):
                raise ConversionError("compact masks cannot be remapped; use dense masks")
            out.mask = (
                np.stack([_warp_label_map(m, matrix, size) for m in result.mask])
                if len(result.mask)
                else np.zeros((0, *size), dtype=bool)
            )
        return out
    if is_sv_keypoints(result):
        out = copy.copy(result)
        out.xy = map_points(matrix, result.xy).astype(np.float32)
        return out
    raise ConversionError(
        f"don't know how to map the coordinates of {type(result).__name__}; "
        "subclass vt.Result and implement transform_coords()"
    )


def summarize(result: Any) -> str:
    if isinstance(result, Result):
        return result.summary()
    if is_sv_detections(result):
        return f"Detections(n={len(result)})"
    return type(result).__name__


def result_to_dict(result: Any) -> dict[str, Any]:
    """JSON-friendly view of any result (Vistral types and ``sv.Detections``)."""
    if isinstance(result, Result):
        return result.to_dict()
    if is_sv_detections(result):
        meta = get_meta(result)
        out: dict[str, Any] = {
            "type": "Detections",
            "xyxy": _jsonable(result.xyxy),
            "confidence": _jsonable(result.confidence),
            "class_id": _jsonable(result.class_id),
            "tracker_id": _jsonable(result.tracker_id),
            "data": _jsonable(dict(result.data)),
        }
        if result.mask is not None:
            out["mask"] = {"shape": list(np.asarray(result.mask).shape)}
        if meta is not None:
            out["meta"] = meta.to_dict()
        return out
    return {"type": type(result).__name__, "value": _jsonable(result)}
