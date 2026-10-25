"""Geometric transforms recorded by stages so results can be mapped back to source pixels.

Coordinates are continuous and corner-based: pixel ``i`` covers ``[i, i + 1)``, origin at the
top-left corner, x to the right, y down. Matrices are 3x3 and map *source* to *destination*.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Matrix = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def _as_matrix(value: Any) -> Matrix:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"transform matrix must be 3x3, got shape {arr.shape}")
    return tuple(tuple(float(v) for v in row) for row in arr)  # type: ignore[return-value]


@dataclass(frozen=True)
class GeoTransform:
    """A recorded geometric transform from ``src_size`` to ``dst_size`` (both ``(H, W)``)."""

    name: str
    matrix: Matrix
    src_size: tuple[int, int]
    dst_size: tuple[int, int]
    producer: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _as_matrix(self.matrix))
        object.__setattr__(self, "src_size", tuple(int(v) for v in self.src_size))
        object.__setattr__(self, "dst_size", tuple(int(v) for v in self.dst_size))

    @property
    def array(self) -> np.ndarray:
        return np.array(self.matrix, dtype=np.float64)

    def inverse(self) -> GeoTransform:
        return GeoTransform(
            f"inverse({self.name})",
            _as_matrix(np.linalg.inv(self.array)),
            self.dst_size,
            self.src_size,
            self.producer,
        )

    # -- common transforms --------------------------------------------------------------------
    @classmethod
    def scale(
        cls,
        sx: float,
        sy: float,
        src_size: tuple[int, int],
        dst_size: tuple[int, int],
        *,
        name: str = "scale",
        producer: str | None = None,
    ) -> GeoTransform:
        return cls(
            name, _as_matrix([[sx, 0, 0], [0, sy, 0], [0, 0, 1]]), src_size, dst_size, producer
        )

    @classmethod
    def resize(
        cls, src_size: tuple[int, int], dst_size: tuple[int, int], *, producer: str | None = None
    ) -> GeoTransform:
        (sh, sw), (dh, dw) = src_size, dst_size
        return cls.scale(dw / sw, dh / sh, src_size, dst_size, name="resize", producer=producer)

    @classmethod
    def translate(
        cls,
        tx: float,
        ty: float,
        src_size: tuple[int, int],
        dst_size: tuple[int, int],
        *,
        name: str = "translate",
        producer: str | None = None,
    ) -> GeoTransform:
        return cls(
            name, _as_matrix([[1, 0, tx], [0, 1, ty], [0, 0, 1]]), src_size, dst_size, producer
        )

    @classmethod
    def crop(
        cls,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        src_size: tuple[int, int],
        *,
        producer: str | None = None,
    ) -> GeoTransform:
        return cls.translate(-x1, -y1, src_size, (y2 - y1, x2 - x1), name="crop", producer=producer)

    @classmethod
    def pad(
        cls,
        top: int,
        bottom: int,
        left: int,
        right: int,
        src_size: tuple[int, int],
        *,
        producer: str | None = None,
    ) -> GeoTransform:
        h, w = src_size
        return cls.translate(
            left, top, src_size, (h + top + bottom, w + left + right), name="pad", producer=producer
        )

    @classmethod
    def flip(
        cls, src_size: tuple[int, int], *, horizontal: bool = True, producer: str | None = None
    ) -> GeoTransform:
        h, w = src_size
        m = [[-1, 0, w], [0, 1, 0], [0, 0, 1]] if horizontal else [[1, 0, 0], [0, -1, h], [0, 0, 1]]
        return cls("flip", _as_matrix(m), src_size, src_size, producer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "matrix": [list(r) for r in self.matrix],
            "src_size": list(self.src_size),
            "dst_size": list(self.dst_size),
            "producer": self.producer,
        }


def compose(transforms: Sequence[GeoTransform]) -> np.ndarray:
    """Matrix of applying ``transforms`` in order (first element first)."""
    out = np.eye(3)
    for t in transforms:
        out = t.array @ out
    return out


def to_source_matrix(transforms: Sequence[GeoTransform]) -> np.ndarray:
    """Matrix mapping coordinates after ``transforms`` back to the source image."""
    return np.linalg.inv(compose(transforms))


def map_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 matrix to points of shape ``(..., 2)``."""
    pts = np.asarray(points, dtype=np.float64)
    flat = pts.reshape(-1, 2)
    homo = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1) @ np.asarray(matrix).T
    out = homo[:, :2] / homo[:, 2:3]
    return out.reshape(pts.shape)


def map_boxes(matrix: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Map ``(N, 4)`` xyxy boxes; returns the axis-aligned box around the mapped corners."""
    boxes = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return boxes.copy()
    x1, y1, x2, y2 = boxes.T
    corners = np.stack(
        [
            np.stack([x1, y1], 1),
            np.stack([x2, y1], 1),
            np.stack([x2, y2], 1),
            np.stack([x1, y2], 1),
        ],
        axis=1,
    )
    mapped = map_points(matrix, corners)
    return np.concatenate([mapped.min(axis=1), mapped.max(axis=1)], axis=1)
