"""``Frame``: the unit that flows through a pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vistral.core.image import Image
from vistral.core.spec import ImageSpec


@dataclass(frozen=True)
class SourceInfo:
    """Where a frame came from. ``stream_id`` groups frames of one stream (video, camera)."""

    uri: str | None = None
    stem: str | None = None
    frame_index: int | None = None
    timestamp_ns: int | None = None
    stream_id: str | None = None
    camera_id: str | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> SourceInfo:
        p = Path(path)
        return cls(uri=str(p), stem=p.stem)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class StageFailure:
    """A failure recorded on a frame when a pipeline runs with ``on_error="skip"|"collect"``."""

    stage: str
    index: int | str
    error: BaseException

    @property
    def message(self) -> str:
        return str(self.error)


@dataclass(eq=False)
class Frame:
    """Current image, original image, results, model intermediates, and source info.

    Pipelines hand each stage a :meth:`fork` (fresh dicts, shared arrays), so a stage may
    mutate the frame it receives without affecting upstream frames.
    """

    image: Image
    original: Image | None = None
    results: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    source: SourceInfo = field(default_factory=SourceInfo)
    errors: list[StageFailure] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.original is None:
            self.original = self.image

    def fork(self) -> Frame:
        return Frame(
            self.image,
            self.original,
            dict(self.results),
            dict(self.artifacts),
            self.source,
            list(self.errors),
        )

    @property
    def failed(self) -> bool:
        return bool(self.errors)

    @classmethod
    def coerce(cls, value: Any, *, spec: ImageSpec | None = None) -> Frame:
        """Build a frame from a Frame, Image, array, PIL image, or path."""
        if isinstance(value, Frame):
            return value
        if isinstance(value, Image):
            return cls(value)
        if isinstance(value, (str, Path)):
            from vistral.io.image import read_image

            return cls(read_image(value, spec=spec), source=SourceInfo.from_path(value))
        if isinstance(value, np.ndarray) or hasattr(value, "device"):
            return cls(Image.from_array(value, spec=spec))
        if hasattr(value, "mode") and hasattr(value, "getbands"):
            return cls(Image.from_pil(value))
        raise TypeError(
            f"cannot make a Frame from {type(value).__name__}; "
            "pass a path, numpy array, PIL image, vistral Image or Frame"
        )

    def __repr__(self) -> str:
        res = ", ".join(self.results) or "-"
        err = f", errors={len(self.errors)}" if self.errors else ""
        return f"Frame({self.image!r}, results=[{res}]{err})"
