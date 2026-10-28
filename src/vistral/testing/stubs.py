"""NumPy-only test doubles that mimic the contracts of real components.

They let the fault corpus and contract tests exercise validation without OpenCV built-ins
(M2) or Hugging Face models (M3). Not intended for production use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from vistral.core.frame import Frame
from vistral.core.geometry import GeoTransform
from vistral.core.image import Image
from vistral.core.results import Measurement, Measurements, stamp
from vistral.core.spec import Accepts, Color, ImageSpec, Layout, Range
from vistral.core.stage import Encoder, Filter, Model, Postprocess, Role, Stage

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def rasterize_box(box: Sequence[float], size: tuple[int, int]) -> np.ndarray:
    """Boolean mask of ``box`` (xyxy) in an image of ``size`` ``(H, W)``.

    Uses Vistral's corner convention (D-009): pixel ``i`` is inside ``[a, b)`` when its
    center ``i + 0.5`` is. This is the ground truth for mask-warping tests.
    """
    h, w = size
    x1, y1, x2, y2 = box
    xs, ys = np.arange(w) + 0.5, np.arange(h) + 0.5
    rows = (ys >= y1) & (ys < y2)
    cols = (xs >= x1) & (xs < x2)
    return rows[:, None] & cols[None, :]


class RGBModelStub(Model):
    """Like most HF/torchvision models: RGB uint8 HWC in, per-channel means out."""

    reads = "image"
    accepts = Accepts(color=Color.RGB, dtype="uint8", layout=Layout.HWC, backend="numpy")

    def predict(self, inputs: Any) -> Any:
        return inputs.reshape(-1, 3).mean(axis=0)


class CHWUnitModelStub(Model):
    """Like an ONNX export: RGB float32 [0, 1] CHW in."""

    reads = "image"
    accepts = Accepts(color=Color.RGB, dtype="float32", range="UNIT", layout=Layout.CHW)

    def predict(self, inputs: Any) -> Any:
        return inputs.mean(axis=(1, 2))


class FixedSizeModelStub(Model):
    """A model that needs an exact input size ``(height, width)``."""

    reads = "image"

    def __init__(self, input_size: Sequence[int] = (48, 64)) -> None:
        self.input_size = tuple(input_size)
        self.accepts = Accepts(size=self.input_size)

    def predict(self, inputs: Any) -> Any:
        return float(np.mean(inputs))


class NormalizeStub(Filter):
    """ImageNet normalization of an RGB uint8 HWC image (what an encoder does)."""

    accepts = Accepts(color=Color.RGB, dtype="uint8", range="U8")

    def produces(self, spec: ImageSpec) -> ImageSpec:
        return spec.replace(dtype="float32", range=Range.normalized(IMAGENET_MEAN, IMAGENET_STD))

    def apply(self, data: Any) -> Any:
        mean, std = np.array(IMAGENET_MEAN, np.float32), np.array(IMAGENET_STD, np.float32)
        return ((data.astype(np.float32) / 255.0) - mean) / std


class EncoderStub(Encoder):
    """Rescales and normalizes RGB uint8 (like HF image processors with do_rescale=True)."""

    accepts = Accepts(color=Color.RGB, dtype="uint8", range="U8", layout=Layout.HWC)

    def encode(self, image: Image, frame: Frame) -> Any:
        mean, std = np.array(IMAGENET_MEAN, np.float32), np.array(IMAGENET_STD, np.float32)
        x = ((image.data.astype(np.float32) / 255.0) - mean) / std
        return {"pixel_values": x.transpose(2, 0, 1)[None]}


class PixelValuesModelStub(Model):
    def predict(self, inputs: Any) -> Any:
        return {"logits": inputs["pixel_values"].mean(axis=(2, 3))}


class ResizeStub(Filter):
    """Nearest-neighbour resize to ``size=(H, W)`` that records its transform."""

    changes_geometry = True

    def __init__(self, size: Sequence[int]) -> None:
        self.size = (int(size[0]), int(size[1]))

    def produces(self, spec: ImageSpec) -> ImageSpec:
        return spec.replace(size=self.size)

    def process(self, frame: Frame) -> Frame:
        data = frame.image.data
        (sh, sw), (dh, dw) = data.shape[:2], self.size
        ys = np.minimum(((np.arange(dh) + 0.5) * sh / dh).astype(int), sh - 1)
        xs = np.minimum(((np.arange(dw) + 0.5) * sw / dw).astype(int), sw - 1)
        out = data[ys][:, xs]
        t = GeoTransform.resize((sh, sw), (dh, dw), producer=self.name)
        frame.image = frame.image.with_transform(out, t)
        return frame


class CropStub(Filter):
    """Crop ``box=(x1, y1, x2, y2)`` (a view) and record the transform."""

    changes_geometry = True

    def __init__(self, box: Sequence[int]) -> None:
        self.box = tuple(int(v) for v in box)

    def produces(self, spec: ImageSpec) -> ImageSpec:
        x1, y1, x2, y2 = self.box
        return spec.replace(size=(y2 - y1, x2 - x1))

    def process(self, frame: Frame) -> Frame:
        x1, y1, x2, y2 = self.box
        t = GeoTransform.crop(x1, y1, x2, y2, frame.image.spec.size, producer=self.name)  # type: ignore[arg-type]
        frame.image = frame.image.with_transform(frame.image.data[y1:y2, x1:x2], t)
        return frame


class BoxDetectorStub(Stage):
    """Emits fixed boxes given in *current image* coordinates as ``sv.Detections``."""

    role = Role.DECODER
    writes = "detections"

    def __init__(
        self,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str] | None = None,
        *,
        masks: bool = False,
    ) -> None:
        self.boxes = [list(map(float, b)) for b in boxes]
        self.labels = list(labels) if labels is not None else None
        self.masks = masks

    def process(self, frame: Frame) -> Frame:
        import supervision as sv

        n = len(self.boxes)
        det = sv.Detections(
            xyxy=np.array(self.boxes, dtype=np.float32).reshape(-1, 4),
            confidence=np.full(n, 0.9, dtype=np.float32),
            class_id=np.zeros(n, dtype=int),
        )
        if self.masks:
            size = frame.image.spec.size
            assert size is not None
            det.mask = (
                np.stack([rasterize_box(b, size) for b in self.boxes])
                if n
                else (np.zeros((0, *size), dtype=bool))
            )
        if self.labels is not None:
            det.data["class_name"] = np.array(self.labels)
        frame.results[self.writes or "detections"] = stamp(det, frame, producer=self.name)
        return frame

    def writes_results(self) -> tuple[str, ...]:
        return (self.writes or "detections",)


class MeasureStub(Postprocess):
    """Box widths in pixels; requires source coordinates (like real measurement)."""

    writes = "measurements"
    requires_source_coords = ("detections",)

    def apply(self, result: Any, frame: Frame) -> Any:
        widths = result.xyxy[:, 2] - result.xyxy[:, 0]
        return Measurements(tuple(Measurement("width", float(w)) for w in widths))


class OpaqueFilterStub(Filter):
    """Declares nothing about its output color (simulates an unannotated custom stage)."""

    def produces(self, spec: ImageSpec) -> ImageSpec:
        return spec.replace(color=None)

    def apply(self, data: Any) -> Any:
        return data.copy()


class LyingFilterStub(Filter):
    """Declares uint8 output but returns float32 (caught in check mode)."""

    def produces(self, spec: ImageSpec) -> ImageSpec:
        return spec.replace(dtype="uint8")

    def apply(self, data: Any) -> Any:
        return data.astype(np.float32)


class CountingStage(Stage):
    """Counts lifecycle calls; for lifecycle and batching tests."""

    def __init__(self) -> None:
        self.setups = self.closes = self.resets = self.calls = self.batch_calls = 0

    def setup(self) -> None:
        self.setups += 1

    def close(self) -> None:
        self.closes += 1

    def reset(self) -> None:
        self.resets += 1

    def process(self, frame: Frame) -> Frame:
        self.calls += 1
        return frame

    def process_batch(self, frames: list[Frame]) -> list[Frame]:
        self.batch_calls += 1
        self.calls += len(frames)
        return frames


class FailingStage(Stage):
    """Raises on frames whose ``source.frame_index`` is in ``fail_on``."""

    def __init__(self, fail_on: Sequence[int] = (0,)) -> None:
        self.fail_on = tuple(fail_on)

    def process(self, frame: Frame) -> Frame:
        if frame.source.frame_index in self.fail_on:
            raise RuntimeError(f"boom on frame {frame.source.frame_index}")
        return frame
