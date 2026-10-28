"""Module-level custom components used by tests (importable, so configs round-trip)."""

from typing import Any

import numpy as np

from vistral.core import vt


@vt.filter(accepts={"color": {"BGR", "GRAY"}, "dtype": "uint8"})
def invert(img: np.ndarray, amount: int = 255) -> np.ndarray:
    return (amount - img).astype(np.uint8)


@vt.filter(accepts={"color": "BGR"}, produces={"color": "GRAY", "layout": "HW"})
def first_channel(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[..., 0])


@vt.postprocess(reads="detections")
def drop_small(det: Any, *, frame: vt.Frame, min_width: float = 1.0) -> Any:
    keep = (det.xyxy[:, 2] - det.xyxy[:, 0]) >= min_width
    return det[keep]


@vt.stage
def tag(frame: vt.Frame, *, label: str = "x") -> vt.Frame:
    frame.artifacts["tag"] = label
    return frame


@vt.stage
def not_a_frame(frame: vt.Frame) -> Any:
    return 1


def is_fail(frame: vt.Frame) -> bool:
    return frame.artifacts.get("tag") == "fail"


class Brighten(vt.Filter):
    accepts = vt.Accepts(dtype="uint8")

    def __init__(self, delta: int = 10, *, clip: bool = True) -> None:
        self.delta = delta
        self.clip = clip

    def apply(self, data: np.ndarray) -> np.ndarray:
        out = data.astype(np.int16) + self.delta
        return np.clip(out, 0, 255).astype(np.uint8) if self.clip else out.astype(np.uint8)


class MeanEncoder(vt.Encoder):
    def encode(self, image: vt.Image, frame: vt.Frame) -> Any:
        return np.asarray(image.data, np.float32).mean(axis=(0, 1))


class SumModel(vt.Model):
    def predict(self, inputs: Any) -> Any:
        return float(np.sum(inputs))

    def predict_batch(self, inputs: list[Any]) -> list[Any]:
        return [float(np.sum(i)) for i in inputs]


class ScoreDecoder(vt.Decoder):
    writes = "score"

    def decode(self, raw: Any, frame: vt.Frame) -> Any:
        return vt.Text(f"{raw:.1f}")

    def provenance(self) -> dict[str, Any]:
        return {"model_id": "sum-model", "model_revision": "r1"}


class Collect(vt.Sink):
    def __init__(self) -> None:
        self.frames: list[vt.Frame] = []
        self.closed = 0

    def write(self, frame: vt.Frame) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed += 1


class Unserializable(vt.Stage):
    def __init__(self, thing: object) -> None:
        self.thing = thing

    def process(self, frame: vt.Frame) -> vt.Frame:
        return frame


def make_number() -> int:
    return 1
