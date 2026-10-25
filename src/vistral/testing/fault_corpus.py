"""Fault-injection corpus: common integration bugs that hand-written glue lets through silently.

Each case builds a faulty pipeline and states how Vistral must catch it: during
``Pipeline.validate`` (before any data is processed) or at runtime (check mode). See
docs/BENCHMARK_PLAN.md section 6. HF-specific cases are added in M3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from vistral.core.errors import CoordinateSpaceError, SpecMismatchError
from vistral.core.image import Image
from vistral.core.ops import ToSourceCoords
from vistral.core.pipeline import Pipeline
from vistral.testing import fixtures as fx
from vistral.testing import stubs


@dataclass(frozen=True)
class FaultCase:
    name: str
    description: str
    build: Callable[[], Pipeline]
    make_input: Callable[[], Image]
    detect: Literal["validate", "runtime"]
    error: type[Exception] = SpecMismatchError
    auto_fixable: bool = False
    """``validate(auto_convert=True)`` repairs it with lossless conversions only."""
    fixed: Callable[[], Pipeline] | None = None
    """A correctly written pipeline that should run without errors."""


def cases() -> list[FaultCase]:
    return [
        FaultCase(
            "bgr_to_rgb_model",
            "OpenCV BGR image fed to a model trained on RGB (colors silently swapped)",
            lambda: Pipeline([stubs.RGBModelStub()]),
            fx.bgr_u8,
            "validate",
            auto_fixable=True,
        ),
        FaultCase(
            "hwc_to_chw_model",
            "channels-last image fed to a channels-first model",
            lambda: Pipeline([stubs.CHWUnitModelStub()]),
            fx.rgb_unit,
            "validate",
            auto_fixable=True,
        ),
        FaultCase(
            "u8_to_unit_model",
            "0-255 uint8 fed to a model expecting float 0-1",
            lambda: Pipeline([stubs.CHWUnitModelStub()]),
            fx.rgb_u8,
            "validate",
            auto_fixable=True,
        ),
        FaultCase(
            "unit_to_u8_encoder",
            "float 0-1 image fed to an encoder that rescales by 1/255 again",
            lambda: Pipeline([stubs.EncoderStub(), stubs.PixelValuesModelStub()]),
            fx.rgb_unit,
            "validate",
        ),
        FaultCase(
            "double_normalization",
            "image normalized by a filter, then normalized again by the encoder",
            lambda: Pipeline(
                [stubs.NormalizeStub(), stubs.EncoderStub(), stubs.PixelValuesModelStub()]
            ),
            fx.rgb_u8,
            "validate",
            fixed=lambda: Pipeline([stubs.EncoderStub(), stubs.PixelValuesModelStub()]),
        ),
        FaultCase(
            "size_swapped",
            "a (width, height) size passed where (height, width) is expected",
            lambda: Pipeline([stubs.FixedSizeModelStub(input_size=(64, 48))]),
            lambda: fx.rgb_u8(48, 64),
            "validate",
            fixed=lambda: Pipeline([stubs.FixedSizeModelStub(input_size=(48, 64))]),
        ),
        FaultCase(
            "gray_to_color_model",
            "single-channel image fed to a 3-channel RGB model",
            lambda: Pipeline([stubs.RGBModelStub()]),
            fx.gray_u8,
            "validate",
            auto_fixable=True,
        ),
        FaultCase(
            "u16_truncated",
            "16-bit industrial image fed to an 8-bit stage (bit depth silently lost)",
            lambda: Pipeline([stubs.RGBModelStub()]),
            fx.gray_u16,
            "validate",
        ),
        FaultCase(
            "alpha_channel",
            "BGRA image (PNG with alpha) fed to an RGB model",
            lambda: Pipeline([stubs.RGBModelStub()]),
            fx.bgra_u8,
            "validate",
        ),
        FaultCase(
            "mask_to_color_model",
            "binary mask fed to a color model",
            lambda: Pipeline([stubs.RGBModelStub()]),
            fx.mask,
            "validate",
        ),
        FaultCase(
            "boxes_not_mapped_back",
            "detections from a resized image measured as if they were in source pixels",
            lambda: Pipeline(
                [
                    stubs.ResizeStub(size=(24, 32)),
                    stubs.BoxDetectorStub([[2, 2, 10, 10]]),
                    stubs.MeasureStub(),
                ]
            ),
            fx.bgr_u8,
            "validate",
            error=CoordinateSpaceError,
            fixed=lambda: Pipeline(
                [
                    stubs.ResizeStub(size=(24, 32)),
                    stubs.BoxDetectorStub([[2, 2, 10, 10]]),
                    ToSourceCoords(),
                    stubs.MeasureStub(),
                ]
            ),
        ),
        FaultCase(
            "boxes_after_crop",
            "detections from an ROI crop measured as if they were in source pixels",
            lambda: Pipeline(
                [
                    stubs.CropStub(box=(8, 4, 40, 36)),
                    stubs.BoxDetectorStub([[0, 0, 4, 4]]),
                    stubs.MeasureStub(),
                ]
            ),
            fx.bgr_u8,
            "validate",
            error=CoordinateSpaceError,
        ),
        FaultCase(
            "model_without_encoder",
            "model reading encoder outputs, but no encoder in the pipeline",
            lambda: Pipeline([stubs.PixelValuesModelStub()]),
            fx.rgb_u8,
            "validate",
            fixed=lambda: Pipeline([stubs.EncoderStub(), stubs.PixelValuesModelStub()]),
        ),
        FaultCase(
            "undeclared_dtype_change",
            "custom stage declares uint8 output but returns float32",
            lambda: Pipeline([stubs.LyingFilterStub(), stubs.RGBModelStub()], check=True),
            fx.rgb_u8,
            "runtime",
        ),
        FaultCase(
            "unknown_color_upstream",
            "an unannotated stage hides the color order from a model that needs RGB",
            lambda: Pipeline([stubs.OpaqueFilterStub(), stubs.RGBModelStub()]),
            fx.rgb_u8,
            "runtime",
        ),
    ]
