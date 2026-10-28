"""Ground-truth tests for Vistral's corner-based pixel convention vs OpenCV's center sampling.

Regression tests for the half-pixel mask drift found in M1 (risk R14). Expected masks are
built by rasterizing the analytically mapped box, never by OpenCV itself.
"""

import cv2
import numpy as np
import pytest

from vistral.core import vt
from vistral.core.geometry import map_boxes, pixel_center_matrix, warp_array
from vistral.testing import fixtures as fx
from vistral.testing import stubs
from vistral.testing.stubs import rasterize_box

SRC = (48, 64)  # (H, W)
BOX = [12, 8, 28, 24]

TRANSFORMS = {
    "upscale x2": vt.GeoTransform.scale(2, 2, SRC, (96, 128)),
    "upscale x3": vt.GeoTransform.scale(3, 3, SRC, (144, 192)),
    "upscale x1.5": vt.GeoTransform.scale(1.5, 1.5, SRC, (72, 96)),
    "downscale x0.5": vt.GeoTransform.scale(0.5, 0.5, SRC, (24, 32)),
    "anisotropic": vt.GeoTransform.scale(2, 0.5, SRC, (24, 128)),
    "crop": vt.GeoTransform.crop(8, 4, 40, 36, SRC),
    "pad": vt.GeoTransform.pad(3, 3, 5, 5, SRC),
    "hflip": vt.GeoTransform.flip(SRC),
    "vflip": vt.GeoTransform.flip(SRC, horizontal=False),
    "rot90": vt.GeoTransform("rot90", [[0, -1, 48], [1, 0, 0], [0, 0, 1]], SRC, (64, 48)),
    "rot180": vt.GeoTransform("rot180", [[-1, 0, 64], [0, -1, 48], [0, 0, 1]], SRC, SRC),
}


def _expected(t: vt.GeoTransform) -> np.ndarray:
    return rasterize_box(map_boxes(t.array, [BOX])[0], t.dst_size)


@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_nearest_warp_matches_ground_truth(name):
    t = TRANSFORMS[name]
    got = warp_array(rasterize_box(BOX, t.src_size), t.array, t.dst_size)
    assert got.dtype == bool and got.shape == t.dst_size
    assert int((got ^ _expected(t)).sum()) == 0


@pytest.mark.parametrize("name", ["upscale x2", "upscale x3", "hflip", "rot90"])
def test_uncorrected_opencv_warp_would_drift(name):
    """Guards the test above: the naive call really is wrong for these transforms."""
    t = TRANSFORMS[name]
    h, w = t.dst_size
    naive = cv2.warpPerspective(
        rasterize_box(BOX, t.src_size).astype(np.uint8), t.array, (w, h), flags=cv2.INTER_NEAREST
    ).astype(bool)
    assert int((naive ^ _expected(t)).sum()) > 0


def test_pixel_center_matrix():
    np.testing.assert_allclose(pixel_center_matrix(np.eye(3)), np.eye(3))
    translate = vt.GeoTransform.translate(3, -2, SRC, SRC).array
    np.testing.assert_allclose(pixel_center_matrix(translate), translate)
    scale = vt.GeoTransform.scale(2, 2, SRC, (96, 128)).array
    np.testing.assert_allclose(pixel_center_matrix(scale), [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]])


def test_linear_warp_samples_continuous_coordinates():
    # Ramp whose pixel i holds value i (sampled at its center i + 0.5). Upscaling x2, the
    # destination center p + 0.5 maps to source coordinate (p + 0.5) / 2, whose value is
    # (p + 0.5) / 2 - 0.5.
    ramp = np.tile(np.arange(64, dtype=np.float32), (48, 1))
    t = TRANSFORMS["upscale x2"]
    out = warp_array(ramp, t.array, t.dst_size, interpolation="linear")
    p = np.arange(2, 124)
    np.testing.assert_allclose(out[40, p], (p + 0.5) / 2 - 0.5, atol=1 / 32)


def test_warp_array_validates_interpolation():
    with pytest.raises(ValueError, match="interpolation"):
        warp_array(np.zeros((2, 2), np.uint8), np.eye(3), (2, 2), interpolation="area")


def test_label_maps_keep_exact_ids():
    labels = np.zeros(SRC, np.int32)
    labels[rasterize_box(BOX, SRC)] = 7
    t = TRANSFORMS["rot90"]
    out = vt.SemanticMask(labels).transform_coords(t.array, t.dst_size).label_map
    assert set(np.unique(out)) == {0, 7}
    np.testing.assert_array_equal(out == 7, _expected(t))


def test_resize_stub_sampling_follows_convention():
    mask = vt.Image.from_array(rasterize_box(BOX, SRC), color="MASK")
    out = stubs.ResizeStub(size=(96, 128))(mask).image.data
    np.testing.assert_array_equal(out, _expected(TRANSFORMS["upscale x2"]))


class FlipH(vt.Filter):
    changes_geometry = True

    def process(self, frame):
        t = vt.GeoTransform.flip(frame.image.spec.size, producer=self.name)
        frame.image = frame.image.with_transform(frame.image.data[:, ::-1], t)
        return frame


@pytest.mark.parametrize(
    ("stages", "box_in_model", "box_in_source"),
    [
        # ROI crop (8,4)-(40,36), upscale x2: model box /2 + offset.
        (
            [stubs.CropStub(box=(8, 4, 40, 36)), stubs.ResizeStub(size=(64, 64))],
            [10, 6, 30, 22],
            [13, 7, 23, 15],
        ),
        # Horizontal flip then upscale x2: x' = 64 - x / 2.
        ([FlipH(), stubs.ResizeStub(size=(96, 128))], [20, 10, 60, 30], [34, 5, 54, 15]),
        # Downscale x0.5 (no sampling ties): model box x2.
        ([stubs.ResizeStub(size=(24, 32))], [3, 2, 9, 6], [6, 4, 18, 12]),
    ],
    ids=["crop+upscale", "flip+upscale", "downscale"],
)
def test_to_source_coords_masks_end_to_end(stages, box_in_model, box_in_source):
    pipe = vt.Pipeline(
        [*stages, stubs.BoxDetectorStub([box_in_model], masks=True), vt.ToSourceCoords()]
    )
    pipe.validate(fx.bgr_u8().spec)
    det = pipe(fx.bgr_u8()).results["detections"]
    np.testing.assert_allclose(det.xyxy[0], box_in_source, atol=1e-5)
    assert det.mask.shape == (1, *SRC)
    np.testing.assert_array_equal(det.mask[0], rasterize_box(box_in_source, SRC))
