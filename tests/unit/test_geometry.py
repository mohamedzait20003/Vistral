import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from vistral.core import GeoTransform, compose, map_boxes, map_points, to_source_matrix

finite = st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False)
positive = st.floats(min_value=0.05, max_value=20, allow_nan=False)


def _affine(draw_scale: tuple[float, float], shift: tuple[float, float]) -> GeoTransform:
    (sx, sy), (tx, ty) = draw_scale, shift
    return GeoTransform("affine", [[sx, 0, tx], [0, sy, ty], [0, 0, 1]], (100, 100), (100, 100))


@given(
    st.lists(
        st.tuples(st.tuples(positive, positive), st.tuples(finite, finite)), min_size=1, max_size=5
    ),
    st.lists(st.tuples(finite, finite), min_size=1, max_size=10),
)
def test_forward_then_inverse_is_identity(chain, points):
    transforms = [_affine(scale, shift) for scale, shift in chain]
    pts = np.array(points)
    forward = map_points(compose(transforms), pts)
    back = map_points(to_source_matrix(transforms), forward)
    np.testing.assert_allclose(back, pts, atol=1e-6, rtol=1e-9)


@given(st.tuples(positive, positive), st.tuples(finite, finite))
def test_inverse_transform_matches_matrix_inverse(scale, shift):
    t = _affine(scale, shift)
    np.testing.assert_allclose(t.inverse().array @ t.array, np.eye(3), atol=1e-9)
    assert t.inverse().src_size == t.dst_size


def test_resize_maps_corners_exactly():
    t = GeoTransform.resize((100, 200), (50, 400))
    np.testing.assert_allclose(map_boxes(t.array, [[10, 20, 30, 40]]), [[20, 10, 60, 20]])


def test_crop_pad_flip():
    crop = GeoTransform.crop(10, 5, 50, 45, (100, 100))
    assert crop.dst_size == (40, 40)
    np.testing.assert_allclose(map_points(crop.array, [[10, 5]]), [[0, 0]])
    pad = GeoTransform.pad(1, 2, 3, 4, (10, 10))
    assert pad.dst_size == (13, 17)
    np.testing.assert_allclose(map_points(pad.array, [[0, 0]]), [[3, 1]])
    hflip = GeoTransform.flip((10, 20))
    np.testing.assert_allclose(map_boxes(hflip.array, [[0, 0, 5, 5]]), [[15, 0, 20, 5]])
    vflip = GeoTransform.flip((10, 20), horizontal=False)
    np.testing.assert_allclose(map_boxes(vflip.array, [[0, 0, 5, 5]]), [[0, 5, 5, 10]])


def test_rotation_reboxes():
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    rot = GeoTransform("rotate", [[c, -s, 0], [s, c, 0], [0, 0, 1]], (10, 10), (10, 10))
    box = map_boxes(rot.array, [[0, 0, 2, 2]])[0]
    assert box[2] - box[0] == pytest.approx(2 * np.sqrt(2))


def test_empty_boxes_and_validation():
    assert map_boxes(np.eye(3), np.zeros((0, 4))).shape == (0, 4)
    with pytest.raises(ValueError):
        GeoTransform("bad", np.eye(2), (1, 1), (1, 1))


def test_to_dict_and_hashable():
    t = GeoTransform.resize((2, 2), (4, 4), producer="r")
    assert t.to_dict()["producer"] == "r"
    assert hash(t) == hash(GeoTransform.resize((2, 2), (4, 4), producer="r"))
