import numpy as np
import pytest
import supervision as sv

from vistral.core import vt
from vistral.core.errors import ConversionError
from vistral.core.results import is_sv_detections, summarize, transform_result
from vistral.testing import fixtures as fx
from vistral.testing import stubs


def test_stamp_and_meta_on_sv_detections_without_mutation():
    det = sv.Detections(xyxy=np.array([[0, 0, 2, 2]], np.float32))
    frame = fx.frame()
    stamped = vt.stamp(det, frame, producer="p", model_id="m")
    assert vt.get_meta(stamped).model_id == "m"
    assert "vistral" not in det.metadata
    assert vt.get_meta(stamped[np.array([True])]).producer == "p"  # survives indexing
    assert vt.get_meta(det) is None


def test_set_meta_on_other_objects_is_noop():
    assert vt.set_meta(5, vt.ResultMeta()) == 5
    assert vt.get_meta(object()) is None


def test_result_types_and_dicts():
    cls = vt.Classification(np.array([3, 1]), np.array([0.2, 0.8]), labels=("cat", "dog"))
    assert cls.top == (1, 0.8, "dog")
    assert isinstance(cls.to_supervision(), sv.Classifications)
    assert "dog" in cls.summary()
    kp = vt.Keypoints(np.zeros((1, 2, 2)), confidence=np.ones((1, 2), np.float32))
    assert isinstance(kp.to_supervision(), sv.KeyPoints)
    m = vt.Measurements((vt.Measurement("w", 5, lo=1, hi=4), vt.Measurement("h", 2)))
    assert not m.all_ok and m.items[1].ok is None
    verdict = vt.Verdict("FAIL", reasons=("scratch",))
    assert verdict.status is vt.Status.FAIL and verdict.to_dict()["status"] == "FAIL"
    big = vt.SemanticMask(np.zeros((200, 200), np.int32))
    assert big.to_dict()["label_map"] == {"shape": [200, 200], "dtype": "int32"}
    for r in (
        vt.Embedding(np.zeros(8)),
        vt.Text("hello"),
        vt.Contours((np.zeros((3, 2)),)),
        big,
        m,
        verdict,
        kp,
        cls,
    ):
        assert summarize(r)
        assert vt.result_to_dict(r)["type"] == type(r).__name__
    assert summarize(3) == "int"
    assert vt.result_to_dict(3) == {"type": "int", "value": 3}


def test_detections_to_dict():
    det = sv.Detections(
        xyxy=np.array([[0, 0, 2, 2]], np.float32),
        mask=np.zeros((1, 4, 4), bool),
        class_id=np.array([0]),
    )
    det = vt.stamp(det, fx.frame(), producer="p")
    out = vt.result_to_dict(det)
    assert out["mask"] == {"shape": [1, 4, 4]} and out["meta"]["producer"] == "p"
    assert is_sv_detections(det) and summarize(det) == "Detections(n=1)"


def test_transform_result_variants():
    m = vt.GeoTransform.scale(2, 2, (4, 4), (8, 8)).array
    det = sv.Detections(xyxy=np.array([[1, 1, 2, 2]], np.float32), mask=np.ones((1, 4, 4), bool))
    out = transform_result(det, m, (8, 8))
    np.testing.assert_allclose(out.xyxy, [[2, 2, 4, 4]])
    assert out.mask.shape == (1, 8, 8) and out.mask.all()
    empty = sv.Detections(xyxy=np.zeros((0, 4), np.float32), mask=np.zeros((0, 4, 4), bool))
    assert transform_result(empty, m, (8, 8)).mask.shape == (0, 8, 8)
    kp = sv.KeyPoints(xy=np.array([[[1, 1]]], np.float32))
    np.testing.assert_allclose(transform_result(kp, m, (8, 8)).xy, [[[2, 2]]])
    contours = vt.Contours((np.array([[1, 1], [2, 2]]),))
    np.testing.assert_allclose(transform_result(contours, m, (8, 8)).points[0], [[2, 2], [4, 4]])
    sem = vt.SemanticMask(np.ones((4, 4), np.uint8))
    assert transform_result(sem, m, (8, 8)).label_map.shape == (8, 8)
    vk = vt.Keypoints(np.array([[[1.0, 1.0]]]))
    np.testing.assert_allclose(transform_result(vk, m, (8, 8)).xy, [[[2, 2]]])
    assert transform_result(vt.Text("x"), m, (8, 8)).text == "x"
    with pytest.raises(ConversionError):
        transform_result(object(), m, (8, 8))


def test_to_source_coords_through_resize_and_crop():
    # Source 48x64; crop (8,4)-(40,36) -> 32x32; resize to 16x16. A box in final coords
    # [2,2,6,6] corresponds to source [8+4, 4+4, 8+12, 4+12].
    pipe = vt.Pipeline(
        [
            stubs.CropStub(box=(8, 4, 40, 36)),
            stubs.ResizeStub(size=(16, 16)),
            stubs.BoxDetectorStub([[2, 2, 6, 6]]),
            vt.ToSourceCoords(),
            stubs.MeasureStub(),
        ]
    )
    pipe.validate(fx.bgr_u8().spec)
    frame = pipe(fx.bgr_u8())
    det = frame.results["detections"]
    np.testing.assert_allclose(det.xyxy, [[12, 8, 20, 16]], atol=1e-5)
    meta = vt.get_meta(det)
    assert meta.coord_space == "source" and meta.image_size == (48, 64) and meta.transforms == ()
    assert frame.results["measurements"].items[0].value == pytest.approx(8)


def test_to_source_coords_explicit_key_errors():
    with pytest.raises(vt.StageError, match="not found"):
        vt.ToSourceCoords(keys=["detections"])(fx.bgr_u8())

    class Weird:
        meta = vt.ResultMeta(
            coord_space="stage:x", transforms=(vt.GeoTransform.resize((4, 4), (2, 2)),)
        )

    frame = fx.frame()
    frame.results["w"] = Weird()
    assert vt.ToSourceCoords()(frame).results["w"] is frame.results["w"]
    with pytest.raises(vt.StageError, match="don't know"):
        vt.ToSourceCoords(keys=["w"])(frame)


def test_result_meta_to_dict():
    meta = vt.ResultMeta(
        producer="p", extra={"k": 1}, transforms=(vt.GeoTransform.resize((2, 2), (4, 4)),)
    )
    out = meta.to_dict()
    assert out["extra"] == {"k": 1} and out["transforms"][0]["name"] == "resize"
    assert "recipe_hash" not in out
