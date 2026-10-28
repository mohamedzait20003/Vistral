"""Every stage must pass vistral.testing.check_stage (D-013, acceptance criterion 5)."""

import pytest

from tests import custom_stages as cs
from vistral.core import vt
from vistral.testing import check_stage, stubs
from vistral.testing import fixtures as fx

ALLOW = ("tests.*", "vistral.*")

CASES = [
    (cs.Brighten(5), fx.bgr_u8),
    (cs.Brighten(5), fx.gray_u8),
    (cs.invert(amount=200), fx.gray_u8),
    (cs.first_channel(), fx.bgr_u8),
    (cs.tag(label="z"), fx.bgr_u8),
    (cs.MeanEncoder(), fx.bgr_u8),
    (vt.Convert(color="RGB"), fx.bgr_u8),
    (vt.Convert(color="GRAY", allow_lossy=True), fx.bgr_u8),
    (vt.Convert(range="UNIT", layout="CHW"), fx.rgb_u8),
    (vt.ToSourceCoords(), fx.bgr_u8),
    (vt.Pipeline([cs.Brighten(), cs.invert()], name="nested"), fx.bgr_u8),
    (vt.Branch({"a": stubs.BoxDetectorStub([[0, 0, 2, 2]])}), fx.bgr_u8),
    (vt.When(cs.is_fail, cs.Brighten(), cs.invert()), fx.bgr_u8),
    (stubs.RGBModelStub(), fx.rgb_u8),
    (stubs.CHWUnitModelStub(), lambda: vt.convert_image(fx.rgb_unit(), {"layout": "CHW"})),
    (stubs.NormalizeStub(), fx.rgb_u8),
    (stubs.ResizeStub(size=(10, 12)), fx.bgr_u8),
    (stubs.CropStub(box=(1, 2, 9, 10)), fx.bgr_u8),
    (stubs.BoxDetectorStub([[0, 0, 3, 3]], labels=["scratch"]), fx.bgr_u8),
    (stubs.FixedSizeModelStub(input_size=(48, 64)), fx.bgr_u8),
]


@pytest.mark.parametrize(("stage", "sample"), CASES, ids=[repr(s)[:40] for s, _ in CASES])
def test_check_stage(stage, sample):
    report = check_stage(stage, sample(), allow_imports=ALLOW)
    assert "config round-trip" in report.passed and "batch == loop" in report.passed
    assert "check_stage" in str(report)


def test_check_stage_detects_mutation():
    class Mutating(vt.Filter):
        def apply(self, data):
            data[...] = 0
            return data

    with pytest.raises(AssertionError, match="mutated"):
        check_stage(Mutating(), fx.bgr_u8(), roundtrip=False)


def test_check_stage_detects_nondeterminism():
    import numpy as np

    class Noisy(vt.Filter):
        def apply(self, data):
            return np.random.default_rng().integers(0, 255, data.shape, dtype=np.uint8)

    with pytest.raises(AssertionError, match="deterministic"):
        check_stage(Noisy(), fx.bgr_u8(), roundtrip=False)
    report = check_stage(Noisy(), fx.bgr_u8(), roundtrip=False, deterministic=False)
    assert report.skipped == ["deterministic", "config round-trip"]


def test_check_stage_rejects_unaccepted_sample():
    with pytest.raises(AssertionError, match="not accepted"):
        check_stage(stubs.RGBModelStub(), fx.bgr_u8())


def test_check_stage_detects_lying_spec():
    with pytest.raises(vt.SpecMismatchError):
        check_stage(stubs.LyingFilterStub(), fx.gray_u8(), roundtrip=False)
