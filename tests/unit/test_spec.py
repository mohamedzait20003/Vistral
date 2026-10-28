import numpy as np
import pytest

from vistral.core import Accepts, Backend, Color, ImageSpec, Layout, Range
from vistral.core.errors import ConfigError, SpecMismatchError
from vistral.core.spec import consistency_violations, derive_spec, infer_layout, observe


class TestRange:
    def test_kinds_and_str(self):
        assert Range("u8") == Range.U8
        assert str(Range.UNIT) == "UNIT"
        with pytest.raises(ValueError):
            Range("BOGUS")

    def test_normalized_requires_stats(self):
        with pytest.raises(ValueError):
            Range("NORMALIZED")
        with pytest.raises(ValueError):
            Range("U8", mean=(0.5,))
        r = Range.normalized([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        assert r.mean == (0.5, 0.5, 0.5)
        assert "NORMALIZED" in str(r)

    def test_coerce_roundtrip(self):
        r = Range.normalized([0.1], [0.2])
        assert Range.coerce(r.to_config()) == r
        assert Range.coerce("UNIT") == Range.UNIT
        assert Range.coerce({"kind": "U16"}) == Range.U16
        assert Range.coerce(None) is None
        with pytest.raises(TypeError):
            Range.coerce(3)


class TestImageSpec:
    def test_coercion_of_fields(self):
        spec = ImageSpec(layout="HWC", color="RGB", dtype=np.uint8, range="U8", size=[4, 5])
        assert spec.layout is Layout.HWC and spec.color is Color.RGB
        assert spec.dtype == "uint8" and spec.range == Range.U8 and spec.size == (4, 5)
        assert spec.channels == 3

    def test_unsupported_dtype(self):
        with pytest.raises(ValueError):
            ImageSpec(dtype="complex64")

    def test_named_constructors(self):
        assert ImageSpec.bgr_u8().color is Color.BGR
        assert ImageSpec.rgb_u8((2, 3)).size == (2, 3)
        assert ImageSpec.gray_u8().layout is Layout.HW
        assert ImageSpec.gray_u16().range == Range.U16
        assert ImageSpec.rgb_unit_chw().layout is Layout.CHW

    def test_config_roundtrip(self):
        spec = ImageSpec.rgb_unit_chw((10, 20))
        assert ImageSpec.coerce(spec.to_config()) == spec
        with pytest.raises(ConfigError):
            ImageSpec.coerce({"colour": "RGB"})
        with pytest.raises(TypeError):
            ImageSpec.coerce(5)

    def test_str_unknown(self):
        assert "?color" in str(ImageSpec())
        assert "48x64" in str(ImageSpec.bgr_u8((48, 64)))


class TestAccepts:
    def test_check_violation_and_unknown(self):
        acc = Accepts(color="RGB", dtype="uint8")
        violations, unknown = acc.check(ImageSpec(color="BGR"))
        assert [v.field for v in violations] == ["color"]
        assert unknown == ["dtype"]
        assert not acc.satisfied_by(ImageSpec(color="RGB"))
        assert acc.satisfied_by(ImageSpec(color="RGB", dtype="uint8"))

    def test_size_swap_hint(self):
        acc = Accepts(size=(64, 48))
        (violation,), _ = acc.check(ImageSpec(size=(48, 64)))
        assert "swapped" in str(violation)

    def test_normalized_hint(self):
        acc = Accepts(range="U8")
        (violation,), _ = acc.check(ImageSpec(range=Range.normalized([0], [1])))
        assert "already normalized" in str(violation)

    def test_device_prefix(self):
        acc = Accepts(device="cuda")
        assert acc.satisfied_by(ImageSpec(device="cuda:1"))
        assert not acc.satisfied_by(ImageSpec(device="cpu"))

    def test_sets_and_coerce(self):
        acc = Accepts(color={"BGR", "GRAY"}, layout=["HW", "HWC"])
        assert acc.color == {Color.BGR, Color.GRAY}
        assert Accepts.coerce(acc.to_config()) == acc
        assert Accepts.coerce(ImageSpec(color="RGB")).color == {Color.RGB}
        assert Accepts.coerce(None) is None
        with pytest.raises(TypeError):
            Accepts.coerce(1)
        assert "color=BGR|GRAY" in str(acc)

    def test_size_config(self):
        acc = Accepts(size=(3, 4))
        assert acc.to_config() == {"size": [3, 4]}
        assert "size=(3, 4)" in str(acc)


class TestObservation:
    def test_observe_numpy(self):
        obs = observe(np.zeros((2, 3, 3), np.float32))
        assert obs.backend is Backend.NUMPY and obs.dtype == "float32" and obs.device == "cpu"

    def test_observe_rejects_other(self):
        with pytest.raises(TypeError):
            observe([1, 2, 3])

    @pytest.mark.parametrize(
        ("shape", "color", "declared", "expected"),
        [
            ((4, 5), None, None, Layout.HW),
            ((4, 5, 3), None, None, Layout.HWC),
            ((3, 4, 5), None, None, Layout.CHW),
            ((3, 4, 3), Color.RGB, Layout.CHW, Layout.CHW),
            ((4, 5, 3), Color.RGB, Layout.CHW, Layout.HWC),
            ((5, 6, 7), None, None, None),
            ((5, 6, 7), Color.RGB, None, None),
            ((1, 2, 3, 4), None, None, None),
        ],
    )
    def test_infer_layout(self, shape, color, declared, expected):
        assert infer_layout(shape, Backend.NUMPY, color=color, declared=declared) == expected

    def test_infer_layout_torch_prefers_chw(self):
        assert infer_layout((3, 4, 3), Backend.TORCH) is Layout.CHW
        assert infer_layout((5, 6, 3), Backend.TORCH) is Layout.HWC
        assert infer_layout((5, 6, 7), Backend.TORCH) is None

    def test_derive_range_rules(self):
        base = ImageSpec(color="GRAY", range="UNIT", dtype="float32")
        assert derive_spec(np.zeros((2, 2), np.float32), base=base).range == Range.UNIT
        assert derive_spec(np.zeros((2, 2), np.float64), base=base).range is None
        assert derive_spec(np.zeros((2, 2), np.uint16)).range == Range.U16
        assert derive_spec(np.zeros((2, 2), bool)).range == Range.ANY
        assert derive_spec(np.zeros((2, 2), np.float32), declared={"range": "U8"}).range is None

    def test_derive_drops_inconsistent_base_color(self):
        base = ImageSpec(color="BGR")
        spec = derive_spec(np.zeros((4, 4), np.uint8), base=base)
        assert spec.color is None and spec.layout is Layout.HW

    def test_derive_strict_declared_color(self):
        with pytest.raises(SpecMismatchError):
            derive_spec(np.zeros((4, 4), np.uint8), declared={"color": "RGB"}, strict=True)


class TestConsistency:
    def test_clean(self):
        data = np.zeros((4, 4, 3), np.uint8)
        assert consistency_violations(data, ImageSpec.bgr_u8((4, 4))) == []

    def test_reports_problems(self):
        data = np.full((4, 4), 2.0, np.float32)
        spec = ImageSpec(color="RGB", dtype="uint8", range="UNIT", backend="numpy", layout="HWC")
        fields = {v.field for v in consistency_violations(data, spec)}
        assert {"dtype", "color", "range", "layout"} <= fields

    def test_u8_range_needs_uint8(self):
        (v,) = consistency_violations(np.zeros((2, 2), np.float32), ImageSpec(range="U8"))
        assert v.field == "range"
