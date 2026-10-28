import numpy as np
import pytest

from vistral.core import Accepts, Color, Image, ImageSpec, Layout, Range, vt
from vistral.core.convert import DEFAULT_GRAPH, ConversionGraph, describe_path
from vistral.core.errors import ConversionError
from vistral.testing import fixtures as fx


def names(path):
    return [c.name for c in path]


class TestPlanning:
    def test_already_satisfied(self):
        assert vt.plan(ImageSpec.bgr_u8(), Accepts(color="BGR")) == []

    def test_unknown_field_cannot_plan(self):
        assert vt.plan(ImageSpec(color=None, backend="numpy"), Accepts(color="RGB")) is None

    def test_bgr_to_rgb(self):
        assert names(vt.plan(ImageSpec.bgr_u8(), Accepts(color="RGB"))) == ["swap_rb"]

    def test_to_chw_unit(self):
        path = vt.plan(ImageSpec.bgr_u8(), Accepts.from_spec(ImageSpec.rgb_unit_chw()))
        assert sorted(names(path)) == ["hwc_to_chw", "swap_rb", "u8_to_unit"]

    def test_lossy_gated(self):
        target = Accepts(dtype="uint8")
        assert vt.plan(ImageSpec.gray_u16(), target) is None
        assert names(vt.plan(ImageSpec.gray_u16(), target, allow_lossy=True)) == ["u16_to_u8"]

    def test_normalized_has_no_path(self):
        spec = ImageSpec.rgb_u8().replace(dtype="float32", range=Range.normalized([0], [1]))
        assert vt.plan(spec, Accepts(range="U8"), allow_lossy=True) is None

    def test_describe(self):
        assert describe_path([]) == "(no conversion)"
        assert "lossy" in describe_path(
            vt.plan(ImageSpec.bgr_u8(), Accepts(color="GRAY"), allow_lossy=True)
        )


class TestConverters:
    def test_swap_rb_values(self):
        img = fx.gradient_bgr()
        out = vt.convert_image(img, Accepts(color="RGB"))
        np.testing.assert_array_equal(out.data, img.data[..., ::-1])
        assert out.spec.color is Color.RGB and out.data.flags.c_contiguous

    def test_swap_rb_4ch_and_chw(self):
        img = fx.bgra_u8()
        out = vt.convert_image(img, Accepts(color="RGBA"))
        np.testing.assert_array_equal(out.data[..., 3], img.data[..., 3])
        np.testing.assert_array_equal(out.data[..., 0], img.data[..., 2])
        chw = vt.convert_image(fx.rgb_unit(), Accepts(layout="CHW"))
        back = vt.convert_image(chw, Accepts(color="BGR"))
        np.testing.assert_array_equal(back.data[0], chw.data[2])

    def test_hwc_chw_is_zero_copy(self):
        img = fx.rgb_u8()
        out = vt.convert_image(img, Accepts(layout="CHW"))
        assert np.shares_memory(out.data, img.data)
        assert out.spec.layout is Layout.CHW and out.spec.size == img.spec.size
        back = vt.convert_image(out, Accepts(layout="HWC"))
        np.testing.assert_array_equal(back.data, img.data)

    def test_gray_expansions(self):
        g = fx.gray_u8()
        rgb = vt.convert_image(g, Accepts(color="RGB"))
        assert rgb.data.shape == (*g.size, 3)
        assert (rgb.data[..., 0] == g.data).all()
        hwc = vt.convert_image(g, Accepts(layout="HWC"))
        assert hwc.data.shape == (*g.size, 1)
        chw = vt.convert_image(g, Accepts(layout="CHW"))
        assert chw.data.shape == (1, *g.size)
        assert vt.convert_image(chw, Accepts(layout="HW")).data.shape == g.size
        chw3 = vt.convert_image(chw, Accepts(color="BGR", layout="CHW"))
        assert chw3.data.shape == (3, *g.size)

    def test_to_gray_matches_opencv(self):
        import cv2

        img = fx.bgr_u8()
        out = vt.convert_image(img, Accepts(color="GRAY"), allow_lossy=True)
        ref = cv2.cvtColor(img.data, cv2.COLOR_BGR2GRAY)
        assert np.abs(out.data.astype(int) - ref).max() <= 1
        rgbf = vt.convert_image(fx.rgb_unit(), Accepts(color="GRAY"), allow_lossy=True)
        assert rgbf.spec.dtype == "float32"

    def test_value_ranges(self):
        img = fx.rgb_u8()
        unit = vt.convert_image(img, Accepts(range="UNIT"))
        assert unit.spec.dtype == "float32" and 0 <= unit.data.min() and unit.data.max() <= 1
        back = vt.convert_image(unit, Accepts(range="U8"), allow_lossy=True)
        np.testing.assert_array_equal(back.data, img.data)
        u16 = vt.convert_image(img, Accepts(dtype="uint16"))
        assert int(u16.data.max()) == int(img.data.max()) * 257
        down = vt.convert_image(u16, Accepts(dtype="uint8"), allow_lossy=True)
        np.testing.assert_array_equal(down.data, img.data)
        g16 = vt.convert_image(fx.gray_u16(), Accepts(range="UNIT"))
        assert g16.data.max() <= 1
        assert vt.convert_image(unit, Accepts(dtype="uint16"), allow_lossy=True).spec.range == (
            Range.U16
        )

    def test_float_widths(self):
        f16 = Image.from_array(np.zeros((2, 2), np.float16), color="GRAY")
        assert vt.convert_image(f16, Accepts(dtype="float32")).spec.dtype == "float32"
        f64 = Image.from_array(np.zeros((2, 2), np.float64), color="GRAY")
        with pytest.raises(ConversionError, match="allow_lossy"):
            vt.convert_image(f64, Accepts(dtype="float32"))
        f32 = vt.convert_image(f64, Accepts(dtype="float32"), allow_lossy=True)
        assert vt.convert_image(f32, Accepts(dtype="float16"), allow_lossy=True).spec.dtype == (
            "float16"
        )

    def test_drop_alpha(self):
        out = vt.convert_image(fx.bgra_u8(), Accepts(color="RGB"), allow_lossy=True)
        assert out.data.shape[-1] == 3 and out.spec.color is Color.RGB

    def test_no_path_error(self):
        with pytest.raises(ConversionError):
            vt.convert_image(fx.mask(), Accepts(color="RGB"))

    def test_records_conversions_when_tracing(self):
        result = vt.Pipeline([vt.Convert(color="RGB", layout="CHW")]).run(fx.bgr_u8(), trace=True)
        convs = result.trace.conversions
        assert sorted(names(convs)) == ["hwc_to_chw", "swap_rb"]
        assert {c.name: c.copied for c in convs} == {"swap_rb": True, "hwc_to_chw": False}
        assert result.trace.copies == 1

    def test_custom_graph(self):
        graph = ConversionGraph()
        assert graph.plan(ImageSpec.bgr_u8(), Accepts(color="RGB")) is None
        assert len(DEFAULT_GRAPH.converters) >= 19


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None, reason="torch not installed"
)
def test_numpy_to_torch_zero_copy():  # pragma: no cover - runs only with the [hf] extra
    img = fx.rgb_u8()
    out = vt.convert_image(img, Accepts(backend="torch"))
    assert out.spec.backend == "torch"
    assert out.data.data_ptr() == img.data.__array_interface__["data"][0]
