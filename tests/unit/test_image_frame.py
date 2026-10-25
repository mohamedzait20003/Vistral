import cv2
import numpy as np
import pytest
from PIL import Image as PILImage

import vistral as vt
from vistral import Color, Frame, Image, ImageSpec, Layout, Range, SourceInfo
from vistral.core.errors import ConversionError, SpecMismatchError


class TestImage:
    def test_from_array_defaults(self):
        assert Image.from_array(np.zeros((4, 5, 3), np.uint8)).spec.color is Color.BGR
        assert Image.from_array(np.zeros((4, 5, 4), np.uint8)).spec.color is Color.BGRA
        assert Image.from_array(np.zeros((4, 5), np.uint8)).spec.color is Color.GRAY
        assert Image.from_array(np.zeros((4, 5), bool)).spec.color is Color.MASK
        spec = Image.from_array(np.zeros((4, 5, 3), np.uint8)).spec
        assert spec.size == (4, 5) and spec.range == Range.U8 and spec.device == "cpu"

    def test_from_array_declarations(self):
        img = Image.from_array(np.zeros((3, 4, 5), np.float32), color="RGB", range="UNIT")
        assert img.spec.layout is Layout.CHW and img.spec.size == (4, 5)
        img = Image.from_array(np.zeros((4, 5, 3), np.uint8), spec=ImageSpec(color="RGB"))
        assert img.spec.color is Color.RGB

    def test_from_array_rejects_contradiction(self):
        with pytest.raises(SpecMismatchError):
            Image.from_array(np.zeros((4, 5), np.uint8), color="RGB")

    def test_with_data_and_transform(self):
        img = Image.from_array(np.zeros((4, 5, 3), np.uint8))
        gray = img.with_data(np.zeros((4, 5), np.uint8), color="GRAY")
        assert gray.spec.color is Color.GRAY
        t = vt.GeoTransform.crop(0, 0, 2, 2, (4, 5))
        cropped = img.with_transform(img.data[:2, :2], t)
        assert cropped.transforms == (t,) and cropped.size == (2, 2)
        assert "transforms=1" in repr(cropped)

    def test_copy_is_independent(self):
        img = Image.from_array(np.zeros((2, 2), np.uint8))
        cp = img.copy()
        cp.data[0, 0] = 9
        assert img.data[0, 0] == 0
        assert img.shape == (2, 2)

    @pytest.mark.parametrize(
        ("mode", "color"),
        [
            ("RGB", Color.RGB),
            ("RGBA", Color.RGBA),
            ("L", Color.GRAY),
            ("1", Color.MASK),
            ("I;16", Color.GRAY),
            ("P", Color.RGB),
        ],
    )
    def test_from_pil(self, mode, color):
        pil = PILImage.new(mode, (5, 4))
        img = Image.from_pil(pil)
        assert img.spec.color is color and img.size == (4, 5)

    def test_to_pil_swaps_bgr(self):
        data = np.zeros((2, 2, 3), np.uint8)
        data[..., 0] = 255  # blue in BGR
        pil = Image.from_array(data).to_pil()
        assert pil.getpixel((0, 0)) == (0, 0, 255)
        bgra = Image.from_array(np.zeros((2, 2, 4), np.uint8)).to_pil()
        assert bgra.mode == "RGBA"
        gray = Image.from_array(np.zeros((2, 2, 1), np.uint8)).to_pil()
        assert gray.mode == "L"

    def test_to_pil_requires_hwc(self):
        img = Image.from_array(np.zeros((3, 4, 5), np.float32), color="RGB", range="UNIT")
        with pytest.raises(ConversionError):
            img.to_pil()


class TestFrame:
    def test_fork_isolates_dicts(self):
        frame = Frame(Image.from_array(np.zeros((2, 2), np.uint8)))
        assert frame.original is frame.image
        child = frame.fork()
        child.results["x"] = 1
        child.errors.append("e")  # type: ignore[arg-type]
        assert frame.results == {} and frame.errors == []
        assert not frame.failed

    def test_coerce_variants(self, tmp_path):
        arr = np.zeros((4, 5, 3), np.uint8)
        assert Frame.coerce(arr).image.spec.color is Color.BGR
        img = Image.from_array(arr)
        assert Frame.coerce(img).image is img
        frame = Frame(img)
        assert Frame.coerce(frame) is frame
        assert Frame.coerce(PILImage.new("RGB", (3, 3))).image.spec.color is Color.RGB
        path = tmp_path / "pic.png"
        cv2.imwrite(str(path), arr)
        loaded = Frame.coerce(path)
        assert loaded.source.stem == "pic" and loaded.image.spec.color is Color.BGR
        with pytest.raises(TypeError):
            Frame.coerce(42)
        assert "results=[-]" in repr(frame)

    def test_source_info(self):
        info = SourceInfo.from_path("a/b/c.png")
        assert info.stem == "c" and "uri" in info.to_dict()


class TestReadImage:
    def test_reads_16bit_and_alpha(self, tmp_path):
        p16 = tmp_path / "d.png"
        cv2.imwrite(str(p16), np.full((3, 4), 40000, np.uint16))
        img = vt.read_image(p16)
        assert img.spec.dtype == "uint16" and img.spec.range == Range.U16
        pa = tmp_path / "a.png"
        cv2.imwrite(str(pa), np.zeros((3, 4, 4), np.uint8))
        assert vt.read_image(pa).spec.color is Color.BGRA

    def test_unicode_path(self, tmp_path):
        path = tmp_path / "pièce_检查.png"
        ok, buf = cv2.imencode(".png", np.zeros((3, 4, 3), np.uint8))
        assert ok
        buf.tofile(str(path))
        assert vt.read_image(path).size == (3, 4)

    def test_errors(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vt.read_image(tmp_path / "missing.png")
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        with pytest.raises(ValueError):
            vt.read_image(bad)
        with pytest.raises(TypeError):
            vt.read_image(3.5)

    def test_passthrough_and_arrays(self):
        img = Image.from_array(np.zeros((2, 2), np.uint8))
        assert vt.read_image(img) is img
        assert vt.read_image(np.zeros((2, 2, 3), np.uint8), color="RGB").spec.color is Color.RGB
        assert vt.read_image(PILImage.new("L", (2, 2))).spec.color is Color.GRAY
