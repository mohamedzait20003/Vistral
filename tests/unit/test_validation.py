import pytest

import vistral as vt
from tests import custom_stages as cs
from vistral.testing import fixtures as fx
from vistral.testing import stubs
from vistral.testing.fault_corpus import cases

CASES = cases()


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_fault_is_caught(case):
    image = case.make_input()
    pipe = case.build()
    if case.detect == "validate":
        with pytest.raises(case.error) as info:
            pipe.validate(image.spec)
        assert info.value.suggestion or isinstance(info.value, vt.SpecMismatchError)
    else:
        pipe.validate(image.spec)  # passes statically (possibly with warnings)
        with pytest.raises(case.error):
            pipe.run(image)


@pytest.mark.parametrize("case", [c for c in CASES if c.auto_fixable], ids=lambda c: c.name)
def test_auto_convert_repairs(case):
    image = case.make_input()
    report = case.build().validate(image.spec, auto_convert=True)
    assert report.ok and report.insertions
    report.pipeline.run(image)


@pytest.mark.parametrize("case", [c for c in CASES if c.fixed], ids=lambda c: c.name)
def test_fixed_pipeline_validates_and_runs(case):
    image = case.make_input()
    pipe = case.fixed()
    assert pipe.validate(image.spec).ok
    pipe.run(image)


def test_every_validate_case_has_a_suggestion():
    for case in CASES:
        if case.detect != "validate":
            continue
        report = case.build().validate(case.make_input().spec, raise_on_error=False)
        assert report.errors, case.name
        err = report.errors[0].error
        assert err.suggestion, f"{case.name}: no suggestion in {err}"


def test_lossy_auto_convert_requires_opt_in():
    pipe = vt.Pipeline([stubs.RGBModelStub()])
    with pytest.raises(vt.SpecMismatchError, match="allow_lossy"):
        pipe.validate(fx.bgra_u8().spec, auto_convert=True)
    report = pipe.validate(fx.bgra_u8().spec, auto_convert=True, allow_lossy=True)
    assert "drop_alpha" in report.insertions[0]
    report.pipeline.run(fx.bgra_u8())


def test_report_collects_all_errors():
    pipe = vt.Pipeline([stubs.RGBModelStub(), stubs.CHWUnitModelStub()])
    report = pipe.validate(fx.bgr_u8().spec, raise_on_error=False)
    assert len(report.errors) == 2 and not report.ok
    with pytest.raises(vt.SpecMismatchError) as info:
        report.raise_for_errors()
    assert "1 more validation error" in "\n".join(info.value.__notes__)
    assert "error(s)" in str(report)


def test_warnings_for_unknowns_and_missing_results():
    pipe = vt.Pipeline([stubs.OpaqueFilterStub(), stubs.RGBModelStub(), cs.drop_small()])
    report = pipe.validate(fx.rgb_u8().spec)
    messages = [w.message for w in report.warnings]
    assert any("cannot verify color" in m for m in messages)
    assert any("'detections'" in m for m in messages)


def test_validate_accepts_images_dicts_and_arrays():
    pipe = vt.Pipeline([cs.Brighten()])
    assert pipe.validate(fx.bgr_u8()).ok
    assert pipe.validate({"dtype": "uint8"}).ok
    assert pipe.validate(fx.bgr_u8().data).ok
    assert pipe.validate(vt.ImageSpec.bgr_u8()).output_spec.color is vt.Color.BGR


def test_nested_index_paths_and_rebuild():
    inner = vt.Pipeline([stubs.RGBModelStub()], name="inner")
    pipe = vt.Pipeline([cs.Brighten(), inner])
    report = pipe.validate(fx.bgr_u8().spec, raise_on_error=False)
    assert report.errors[0].index == "1.0"
    fixed = pipe.validate(fx.bgr_u8().spec, auto_convert=True)
    assert fixed.pipeline is not pipe and fixed.pipeline["inner"].name == "inner"
    assert len(fixed.pipeline["inner"]) == 2
    assert pipe.validate(fx.rgb_u8().spec).pipeline is pipe


def test_convert_stage_without_path_is_an_error():
    pipe = vt.Pipeline([vt.Convert(color="RGB")])
    with pytest.raises(vt.SpecMismatchError, match="cannot convert"):
        pipe.validate(fx.mask().spec)


def test_geometry_and_coordinate_tracking():
    pipe = vt.Pipeline(
        [
            stubs.ResizeStub(size=(24, 32)),
            stubs.BoxDetectorStub([[0, 0, 4, 4]]),
            vt.ToSourceCoords(keys=["detections", "missing"]),
            stubs.MeasureStub(),
        ]
    )
    report = pipe.validate(fx.bgr_u8().spec)
    assert report.state.result_spaces["detections"] == "source"
    assert report.output_spec.size == (24, 32)
    assert any("'missing'" in w.message for w in report.warnings)
