import numpy as np
import pytest

import vistral as vt
from tests import custom_stages as cs
from vistral.testing import fixtures as fx
from vistral.testing import stubs


class TestBranch:
    def test_namespaced_results_and_image_passthrough(self):
        branch = vt.Branch(
            {
                "a": stubs.BoxDetectorStub([[0, 0, 1, 1]]),
                "b": [cs.Brighten(), stubs.BoxDetectorStub([[0, 0, 2, 2], [1, 1, 3, 3]])],
            }
        )
        img = fx.bgr_u8()
        out = branch(img)
        assert len(out.results["a.detections"]) == 1 and len(out.results["b.detections"]) == 2
        assert out.image is img

    def test_validation_namespaces_and_auto_convert(self):
        branch = vt.Branch({"rgb": stubs.RGBModelStub(), "det": stubs.BoxDetectorStub([])})
        pipe = vt.Pipeline([branch])
        report = pipe.validate(fx.bgr_u8().spec, raise_on_error=False)
        assert report.errors[0].index.endswith("[rgb].0")
        fixed = pipe.validate(fx.bgr_u8().spec, auto_convert=True)
        assert "det.detections" in fixed.state.result_spaces
        fixed.pipeline.run(fx.bgr_u8())

    def test_lifecycle_and_errors(self):
        counter = stubs.CountingStage()
        branch = vt.Branch({"x": counter})
        with vt.Pipeline([branch]) as pipe:
            pipe(fx.bgr_u8())
            pipe.reset()
        assert counter.setups == 1 and counter.closes == 1 and counter.resets == 1
        with pytest.raises(ValueError):
            vt.Branch({})
        with pytest.raises(ValueError):
            vt.Branch({"x": counter}, merge="flat")

    def test_failure_in_skip_branch(self):
        failing = vt.Pipeline([stubs.FailingStage(fail_on=[None])], on_error="skip")
        out = vt.Branch({"f": failing})(fx.bgr_u8())
        assert out.failed


class TestWhen:
    def test_condition_routing(self):
        verdict = vt.Verdict("FAIL")

        @vt.stage
        def set_verdict(frame, *, status="PASS"):
            frame.results["verdict"] = vt.Verdict(status)
            return frame

        when = vt.When(
            {"result": "verdict", "equals": "FAIL"}, cs.tag(label="escalated"), cs.tag(label="ok")
        )
        assert (set_verdict(status="FAIL") | when)(fx.bgr_u8()).artifacts["tag"] == "escalated"
        assert (set_verdict() | when)(fx.bgr_u8()).artifacts["tag"] == "ok"
        assert vt.When(cs.is_fail, cs.Brighten())(fx.bgr_u8()).image.data is not None
        assert verdict.status == "FAIL"

    def test_condition_forms(self):
        frame = fx.frame()
        frame.results["v"] = vt.Verdict("REVIEW")
        frame.results["n"] = 3
        C = vt.Condition
        assert C(result="v", in_=("REVIEW", "FAIL"))(frame)
        assert C.coerce({"result": "v", "in": ["PASS"]})(frame) is False
        assert C(result="v", attr="status", equals="REVIEW")(frame)
        assert C(result="n", equals=3)(frame)
        assert C(result="missing", exists=False)(frame)
        assert not C(result="missing")(frame)
        assert not C(result="n", exists=False)(frame)
        assert C(result="n")(frame)
        cond = C(result="v", in_=("A",), attr="status", exists=True, equals="x")
        assert C.coerce(cond.to_config()) == cond
        assert C.coerce(cond) is cond
        with pytest.raises(TypeError):
            C.coerce(5)

    def test_validation_merges_branches(self):
        when = vt.When({"result": "x", "exists": True}, cs.first_channel(), None)
        report = vt.Pipeline([when]).validate(fx.bgr_u8().spec)
        assert report.output_spec.color is None
        assert any("different" in w.message for w in report.warnings)
        geo = vt.When(cs.is_fail, stubs.ResizeStub(size=(2, 2)), stubs.BoxDetectorStub([]))
        report = vt.Pipeline([geo, stubs.BoxDetectorStub([])]).validate(fx.bgr_u8().spec)
        assert any("geometry" in w.message for w in report.warnings)

    def test_auto_convert_inside_when(self):
        when = vt.When(cs.is_fail, stubs.RGBModelStub(), stubs.RGBModelStub())
        fixed = vt.Pipeline([when]).validate(fx.bgr_u8().spec, auto_convert=True)
        assert len(fixed.insertions) == 2
        frame = vt.Frame(fx.bgr_u8())
        frame.artifacts["tag"] = "fail"
        out = fixed.pipeline.run(frame).frame
        assert "raw_outputs" in out.artifacts

    def test_lifecycle(self):
        counter = stubs.CountingStage()
        with vt.Pipeline([vt.When(cs.is_fail, counter, counter)]) as pipe:
            pipe.reset()
        assert counter.setups == 1 and counter.resets == 2 and counter.closes == 2

    def test_skipped_branch_returns_frame(self):
        out = vt.When(cs.is_fail, stubs.FailingStage())(np.zeros((2, 2, 3), np.uint8))
        assert not out.failed
