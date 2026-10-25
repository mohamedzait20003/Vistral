import numpy as np
import pytest

import vistral as vt
from tests import custom_stages as cs
from vistral.testing import fixtures as fx
from vistral.testing import stubs


def frames(n: int, stream: str | None = None) -> list[vt.Frame]:
    img = fx.bgr_u8()
    return [vt.Frame(img, source=vt.SourceInfo(frame_index=i, stream_id=stream)) for i in range(n)]


class TestComposition:
    def test_pipe_operator_flattens_anonymous(self):
        pipe = cs.Brighten() | cs.invert() | cs.Brighten(1)
        assert len(pipe) == 3
        named = vt.Pipeline([cs.Brighten()], name="pre")
        assert len(named | cs.invert()) == 2 and (named | cs.invert())[0] is named
        assert len(cs.invert() | named) == 2

    def test_unique_step_names_and_lookup(self):
        inner = vt.Pipeline([cs.Brighten()], name="inner")
        pipe = vt.Pipeline([cs.Brighten(), cs.Brighten(2), inner])
        assert pipe.step_names == ["Brighten", "Brighten#2", "inner"]
        assert pipe["Brighten#2"].delta == 2
        assert pipe["inner.Brighten"] is inner[0]
        assert pipe[0] is pipe.stages[0]
        with pytest.raises(KeyError):
            pipe["nope"]
        with pytest.raises(KeyError):
            pipe["Brighten.x"]
        assert list(pipe) == list(pipe.stages)

    def test_invalid_on_error(self):
        with pytest.raises(ValueError):
            vt.Pipeline([], on_error="ignore")  # type: ignore[arg-type]

    def test_nested_pipeline_runs(self):
        pipe = vt.Pipeline([vt.Pipeline([cs.Brighten(10)]), cs.Brighten(5)])
        out = pipe(np.zeros((2, 2, 3), np.uint8))
        assert int(out.image.data.max()) == 15

    def test_produces_folds(self):
        pipe = cs.first_channel() | vt.Convert(color="RGB")
        assert pipe.produces(vt.ImageSpec.bgr_u8()).color is vt.Color.RGB


class TestRun:
    def test_original_preserved_and_input_not_mutated(self):
        img = fx.bgr_u8()
        before = img.data.copy()
        out = (cs.Brighten(50) | cs.invert())(img)
        assert out.original is img
        np.testing.assert_array_equal(img.data, before)

    def test_trace_records(self):
        pipe = vt.Pipeline(
            [
                cs.Brighten(),
                vt.Pipeline([cs.invert()], name="sub"),
                stubs.BoxDetectorStub([[0, 0, 4, 4]]),
            ]
        )
        result = pipe.run(fx.bgr_u8(), trace=True)
        trace = result.trace
        assert [r.name for r in trace] == ["Brighten", "sub", "BoxDetectorStub"]
        assert trace["invert"].role == "filter" and trace["sub"].children
        assert trace[0].allocated_bytes == fx.bgr_u8().data.nbytes
        assert trace["BoxDetectorStub"].results == {"detections": "Detections(n=1)"}
        assert trace.total_ns > 0 and len(trace) == 3
        table = trace.table()
        assert "Brighten" in table and "  invert" in table
        assert '"records"' in trace.to_json()
        with pytest.raises(KeyError):
            trace["missing"]

    def test_no_trace_by_default(self):
        assert vt.Pipeline([cs.Brighten()]).run(fx.bgr_u8()).trace is None

    def test_runtime_accepts_check(self):
        with pytest.raises(vt.SpecMismatchError, match="does not accept") as info:
            stubs.RGBModelStub()(fx.bgr_u8())
        assert "Convert(color='RGB')" in str(info.value)

    def test_unknown_field_at_runtime(self):
        with pytest.raises(vt.SpecMismatchError, match="unknown"):
            (stubs.OpaqueFilterStub() | stubs.RGBModelStub())(fx.rgb_u8())

    def test_check_mode(self, monkeypatch):
        pipe = stubs.LyingFilterStub() | cs.Brighten()
        with pytest.raises(vt.SpecMismatchError, match="declared spec"):
            pipe.run(fx.gray_u8(), check=True)
        monkeypatch.setenv("VISTRAL_DEBUG", "1")
        with pytest.raises(vt.SpecMismatchError):
            pipe(fx.gray_u8())

    def test_check_mode_flags_undeclared_dtype_change(self):
        @vt.filter
        def to_float(img):
            return img.astype(np.float32)

        with pytest.raises(vt.SpecMismatchError, match="dtype"):
            vt.Pipeline([to_float()], check=True)(fx.gray_u8())

    def test_stage_error_wraps_with_context(self):
        with pytest.raises(vt.StageError) as info:
            stubs.FailingStage()(
                vt.Frame(fx.bgr_u8(), source=vt.SourceInfo(uri="x.png", frame_index=0))
            )
        err = info.value
        assert err.stage == "FailingStage" and err.index == 0 and err.source == "x.png"
        assert isinstance(err.__cause__, RuntimeError)

    def test_skip_and_collect(self):
        pipe = vt.Pipeline([stubs.FailingStage(fail_on=[0]), cs.Brighten()], on_error="skip")
        out = pipe.run(frames(1)[0]).frame
        assert out.failed and out.errors[0].stage == "FailingStage"
        assert "boom" in out.errors[0].message
        collect = pipe.map(frames(3), on_error="collect")
        assert [f.failed for f in collect] == [True, False, False]
        assert len(pipe.failures) == 1

    def test_run_on_error_override(self):
        pipe = vt.Pipeline([stubs.FailingStage(fail_on=[0])])
        assert pipe.run(frames(1)[0], on_error="skip").frame.failed


class TestMapAndStream:
    def test_map_batches_where_supported(self):
        counter = stubs.CountingStage()
        model = cs.SumModel()
        pipe = vt.Pipeline([counter, cs.MeanEncoder(), model])
        out = pipe.map(frames(5), batch_size=2)
        assert (
            len(out) == 5 and counter.batch_calls == 2 and counter.calls == 5
        )  # last batch of 1 loops
        assert all("raw_outputs" in f.artifacts for f in out)
        with pytest.raises(ValueError):
            pipe.map([], batch_size=0)

    def test_map_isolates_failures_in_batch(self):
        class Batched(stubs.FailingStage):
            def process_batch(self, fs):
                return [self.process(f) for f in fs]

        pipe = vt.Pipeline([Batched(fail_on=[1]), cs.Brighten()], on_error="skip")
        out = pipe.map(frames(3), batch_size=3)
        assert [f.failed for f in out] == [False, True, False]

    def test_map_order_and_equivalence(self):
        pipe = cs.Brighten(7) | cs.invert()
        looped = [pipe(f) for f in frames(4)]
        mapped = pipe.map(frames(4), batch_size=3)
        for a, b in zip(looped, mapped, strict=True):
            np.testing.assert_array_equal(a.image.data, b.image.data)

    def test_stream_sinks_reset_and_close(self):
        counter = stubs.CountingStage()
        sink = cs.Collect()
        pipe = vt.Pipeline([counter, stubs.FailingStage(fail_on=[2])], on_error="skip")
        items = frames(2, "a") + frames(3, "b")
        out = list(pipe.stream(items, sinks=[sink]))
        assert len(out) == 5 and len(sink.frames) == 4  # frame index 2 of stream b failed
        assert counter.resets == 1 and sink.closed == 1

    def test_lifecycle_context_manager(self):
        counter = stubs.CountingStage()
        pipe = vt.Pipeline([counter])
        with pipe:
            pipe(fx.bgr_u8())
            pipe(fx.bgr_u8())
        assert counter.setups == 1 and counter.closes == 1
        pipe(fx.bgr_u8())
        assert counter.setups == 2
