import json

import numpy as np
import pytest

from tests import custom_stages as cs
from vistral.core import vt
from vistral.core.config import (
    builtin_names,
    from_config_value,
    import_path,
    register_builtin,
    resolve,
    to_config_value,
)
from vistral.core.errors import ConfigError
from vistral.testing import fixtures as fx

ALLOW = ("tests.*",)


class TestFunctionStages:
    def test_filter_runs_and_declares(self):
        out = cs.first_channel()(fx.bgr_u8())
        assert out.image.spec.color is vt.Color.GRAY

    def test_factory_rejects_positional_and_bad_kwargs(self):
        with pytest.raises(TypeError, match="keyword"):
            cs.invert(np.zeros(2))
        with pytest.raises(TypeError, match="invert"):
            cs.invert(bogus=1)
        assert cs.invert.fn(np.zeros(2, np.uint8)).dtype == np.uint8
        assert "factory" in repr(cs.invert)

    def test_uncalled_factory_composes(self):
        pipe = cs.invert | cs.Brighten()
        assert [s.name for s in pipe] == ["invert", "Brighten"]
        pipe2 = cs.Brighten() | cs.invert
        assert pipe2.step_names == ["Brighten", "invert"]

    def test_postprocess_injects_frame_and_keeps_meta(self):
        from vistral.testing.stubs import BoxDetectorStub

        pipe = vt.Pipeline(
            [BoxDetectorStub([[0, 0, 1, 1], [0, 0, 5, 5]]), cs.drop_small(min_width=2)]
        )
        det = pipe(fx.bgr_u8()).results["detections"]
        assert len(det) == 1 and vt.get_meta(det).producer == "BoxDetectorStub"

    def test_frame_stage(self):
        assert cs.tag(label="y")(fx.bgr_u8()).artifacts["tag"] == "y"
        with pytest.raises(vt.StageError, match="must return a Frame"):
            cs.not_a_frame()(fx.bgr_u8())

    def test_postprocess_missing_result(self):
        with pytest.raises(vt.StageError, match="not found"):
            cs.drop_small()(fx.bgr_u8())


class TestConfig:
    def test_capture_includes_defaults(self):
        assert cs.Brighten(5).get_config() == {
            "type": "tests.custom_stages:Brighten",
            "delta": 5,
            "clip": True,
        }

    def test_roundtrip_classes_functions_and_builtins(self):
        stages = [
            cs.Brighten(3, clip=False),
            cs.invert(amount=200),
            vt.Convert(color="RGB", layout="CHW"),
            vt.ToSourceCoords(keys=["detections"]),
            vt.Pipeline([cs.Brighten(), cs.invert()], name="p", on_error="skip"),
            vt.Branch({"a": cs.Brighten(), "b": [cs.invert(), cs.Brighten(1)]}),
            vt.When({"result": "verdict", "equals": "FAIL"}, cs.Brighten(), cs.invert()),
            vt.When(cs.is_fail, cs.Brighten()),
        ]
        for stage in stages:
            config = json.loads(json.dumps(stage.get_config()))
            rebuilt = vt.stage_from_config(config, allow_imports=ALLOW)
            assert rebuilt == stage, config
            assert rebuilt.get_config() == config

    def test_with_options_roundtrip(self):
        stage = cs.Brighten().with_options(name="lift", reads="a", writes="b")
        config = stage.get_config()
        assert config["stage_name"] == "lift"
        rebuilt = vt.stage_from_config(config, allow_imports=ALLOW)
        assert rebuilt.name == "lift" and rebuilt.reads == "a" and rebuilt.writes == "b"

    def test_import_allowlist(self):
        config = cs.Brighten().get_config()
        with pytest.raises(ConfigError, match="not allowed"):
            vt.stage_from_config(config)
        with pytest.raises(ConfigError, match="cannot import"):
            resolve("tests.custom_stages:Nope", ("tests.*",))
        with pytest.raises(ConfigError, match="unknown component"):
            resolve("Nope")
        assert resolve("tests.custom_stages:Brighten", ("tests.custom_stages",)) is cs.Brighten

    def test_bad_configs(self):
        with pytest.raises(ConfigError, match="'type'"):
            vt.stage_from_config({})
        with pytest.raises(ConfigError, match="cannot build"):
            vt.stage_from_config({"type": "Convert", "colour": "RGB"})
        with pytest.raises(ConfigError, match="not a stage"):
            vt.stage_from_config({"type": "tests.custom_stages:make_number"}, allow_imports=ALLOW)

    def test_unserializable_values(self):
        with pytest.raises(ConfigError, match="not config-serializable"):
            cs.Unserializable(object()).get_config()
        assert cs.Unserializable(object()) != cs.Unserializable(object())
        with pytest.raises(ConfigError, match="lambda"):
            to_config_value(lambda f: True)
        with pytest.raises(ConfigError, match="keys"):
            to_config_value({1: 2})

    def test_value_conversions(self):
        from pathlib import Path

        assert to_config_value(Path("a/b")) in ("a/b", "a\\b")
        assert to_config_value(np.float32(1.5)) == 1.5
        assert to_config_value(np.array([[1, 2]])) == [[1, 2]]
        assert to_config_value({"b", "a"}) == ["a", "b"]
        assert to_config_value(vt.Color.RGB) == "RGB"
        assert from_config_value({"$callable": "tests.custom_stages:is_fail"}, ALLOW) is cs.is_fail
        assert from_config_value({"type": "not-a-stage", "x": [1]}) == {
            "type": "not-a-stage",
            "x": [1],
        }

    def test_registry(self):
        assert {"Convert", "Pipeline", "Branch", "When", "ToSourceCoords"} <= set(builtin_names())
        with pytest.raises(ValueError):
            register_builtin("Convert")(object)
        with pytest.raises(ConfigError):
            import_path(object())


class TestRoleBases:
    def test_encoder_model_decoder_chain(self):
        pipe = cs.MeanEncoder() | cs.SumModel() | cs.ScoreDecoder()
        frame = pipe(fx.bgr_u8())
        score = frame.results["score"]
        assert isinstance(score, vt.Text)
        meta = vt.get_meta(score)
        assert meta.model_id == "sum-model" and meta.coord_space == "source"

    def test_model_without_inputs_errors(self):
        with pytest.raises(vt.StageError, match="Encoder"):
            cs.SumModel()(fx.bgr_u8())
        with pytest.raises(vt.StageError, match="Model before"):
            cs.ScoreDecoder()(fx.bgr_u8())

    def test_not_implemented_hooks(self):
        for base in (
            vt.Filter,
            vt.Encoder,
            vt.Model,
            vt.Decoder,
            vt.Postprocess,
            vt.Sink,
            vt.Stage,
        ):
            stage = base()
            frame = fx.frame()
            frame.artifacts.update(model_inputs=1, raw_outputs=1)
            frame.results["detections"] = 1
            with pytest.raises(NotImplementedError):
                stage.process(frame)

    def test_stage_repr_and_hash(self):
        b = cs.Brighten(2)
        assert repr(b) == "Brighten(delta=2, clip=True)"
        assert len({b, b}) == 1
        assert b == cs.Brighten(2) and b != cs.Brighten(3)

    def test_as_stage_errors(self):
        with pytest.raises(TypeError, match="plain function"):
            vt.Pipeline([lambda x: x])
        with pytest.raises(TypeError, match="not a stage"):
            vt.Pipeline([3])

    def test_convert_requires_target(self):
        with pytest.raises(ValueError):
            vt.Convert()
        assert vt.Convert(color="RGB").name == "Convert(color=RGB)"
