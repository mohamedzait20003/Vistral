"""The ``Stage`` protocol, role base classes, and function decorators.

Every stage reads and writes a :class:`Frame`. Roles separate responsibilities that are easy
to conflate: filters (image -> image), encoders (image -> model inputs), models (inputs -> raw
outputs), decoders (raw outputs -> result), postprocessors (result -> result) and sinks.
"""

from __future__ import annotations

import copy
import functools
import inspect
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

from vistral.core.config import to_config_value, type_name
from vistral.core.errors import ConfigError, SpecMismatchError
from vistral.core.frame import Frame
from vistral.core.image import Image
from vistral.core.results import get_meta, set_meta, stamp
from vistral.core.spec import Accepts, ImageSpec

if TYPE_CHECKING:
    from vistral.core.pipeline import Pipeline
    from vistral.core.validation import ValidationContext, ValidationState


class Role(StrEnum):
    SOURCE = "source"
    FILTER = "filter"
    ENCODER = "encoder"
    MODEL = "model"
    DECODER = "decoder"
    POSTPROCESS = "postprocess"
    SINK = "sink"
    PIPELINE = "pipeline"
    CONTROL = "control"


IMAGE = "image"  # Model.reads value meaning "consume frame.image.data directly"


def _capture_init(init: Callable[..., None]) -> Callable[..., None]:
    sig = inspect.signature(init)

    @functools.wraps(init)
    def wrapper(self: Stage, *args: Any, **kwargs: Any) -> None:
        if "_vt_params" not in self.__dict__:
            self.__dict__["_vt_params"] = _captured_init_params(sig, self, args, kwargs)
        init(self, *args, **kwargs)

    wrapper._vt_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def _captured_init_params(
    signature: inspect.Signature, instance: Stage, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    bound = signature.bind(instance, *args, **kwargs)
    bound.apply_defaults()
    params: dict[str, Any] = {}
    for name, value in list(bound.arguments.items())[1:]:
        _store_init_param(params, name, value, signature.parameters[name].kind)
    return params


def _store_init_param(params: dict[str, Any], name: str, value: Any, kind: Any) -> None:
    if kind is inspect.Parameter.VAR_KEYWORD:
        params.update(value)
    elif kind is inspect.Parameter.VAR_POSITIONAL:
        if value:
            params[name] = list(value)
    else:
        params[name] = value


class Stage:
    """Base class for all pipeline stages.

    Subclasses implement :meth:`process` (or a role hook such as ``Filter.apply``). Constructor
    arguments are captured automatically for :meth:`get_config`; keep them config-serializable
    (scalars, lists, dicts, enums, paths, specs, stages).
    """

    role: ClassVar[Role] = Role.FILTER
    accepts: Accepts | None = None
    changes_geometry: ClassVar[bool] = False
    requires_source_coords: tuple[str, ...] = ()
    reads: str | None = None
    writes: str | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        init = cls.__dict__.get("__init__")
        if init is not None and not getattr(init, "_vt_wrapped", False):
            cls.__init__ = _capture_init(init)  # type: ignore[method-assign]
        raw = cls.__dict__.get("accepts")
        if raw is not None and not isinstance(raw, Accepts):
            cls.accepts = Accepts.coerce(raw)

    # -- identity & config ----------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.__dict__.get("_vt_name") or self.default_name()

    def default_name(self) -> str:
        return type(self).__name__

    def with_options(
        self, *, name: str | None = None, reads: str | None = None, writes: str | None = None
    ) -> Self:
        """Copy of this stage with a different step name or result/artifact keys."""
        out = copy.copy(self)
        options = dict(self.__dict__.get("_vt_options", {}))
        if name is not None:
            out.__dict__["_vt_name"] = name
            options["stage_name"] = name
        if reads is not None:
            out.reads = reads
            options["reads"] = reads
        if writes is not None:
            out.writes = writes
            options["writes"] = writes
        out.__dict__["_vt_options"] = options
        out.__dict__["_vt_ready"] = False
        return out

    def config_type(self) -> str:
        return type_name(type(self))

    def config_params(self) -> dict[str, Any]:
        return dict(self.__dict__.get("_vt_params", {}))

    def get_config(self) -> dict[str, Any]:
        """JSON-compatible config that rebuilds this stage via ``stage_from_config``."""
        out: dict[str, Any] = {"type": self.config_type()}
        for key, value in self.config_params().items():
            out[key] = to_config_value(value, f"{self.name}.{key}")
        out.update(self.__dict__.get("_vt_options", {}))
        return out

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        try:
            return self.get_config() == other.get_config()  # type: ignore[union-attr]
        except ConfigError:
            return self is other

    __hash__ = object.__hash__

    # -- spec & effects -------------------------------------------------------------------
    def produces(self, spec: ImageSpec) -> ImageSpec:
        """Spec of ``frame.image`` after this stage, given the spec before. Default: unchanged."""
        return spec

    def reads_artifacts(self) -> tuple[str, ...]:
        return ()

    def writes_artifacts(self) -> tuple[str, ...]:
        return ()

    def reads_results(self) -> tuple[str, ...]:
        return ()

    def writes_results(self) -> tuple[str, ...]:
        return ()

    # -- execution ------------------------------------------------------------------------
    def process(self, frame: Frame) -> Frame:
        raise NotImplementedError(f"{type(self).__name__} must implement process()")

    def process_batch(self, frames: list[Frame]) -> list[Frame]:
        return [self.process(f) for f in frames]

    @property
    def batched(self) -> bool:
        return type(self).process_batch is not Stage.process_batch

    def setup(self) -> None:
        """Acquire resources (load models). Called once before first use."""

    def close(self) -> None:
        """Release resources. Must be idempotent."""

    def reset(self) -> None:
        """Clear per-stream state (stateful stages). Called at stream boundaries."""

    def ensure_setup(self) -> None:
        if not self.__dict__.get("_vt_ready"):
            self.setup()
            self.__dict__["_vt_ready"] = True

    def shutdown(self) -> None:
        """Close and mark not-ready, so the next use calls :meth:`setup` again."""
        self.close()
        self.__dict__["_vt_ready"] = False

    # -- validation -----------------------------------------------------------------------
    def infer(
        self, state: ValidationState, ctx: ValidationContext
    ) -> tuple[ValidationState, list[Stage]]:
        from vistral.core.validation import infer_stage

        return infer_stage(self, state, ctx)

    # -- composition ----------------------------------------------------------------------
    def __or__(self, other: Any) -> Pipeline:
        from vistral.core.pipeline import Pipeline

        return Pipeline([self, other])

    def __ror__(self, other: Any) -> Pipeline:
        from vistral.core.pipeline import Pipeline

        return Pipeline([other, self])

    def __call__(self, x: Any, **kwargs: Any) -> Frame:
        """Run this stage alone, as a one-stage pipeline."""
        from vistral.core.pipeline import Pipeline

        return Pipeline([self])(x, **kwargs)

    def __repr__(self) -> str:
        try:
            params = ", ".join(f"{k}={v!r}" for k, v in self.config_params().items())
        except Exception:  # pragma: no cover - repr must never fail
            params = "..."
        return f"{self.name}({params})"


def _changed_fields(before: ImageSpec, after: ImageSpec) -> dict[str, Any]:
    return {
        f: getattr(after, f)
        for f in ("color", "range", "layout")
        if getattr(after, f) != getattr(before, f)
    }


def apply_image_output(stage: Stage, frame: Frame, data: Any) -> Frame:
    """Store filter output, deriving its spec from the data and from ``stage.produces``."""
    before = frame.image.spec
    declared = _changed_fields(before, stage.produces(before))
    frame.image = frame.image.with_data(data, **declared)
    return frame


class Filter(Stage):
    """Image -> image. Implement :meth:`apply` on the raw array, or override :meth:`process`."""

    role = Role.FILTER

    def apply(self, data: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement apply()")

    def process(self, frame: Frame) -> Frame:
        return apply_image_output(self, frame, self.apply(frame.image.data))


class Encoder(Stage):
    """Image -> model inputs in ``frame.artifacts[writes]``. Not media decoding."""

    role = Role.ENCODER
    writes = "model_inputs"

    def encode(self, image: Image, frame: Frame) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement encode()")

    def process(self, frame: Frame) -> Frame:
        frame.artifacts[self.writes or "model_inputs"] = self.encode(frame.image, frame)
        return frame

    def writes_artifacts(self) -> tuple[str, ...]:
        return (self.writes or "model_inputs",)


class Model(Stage):
    """Model inputs -> raw outputs. Inference only.

    ``reads = "image"`` makes the model consume ``frame.image.data`` directly; ``accepts`` then
    describes that image.
    """

    role = Role.MODEL
    reads = "model_inputs"
    writes = "raw_outputs"

    def predict(self, inputs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement predict()")

    def predict_batch(self, inputs: list[Any]) -> list[Any]:
        return [self.predict(i) for i in inputs]

    def _inputs(self, frame: Frame) -> Any:
        key = self.reads or "model_inputs"
        if key == IMAGE:
            return frame.image.data
        try:
            return frame.artifacts[key]
        except KeyError:
            raise KeyError(
                f"artifact {key!r} not found; add an Encoder before this model, "
                f"or set reads='image' to consume the image directly"
            ) from None

    def process(self, frame: Frame) -> Frame:
        frame.artifacts[self.writes or "raw_outputs"] = self.predict(self._inputs(frame))
        return frame

    def process_batch(self, frames: list[Frame]) -> list[Frame]:
        outputs = self.predict_batch([self._inputs(f) for f in frames])
        for frame, out in zip(frames, outputs, strict=True):
            frame.artifacts[self.writes or "raw_outputs"] = out
        return frames

    @property
    def batched(self) -> bool:
        return type(self).predict_batch is not Model.predict_batch

    def reads_artifacts(self) -> tuple[str, ...]:
        key = self.reads or "model_inputs"
        return () if key == IMAGE else (key,)

    def writes_artifacts(self) -> tuple[str, ...]:
        return (self.writes or "raw_outputs",)


class Decoder(Stage):
    """Raw outputs -> a result in ``frame.results[writes]``, stamped with provenance.

    Coordinates in the result refer to ``frame.image`` at this point; the stamp records the
    transforms since the source so :class:`ToSourceCoords` can map them back.
    """

    role = Role.DECODER
    reads = "raw_outputs"
    writes = "result"

    def decode(self, raw: Any, frame: Frame) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement decode()")

    def provenance(self) -> dict[str, Any]:
        """Extra provenance (``model_id``, ``model_revision``, ...) for the result stamp."""
        return {}

    def process(self, frame: Frame) -> Frame:
        key = self.reads or "raw_outputs"
        try:
            raw = frame.artifacts[key]
        except KeyError:
            raise KeyError(f"artifact {key!r} not found; add a Model before this decoder") from None
        result = self.decode(raw, frame)
        frame.results[self.writes or "result"] = stamp(
            result, frame, producer=self.name, **self.provenance()
        )
        return frame

    def reads_artifacts(self) -> tuple[str, ...]:
        return (self.reads or "raw_outputs",)

    def writes_results(self) -> tuple[str, ...]:
        return (self.writes or "result",)


class Postprocess(Stage):
    """Result -> result. Reads ``frame.results[reads]``; writes ``writes`` (default: same key)."""

    role = Role.POSTPROCESS
    reads = "detections"

    def apply(self, result: Any, frame: Frame) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement apply()")

    def process(self, frame: Frame) -> Frame:
        key = self.reads or "detections"
        if key not in frame.results:
            raise KeyError(f"result {key!r} not found; available: {sorted(frame.results)}")
        before = frame.results[key]
        out = self.apply(before, frame)
        meta = get_meta(before)
        if meta is not None and get_meta(out) is None:
            out = set_meta(out, meta)
        frame.results[self.writes or key] = out
        return frame

    def reads_results(self) -> tuple[str, ...]:
        return (self.reads or "detections",)

    def writes_results(self) -> tuple[str, ...]:
        return (self.writes or self.reads or "detections",)


class Sink(Stage):
    """Side effect (write a file, call back). Passes the frame through unchanged."""

    role = Role.SINK

    def write(self, frame: Frame) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement write()")

    def process(self, frame: Frame) -> Frame:
        self.write(frame)
        return frame


# -- function stages --------------------------------------------------------------------------


def _spec_transform(produces: Any) -> Callable[[ImageSpec], ImageSpec] | None:
    if produces is None:
        return None
    if callable(produces) and not isinstance(produces, ImageSpec):
        return cast(Callable[[ImageSpec], ImageSpec], produces)
    spec = ImageSpec.coerce(produces)
    assert spec is not None
    overrides = spec.known()
    return lambda s: s.replace(**overrides)


class FunctionStageFactory:
    """What ``@vistral.filter`` / ``@vistral.postprocess`` / ``@vistral.stage`` return.

    Calling it with keyword parameters returns a stage; ``.fn`` is the original function.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        kind: Role,
        *,
        accepts: Any = None,
        produces: Any = None,
        reads: str | None = None,
        writes: str | None = None,
        requires_source_coords: tuple[str, ...] = (),
        name: str | None = None,
    ) -> None:
        functools.update_wrapper(self, fn)
        self.fn = fn
        self.kind = kind
        self.accepts = Accepts.coerce(accepts)
        self.produces = _spec_transform(produces)
        self.reads = reads
        self.writes = writes
        self.requires_source_coords = tuple(requires_source_coords)
        self.stage_name = name or fn.__name__
        self.signature = inspect.signature(fn)
        self.takes_frame = "frame" in self.signature.parameters

    def __call__(self, *args: Any, **params: Any) -> FunctionStage:
        if args:
            raise TypeError(
                f"{self.stage_name}() takes keyword parameters and returns a stage; "
                f"call {self.stage_name}.fn(...) to run the function directly"
            )
        try:
            self.signature.bind_partial(None, **params)
        except TypeError as exc:
            raise TypeError(f"{self.stage_name}(): {exc}") from None
        return FunctionStage(self, params)

    def __or__(self, other: Any) -> Pipeline:
        return self() | other

    def __ror__(self, other: Any) -> Pipeline:
        return other | self()

    def __repr__(self) -> str:
        return f"<vistral {self.kind} factory {self.stage_name}>"


class FunctionStage(Stage):
    """A stage built from a decorated function. See :class:`FunctionStageFactory`."""

    def __init__(self, factory: FunctionStageFactory, params: Mapping[str, Any]) -> None:
        self.factory = factory
        self.params = dict(params)
        self.__dict__["_vt_params"] = dict(params)
        self.accepts = factory.accepts
        self.requires_source_coords = factory.requires_source_coords
        self.reads = factory.reads
        self.writes = factory.writes

    @property
    def role(self) -> Role:  # type: ignore[override]
        return self.factory.kind

    def default_name(self) -> str:
        return self.factory.stage_name

    def config_type(self) -> str:
        return type_name(self.factory)

    def produces(self, spec: ImageSpec) -> ImageSpec:
        fn = self.factory.produces
        return spec if fn is None else fn(spec)

    def _kwargs(self, frame: Frame) -> dict[str, Any]:
        return {**self.params, "frame": frame} if self.factory.takes_frame else dict(self.params)

    def process(self, frame: Frame) -> Frame:
        fn = self.factory.fn
        kind = self.factory.kind
        if kind is Role.FILTER:
            return apply_image_output(self, frame, fn(frame.image.data, **self._kwargs(frame)))
        if kind is Role.POSTPROCESS:
            key = self.reads or "detections"
            if key not in frame.results:
                raise KeyError(f"result {key!r} not found; available: {sorted(frame.results)}")
            before = frame.results[key]
            out = fn(before, **self._kwargs(frame))
            meta = get_meta(before)
            if meta is not None and get_meta(out) is None:
                out = set_meta(out, meta)
            frame.results[self.writes or key] = out
            return frame
        out = fn(frame, **self.params)
        if not isinstance(out, Frame):
            raise TypeError(f"{self.name} must return a Frame, got {type(out).__name__}")
        return out

    def reads_results(self) -> tuple[str, ...]:
        reads: list[str] = []
        if self.factory.kind is Role.POSTPROCESS:
            reads.append(self.reads or "detections")
        return tuple(reads)

    def writes_results(self) -> tuple[str, ...]:
        writes: list[str] = []
        if self.factory.kind is Role.POSTPROCESS:
            writes.append(self.writes or self.reads or "detections")
        elif self.writes:
            writes.append(self.writes)
        return tuple(writes)


def filter_stage(
    fn: Callable[..., Any] | None = None,
    *,
    accepts: Any = None,
    produces: Any = None,
    name: str | None = None,
) -> Any:
    """Decorate ``fn(array, **params) -> array`` as a filter stage factory.

    ``produces`` is an ``ImageSpec`` (or dict) of fields the output changes, e.g.
    ``{"color": "GRAY", "layout": "HW"}``, or a callable ``spec -> spec``. Observable fields
    (dtype, size, layout) are always re-derived from the returned array.
    """

    def deco(f: Callable[..., Any]) -> FunctionStageFactory:
        return FunctionStageFactory(f, Role.FILTER, accepts=accepts, produces=produces, name=name)

    return deco(fn) if fn is not None else deco


def postprocess_stage(
    fn: Callable[..., Any] | None = None,
    *,
    reads: str = "detections",
    writes: str | None = None,
    requires_source_coords: bool = False,
    name: str | None = None,
) -> Any:
    """Decorate ``fn(result, **params) -> result`` (optionally taking ``frame=``)."""

    def deco(f: Callable[..., Any]) -> FunctionStageFactory:
        return FunctionStageFactory(
            f,
            Role.POSTPROCESS,
            reads=reads,
            writes=writes,
            requires_source_coords=(reads,) if requires_source_coords else (),
            name=name,
        )

    return deco(fn) if fn is not None else deco


def frame_stage(
    fn: Callable[..., Any] | None = None,
    *,
    accepts: Any = None,
    produces: Any = None,
    writes: str | None = None,
    name: str | None = None,
) -> Any:
    """Decorate ``fn(frame, **params) -> Frame`` as a general stage factory."""

    def deco(f: Callable[..., Any]) -> FunctionStageFactory:
        return FunctionStageFactory(
            f, Role.CONTROL, accepts=accepts, produces=produces, writes=writes, name=name
        )

    return deco(fn) if fn is not None else deco


def as_stage(obj: Any) -> Stage:
    """Coerce a stage, an uncalled decorated function, or a list of stages to a stage."""
    if isinstance(obj, Stage):
        return obj
    if isinstance(obj, FunctionStageFactory):
        return obj()
    if isinstance(obj, (list, tuple)):
        from vistral.core.pipeline import Pipeline

        return Pipeline(list(obj))
    if callable(obj):
        raise TypeError(
            f"{getattr(obj, '__name__', obj)!r} is a plain function; decorate it with "
            "@vistral.filter / @vistral.postprocess / @vistral.stage (or use cv.wrap in M2)"
        )
    raise TypeError(f"{obj!r} is not a stage")


def check_runtime_accepts(stage: Stage, spec: ImageSpec, step: str, index: int | str) -> None:
    """Raise if ``spec`` violates ``stage.accepts`` or leaves a constrained field unknown."""
    accepts = stage.accepts
    if accepts is None:
        return
    violations, unknown = accepts.check(spec)
    if not violations and not unknown:
        return
    from vistral.core.validation import conversion_suggestion

    suggestion = conversion_suggestion(spec, accepts, unknown=unknown)
    raise SpecMismatchError(
        f"stage {index} '{step}' does not accept the image it receives ({spec})",
        stage=step,
        index=index,
        violations=tuple(violations),
        suggestion=suggestion,
    )
