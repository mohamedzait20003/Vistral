"""Sequential pipelines: run, map (batched), stream, validate, trace."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from vistral.core.config import register_builtin
from vistral.core.errors import SpecMismatchError, StageError, VistralError
from vistral.core.frame import Frame, StageFailure
from vistral.core.image import Image
from vistral.core.results import summarize
from vistral.core.spec import ImageSpec, consistency_violations
from vistral.core.stage import Role, Stage, as_stage, check_runtime_accepts
from vistral.core.trace import (
    ACTIVE_CONVERSIONS,
    ACTIVE_RECORDS,
    Trace,
    TraceRecord,
)
from vistral.core.validation import ValidationContext, ValidationReport, ValidationState

OnError = Literal["raise", "skip", "collect"]
_ON_ERROR = ("raise", "skip", "collect")

# Check mode inherited by nested pipelines during one run.
_CHECK: ContextVar[bool] = ContextVar("vistral_check", default=False)


def _env_check() -> bool:
    return os.environ.get("VISTRAL_DEBUG", "").lower() not in ("", "0", "false", "no")


@dataclass
class RunResult:
    frame: Frame
    trace: Trace | None = None


def _unique_names(stages: Sequence[Stage]) -> list[str]:
    seen: dict[str, int] = {}
    names = []
    for stage in stages:
        base = stage.name
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base}#{seen[base]}")
    return names


def _batches(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _allocated(before: Image, after: Image) -> int:
    if after is before or after.data is before.data:
        return 0
    if isinstance(after.data, np.ndarray) and isinstance(before.data, np.ndarray):
        return 0 if np.may_share_memory(before.data, after.data) else int(after.data.nbytes)
    return 0


def _stage_error(exc: Exception, frame: Frame, step: str, index: int) -> VistralError:
    if isinstance(exc, (StageError, SpecMismatchError)):
        return exc
    return StageError(step, index, exc, source=frame.source.uri, input_spec=frame.image.spec)


@register_builtin("Pipeline")
class Pipeline(Stage):
    """An ordered list of stages, itself usable as a stage (nesting, reuse).

    ``on_error``: ``"raise"`` stops at the first failure; ``"skip"`` records a
    :class:`StageFailure` on the frame, skips its remaining stages, and continues with the next
    frame; ``"collect"`` does the same and also appends failures to :attr:`failures`.
    """

    role = Role.PIPELINE

    def __init__(
        self,
        stages: Sequence[Any],
        *,
        name: str | None = None,
        on_error: OnError = "raise",
        check: bool = False,
    ) -> None:
        if on_error not in _ON_ERROR:
            raise ValueError(f"on_error must be one of {_ON_ERROR}, got {on_error!r}")
        self.stages: tuple[Stage, ...] = tuple(as_stage(s) for s in stages)
        self.on_error: OnError = on_error
        self.check = check
        self.failures: list[StageFailure] = []
        self._names = _unique_names(self.stages)
        self.__dict__["_vt_params"]["stages"] = list(self.stages)
        if name:
            self.__dict__["_vt_name"] = name

    @property
    def anonymous(self) -> bool:
        return "_vt_name" not in self.__dict__ and self.on_error == "raise" and not self.check

    @property
    def step_names(self) -> list[str]:
        return list(self._names)

    def __len__(self) -> int:
        return len(self.stages)

    def __iter__(self) -> Iterator[Stage]:
        return iter(self.stages)

    def __getitem__(self, key: int | str) -> Stage:
        if isinstance(key, int):
            return self.stages[key]
        head, _, rest = key.partition(".")
        if head in self._names:
            stage = self.stages[self._names.index(head)]
            if rest:
                if not isinstance(stage, Pipeline):
                    raise KeyError(key)
                return stage[rest]
            return stage
        raise KeyError(f"no stage named {key!r}; stages: {self._names}")

    def __or__(self, other: Any) -> Pipeline:
        if self.anonymous:
            return Pipeline([*self.stages, other])
        return Pipeline([self, other])

    def __ror__(self, other: Any) -> Pipeline:
        return Pipeline([other, *self.stages]) if self.anonymous else Pipeline([other, self])

    # -- lifecycle ------------------------------------------------------------------------
    def setup(self) -> None:
        for stage in self.stages:
            stage.ensure_setup()

    def close(self) -> None:
        for stage in self.stages:
            stage.shutdown()

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()

    def __enter__(self) -> Pipeline:
        self.ensure_setup()
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    # -- static ---------------------------------------------------------------------------
    def produces(self, spec: ImageSpec) -> ImageSpec:
        for stage in self.stages:
            spec = stage.produces(spec)
        return spec

    def _rebuild(self, stages: list[Stage]) -> Pipeline:
        out = Pipeline(stages, on_error=self.on_error, check=self.check)
        if "_vt_name" in self.__dict__:
            out.__dict__["_vt_name"] = self.__dict__["_vt_name"]
        out.__dict__["_vt_options"] = dict(self.__dict__.get("_vt_options", {}))
        return out

    def infer(
        self, state: ValidationState, ctx: ValidationContext
    ) -> tuple[ValidationState, list[Stage]]:
        new: list[Stage] = []
        changed = False
        for i, (stage, step) in enumerate(zip(self.stages, self._names, strict=True)):
            state, repl = stage.infer(state, ctx.at(i, step))
            changed |= len(repl) != 1 or repl[0] is not stage
            new.extend(repl)
        return state, [self._rebuild(new) if changed else self]

    def validate(
        self,
        input_spec: ImageSpec | Image | Any,
        *,
        auto_convert: bool = False,
        allow_lossy: bool = False,
        raise_on_error: bool = True,
    ) -> ValidationReport:
        """Check the pipeline against ``input_spec`` before running it.

        Raises the first :class:`SpecMismatchError` (with all others attached as notes) unless
        ``raise_on_error=False``. With ``auto_convert=True``, lossless conversions (lossy too if
        ``allow_lossy``) are inserted where possible; use ``report.pipeline`` to run them.
        """
        if isinstance(input_spec, Image):
            spec = input_spec.spec
        elif isinstance(input_spec, ImageSpec):
            spec = input_spec
        elif isinstance(input_spec, dict):
            spec = ImageSpec.coerce(input_spec) or ImageSpec()
        else:
            spec = Frame.coerce(input_spec).image.spec
        ctx = ValidationContext(auto_convert=auto_convert, allow_lossy=allow_lossy)
        state = ValidationState(spec)
        out_state, repl = self.infer(state, ctx)
        report = ValidationReport(
            input_spec=spec,
            output_spec=out_state.spec,
            issues=ctx.issues,
            insertions=ctx.insertions,
            pipeline=repl[0],  # type: ignore[arg-type]
            state=out_state,
        )
        if raise_on_error:
            report.raise_for_errors()
        return report

    # -- execution ------------------------------------------------------------------------
    def process(self, frame: Frame) -> Frame:
        return self._execute(frame, self.on_error)

    def run(
        self,
        x: Any,
        *,
        trace: bool = False,
        check: bool | None = None,
        spec: ImageSpec | None = None,
        on_error: OnError | None = None,
    ) -> RunResult:
        """Run one input (path, array, PIL image, Image or Frame) through the pipeline."""
        frame = Frame.coerce(x, spec=spec)
        self.failures = []
        check_token = _CHECK.set(bool(check) if check is not None else (_CHECK.get() or self.check))
        records: list[TraceRecord] | None = [] if trace else None
        trace_token = ACTIVE_RECORDS.set(records)
        t0 = time.perf_counter_ns()
        try:
            self.ensure_setup()
            out = self._execute(frame, on_error or self.on_error)
        finally:
            ACTIVE_RECORDS.reset(trace_token)
            _CHECK.reset(check_token)
        result_trace = Trace(records, time.perf_counter_ns() - t0) if records is not None else None
        return RunResult(out, result_trace)

    def __call__(self, x: Any, **kwargs: Any) -> Frame:
        return self.run(x, **kwargs).frame

    def map(
        self,
        inputs: Iterable[Any],
        *,
        batch_size: int = 1,
        on_error: OnError | None = None,
        check: bool | None = None,
    ) -> list[Frame]:
        """Run many inputs; stages that implement batching receive lists of ``batch_size``."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        policy = on_error or self.on_error
        self.failures = []
        token = _CHECK.set(bool(check) if check is not None else (_CHECK.get() or self.check))
        try:
            self.ensure_setup()
            out: list[Frame] = []
            for batch in _batches((Frame.coerce(x) for x in inputs), batch_size):
                out.extend(self._execute_batch(batch, policy))
            return out
        finally:
            _CHECK.reset(token)

    def stream(
        self,
        source: Iterable[Any],
        *,
        sinks: Sequence[Stage] = (),
        on_error: OnError | None = None,
        check: bool | None = None,
    ) -> Iterator[Frame]:
        """Yield processed frames one by one; sinks run on each successful frame.

        Stateful stages are :meth:`reset` when ``frame.source.stream_id`` changes.
        """
        policy = on_error or self.on_error
        self.failures = []
        sinks = [as_stage(s) for s in sinks]
        self.ensure_setup()
        for sink in sinks:
            sink.ensure_setup()
        last_stream: Any = _UNSET
        try:
            for item in source:
                frame = Frame.coerce(item)
                stream_id = frame.source.stream_id
                last_stream = self._advance_stream(stream_id, last_stream)
                frame = self._execute_checked(frame, policy, check)
                self._process_sinks(frame, sinks)
                yield frame
        finally:
            for sink in sinks:
                sink.shutdown()

    def _advance_stream(self, stream_id: Any, last_stream: Any) -> Any:
        if stream_id is None:
            return last_stream
        if last_stream is not _UNSET and stream_id != last_stream:
            self.reset()
        return stream_id

    def _execute_checked(self, frame: Frame, policy: OnError, check: bool | None) -> Frame:
        token = _CHECK.set(bool(check) if check is not None else (_CHECK.get() or self.check))
        try:
            return self._execute(frame, policy)
        finally:
            _CHECK.reset(token)

    @staticmethod
    def _process_sinks(frame: Frame, sinks: Sequence[Stage]) -> None:
        if frame.errors:
            return
        for sink in sinks:
            sink.process(frame)

    # -- internals ------------------------------------------------------------------------
    def _execute(self, frame: Frame, policy: OnError) -> Frame:
        for i, stage in enumerate(self.stages):
            frame, failed = self._step(i, stage, frame, policy)
            if failed:
                break
        return frame

    def _execute_batch(self, frames: list[Frame], policy: OnError) -> list[Frame]:
        frames = list(frames)
        failed = [False] * len(frames)
        for i, stage in enumerate(self.stages):
            live = [k for k, bad in enumerate(failed) if not bad]
            if not live:
                break
            if self._try_batch(stage, live, frames, i):
                continue
            for k in live:
                frames[k], failed[k] = self._step(i, stage, frames[k], policy)
        return frames

    def _try_batch(self, stage: Stage, live: list[int], frames: list[Frame], i: int) -> bool:
        if len(live) <= 1 or not stage.batched or ACTIVE_RECORDS.get() is not None:
            return False
        try:
            stage.ensure_setup()
            for k in live:
                check_runtime_accepts(stage, frames[k].image.spec, self._names[i], i)
            outs = stage.process_batch([frames[k].fork() for k in live])
            for k, out in zip(live, outs, strict=True):
                frames[k] = out
            return True
        except Exception:
            return False

    def _step(self, i: int, stage: Stage, frame: Frame, policy: OnError) -> tuple[Frame, bool]:
        step = self._names[i]
        record = self._start_record(i, step, stage, frame)
        conv_token = ACTIVE_CONVERSIONS.set(record.conversions if record is not None else None)
        child_token = ACTIVE_RECORDS.set(record.children if record is not None else None)
        before = frame.image
        t0 = time.perf_counter_ns()
        try:
            out = self._run_stage(stage, step, i, before, frame)
        except Exception as exc:
            if record is not None:
                record.wall_ns = time.perf_counter_ns() - t0
                record.error = f"{type(exc).__name__}: {exc}"
            err = _stage_error(exc, frame, step, i)
            if policy == "raise":
                if err is exc:
                    raise
                raise err from exc
            return self._record_step_failure(frame, err, step, i, policy)
        finally:
            ACTIVE_CONVERSIONS.reset(conv_token)
            ACTIVE_RECORDS.reset(child_token)
        if record is not None:
            self._finish_record(record, out, frame, before, t0)
        return out, False

    @staticmethod
    def _start_record(i: int, step: str, stage: Stage, frame: Frame) -> TraceRecord | None:
        records = ACTIVE_RECORDS.get()
        if records is None:
            return None
        record = TraceRecord(i, step, str(stage.role), in_spec=str(frame.image.spec))
        records.append(record)
        return record

    def _run_stage(self, stage: Stage, step: str, i: int, before: Image, frame: Frame) -> Frame:
        stage.ensure_setup()
        if stage.role is not Role.PIPELINE:
            check_runtime_accepts(stage, frame.image.spec, step, i)
        out = stage.process(frame.fork())
        if not isinstance(out, Frame):
            raise TypeError(f"process() must return a Frame, got {type(out).__name__}")
        if _CHECK.get() or _env_check():
            self._check_output(stage, step, i, before.spec, out)
        return out

    def _record_step_failure(
        self,
        frame: Frame,
        err: VistralError,
        step: str,
        i: int,
        policy: OnError,
    ) -> tuple[Frame, bool]:
        failure = StageFailure(step, i, err)
        frame = frame.fork()
        frame.errors.append(failure)
        if policy == "collect":
            self.failures.append(failure)
        return frame, True

    @staticmethod
    def _finish_record(
        record: TraceRecord, out: Frame, frame: Frame, before: Image, started_ns: int
    ) -> None:
        record.wall_ns = time.perf_counter_ns() - started_ns
        record.out_spec = str(out.image.spec)
        record.allocated_bytes = _allocated(before, out.image)
        record.results = {
            key: summarize(value)
            for key, value in out.results.items()
            if frame.results.get(key) is not value
        }

    def _check_output(
        self, stage: Stage, step: str, i: int, in_spec: ImageSpec, out: Frame
    ) -> None:
        if stage.role in (Role.PIPELINE, Role.CONTROL):
            return
        violations = consistency_violations(out.image.data, out.image.spec)
        expected = stage.produces(in_spec)
        for name in ("layout", "color", "dtype", "range", "backend"):
            want, got = getattr(expected, name), getattr(out.image.spec, name)
            if want is not None and got != want:
                from vistral.core.errors import Violation

                violations.append(Violation(name, got, want, "declared by produces()"))
        if violations:
            raise SpecMismatchError(
                f"stage {i} '{step}' produced an image that does not match its declared spec",
                stage=step,
                index=i,
                violations=tuple(violations),
                suggestion="fix the stage's produces= declaration or its output",
            )


_UNSET = object()
