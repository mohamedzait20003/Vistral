"""Contract checks any stage (built-in or custom) should pass."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vistral.core.config import stage_from_config
from vistral.core.frame import Frame, SourceInfo
from vistral.core.image import Image
from vistral.core.pipeline import Pipeline
from vistral.core.results import result_to_dict
from vistral.core.stage import Stage, as_stage


@dataclass
class CheckReport:
    stage: str
    passed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"check_stage({self.stage}): passed {self.passed}, skipped {self.skipped}"


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.shape == b.shape
            and (bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f")))
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    try:
        return bool(a == b)
    except ValueError:  # array-like comparisons
        return _same(np.asarray(a), np.asarray(b))


def _strip_times(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _strip_times(v) for k, v in d.items() if k != "created_ns"}
    if isinstance(d, list):
        return [_strip_times(v) for v in d]
    return d


def _frames_equal(a: Frame, b: Frame) -> bool:
    if not _same(a.image.data, b.image.data) or a.image.spec != b.image.spec:
        return False
    if a.results.keys() != b.results.keys():
        return False
    return all(
        _same(
            _strip_times(result_to_dict(a.results[k])), _strip_times(result_to_dict(b.results[k]))
        )
        for k in a.results
    )


def check_stage(
    stage: Any,
    sample: Image | Frame | Any,
    *,
    allow_imports: Iterable[str] = ("*",),
    roundtrip: bool = True,
    deterministic: bool = True,
) -> CheckReport:
    """Run contract checks; raises ``AssertionError`` describing the first failure.

    Checks: the stage accepts the sample; its output matches its declared spec (check mode);
    it does not mutate its input; it is deterministic; ``get_config()`` is JSON-serializable
    and rebuilds an equal stage; ``close()`` is idempotent; batch processing equals looping.
    """
    stage = as_stage(stage)
    report = CheckReport(stage.name)
    frame = Frame.coerce(sample)
    pipe = Pipeline([stage], check=True)

    if stage.accepts is not None:
        violations, unknown = stage.accepts.check(frame.image.spec)
        assert not violations and not unknown, (
            f"sample {frame.image.spec} is not accepted by {stage.name}: {violations or unknown}; "
            "pass a sample the stage accepts"
        )
    report.passed.append("accepts sample")

    before = frame.image.data.copy() if isinstance(frame.image.data, np.ndarray) else None
    first = pipe.run(frame.fork()).frame
    report.passed.append("declared spec matches output")
    if before is not None:
        assert np.array_equal(before, frame.image.data), f"{stage.name} mutated its input array"
        report.passed.append("does not mutate input")

    if deterministic:
        second = pipe.run(frame.fork()).frame
        assert _frames_equal(first, second), f"{stage.name} is not deterministic"
        report.passed.append("deterministic")
    else:
        report.skipped.append("deterministic")

    if roundtrip:
        config = stage.get_config()
        json.dumps(config)
        rebuilt = stage_from_config(json.loads(json.dumps(config)), allow_imports=allow_imports)
        assert rebuilt == stage, f"config round-trip changed the stage: {config}"
        assert rebuilt.get_config() == config, "config is not stable across a round-trip"
        if deterministic:
            again = Pipeline([rebuilt], check=True).run(frame.fork()).frame
            assert _frames_equal(first, again), "rebuilt stage produces different output"
        report.passed.append("config round-trip")
    else:
        report.skipped.append("config round-trip")

    stage.close()
    stage.close()
    stage.shutdown()
    report.passed.append("close() idempotent")

    frames = [
        Frame(frame.image, source=SourceInfo(frame_index=0)),
        Frame(frame.image, source=SourceInfo(frame_index=1)),
    ]
    looped = [pipe.run(f.fork()).frame for f in frames]
    batched = pipe.map([f.fork() for f in frames], batch_size=2)
    if deterministic:
        assert all(_frames_equal(a, b) for a, b in zip(looped, batched, strict=True)), (
            f"{stage.name}: process_batch differs from process"
        )
        report.passed.append("batch == loop")
    return report


def is_stage(obj: Any) -> bool:
    return isinstance(obj, Stage)
