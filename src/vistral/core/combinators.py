"""``Branch`` (several sub-pipelines on one frame) and ``When`` (conditional routing)."""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce
from typing import Any, cast

from vistral.core.config import register_builtin
from vistral.core.frame import Frame
from vistral.core.pipeline import Pipeline
from vistral.core.spec import SPEC_FIELDS, ImageSpec
from vistral.core.stage import Role, Stage, as_stage
from vistral.core.validation import ValidationContext, ValidationState


def _as_pipeline(obj: Any, name: str) -> Pipeline:
    stage = as_stage(obj)
    return stage if isinstance(stage, Pipeline) else Pipeline([stage], name=name)


@register_builtin("Branch")
class Branch(Stage):
    """Run sub-pipelines on the same input frame and merge their new results.

    The image passes through unchanged. With ``merge="namespace"`` a result ``detections``
    from branch ``a`` is stored as ``a.detections``.
    """

    role = Role.CONTROL

    def __init__(self, branches: Mapping[str, Any], *, merge: str = "namespace") -> None:
        if merge != "namespace":
            raise ValueError("only merge='namespace' is supported")
        if not branches:
            raise ValueError("Branch needs at least one branch")
        self.branches: dict[str, Pipeline] = {
            name: _as_pipeline(stage, name) for name, stage in branches.items()
        }
        self.merge = merge
        self.__dict__["_vt_params"]["branches"] = dict(self.branches)

    def setup(self) -> None:
        for pipe in self.branches.values():
            pipe.ensure_setup()

    def close(self) -> None:
        for pipe in self.branches.values():
            pipe.shutdown()

    def reset(self) -> None:
        for pipe in self.branches.values():
            pipe.reset()

    def process(self, frame: Frame) -> Frame:
        out = frame.fork()
        for name, pipe in self.branches.items():
            sub = pipe._execute(frame.fork(), pipe.on_error)
            for key, value in sub.results.items():
                if frame.results.get(key) is not value:
                    out.results[f"{name}.{key}"] = value
            out.errors.extend(sub.errors[len(frame.errors) :])
            if sub.errors[len(frame.errors) :] and pipe.on_error == "raise":
                break
        return out

    def writes_results(self) -> tuple[str, ...]:
        return ()

    def infer(
        self, state: ValidationState, ctx: ValidationContext
    ) -> tuple[ValidationState, list[Stage]]:
        merged = state.copy()
        new: dict[str, Pipeline] = {}
        changed = False
        for name, pipe in self.branches.items():
            sub, repl = pipe.infer(state.copy(), ctx.at_branch(name))
            new[name] = repl[0]  # type: ignore[assignment]
            changed |= repl[0] is not pipe
            for key, space in sub.result_spaces.items():
                if state.result_spaces.get(key) != space or key not in state.result_spaces:
                    merged.result_spaces[f"{name}.{key}"] = space
        if changed:
            rebuilt = Branch(new, merge=self.merge)
            rebuilt.__dict__.update(
                {k: v for k, v in self.__dict__.items() if k in ("_vt_name", "_vt_options")}
            )
            return merged, [rebuilt]
        return merged, [self]


@dataclass(frozen=True)
class Condition:
    """Structured, serializable condition on a result (never an evaluated expression).

    ``Condition(result="verdict", equals="FAIL")`` compares ``results["verdict"].status`` (or
    the result itself if it has no ``status``); ``attr`` picks another attribute;
    ``in_`` tests membership; ``exists`` tests presence.
    """

    result: str
    equals: Any = None
    in_: tuple[Any, ...] | None = None
    attr: str | None = None
    exists: bool | None = None

    @classmethod
    def coerce(cls, value: Any) -> Condition | Callable[[Frame], bool]:
        if isinstance(value, Condition):
            return value
        if isinstance(value, Mapping):
            data = dict(value)
            if "in" in data:
                data["in_"] = tuple(data.pop("in"))
            return cls(**data)
        if callable(value):
            return cast(Callable[[Frame], bool], value)
        raise TypeError(f"cannot interpret {value!r} as a condition")

    def __call__(self, frame: Frame) -> bool:
        present = self.result in frame.results
        if self.exists is not None and present != self.exists:
            return False
        if not present:
            return self.exists is False
        value = frame.results[self.result]
        if self.attr:
            value = reduce(getattr, self.attr.split("."), value)
        elif hasattr(value, "status"):
            value = value.status
        if self.in_ is not None:
            return value in self.in_ or str(value) in {str(v) for v in self.in_}
        if self.equals is not None:
            return bool(operator.eq(value, self.equals) or str(value) == str(self.equals))
        return True

    def to_config(self) -> dict[str, Any]:
        out: dict[str, Any] = {"result": self.result}
        if self.equals is not None:
            out["equals"] = self.equals
        if self.in_ is not None:
            out["in"] = list(self.in_)
        if self.attr is not None:
            out["attr"] = self.attr
        if self.exists is not None:
            out["exists"] = self.exists
        return out


def _merge_specs(a: ImageSpec, b: ImageSpec) -> tuple[ImageSpec, list[str]]:
    diff = [f for f in SPEC_FIELDS if getattr(a, f) != getattr(b, f)]
    return a.replace(**{f: None for f in diff}), diff


@register_builtin("When")
class When(Stage):
    """Run ``then`` if ``condition(frame)`` is true, else ``otherwise`` (or nothing)."""

    role = Role.CONTROL

    def __init__(self, condition: Any, then: Any, otherwise: Any = None) -> None:
        self.condition = Condition.coerce(condition)
        self.then = _as_pipeline(then, "then")
        self.otherwise = _as_pipeline(otherwise, "otherwise") if otherwise is not None else None
        params = self.__dict__["_vt_params"]
        params.update(condition=self.condition, then=self.then, otherwise=self.otherwise)

    def _branches(self) -> list[Pipeline]:
        return [p for p in (self.then, self.otherwise) if p is not None]

    def setup(self) -> None:
        for pipe in self._branches():
            pipe.ensure_setup()

    def close(self) -> None:
        for pipe in self._branches():
            pipe.shutdown()

    def reset(self) -> None:
        for pipe in self._branches():
            pipe.reset()

    def process(self, frame: Frame) -> Frame:
        branch = self.then if self.condition(frame) else self.otherwise
        if branch is None:
            return frame
        return branch._execute(frame, branch.on_error)

    def infer(
        self, state: ValidationState, ctx: ValidationContext
    ) -> tuple[ValidationState, list[Stage]]:
        then_state, then_repl = self.then.infer(state.copy(), ctx.at_branch("then"))
        if self.otherwise is not None:
            other_state, other_repl = self.otherwise.infer(state.copy(), ctx.at_branch("otherwise"))
        else:
            other_state, other_repl = state.copy(), [None]
        spec, diff = _merge_specs(then_state.spec, other_state.spec)
        if diff:
            ctx.warn(f"branches produce different {', '.join(diff)}; treated as unknown downstream")
        merged = ValidationState(
            spec,
            then_state.geometry,
            dict(other_state.result_spaces),
            then_state.artifacts & other_state.artifacts,
        )
        for key, space in then_state.result_spaces.items():
            if merged.result_spaces.get(key, space) != space:
                ctx.warn(f"result {key!r} is in different coordinate spaces in the two branches")
            merged.result_spaces[key] = space
        if then_state.geometry != other_state.geometry:
            ctx.warn("branches change geometry differently")
        if then_repl[0] is not self.then or other_repl[0] is not self.otherwise:
            rebuilt = When(self.condition, then_repl[0], other_repl[0])
            return merged, [rebuilt]
        return merged, [self]
