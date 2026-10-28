"""Static validation: propagate specs, artifacts and result coordinate spaces through a pipeline.

``Pipeline.validate(input_spec)`` walks the stages *before any data is processed* and reports
contract violations with suggested fixes, optionally inserting lossless conversions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from vistral.core.convert import Converter, describe_path, plan
from vistral.core.errors import CoordinateSpaceError, SpecMismatchError, VistralError
from vistral.core.spec import SPEC_FIELDS, Accepts, ImageSpec

if TYPE_CHECKING:
    from vistral.core.pipeline import Pipeline
    from vistral.core.stage import Stage


@dataclass
class ValidationState:
    spec: ImageSpec
    geometry: tuple[str, ...] = ()
    result_spaces: dict[str, str] = field(default_factory=dict)
    artifacts: set[str] = field(default_factory=set)

    def copy(self) -> ValidationState:
        return ValidationState(
            self.spec, self.geometry, dict(self.result_spaces), set(self.artifacts)
        )

    @property
    def coord_space(self) -> str:
        return "source" if not self.geometry else f"stage:{self.geometry[-1]}"


@dataclass(frozen=True)
class Issue:
    level: Literal["error", "warning"]
    index: str
    stage: str
    message: str
    error: VistralError | None = None

    def __str__(self) -> str:
        return f"[{self.level}] stage {self.index} '{self.stage}': {self.message}"


@dataclass
class ValidationContext:
    auto_convert: bool = False
    allow_lossy: bool = False
    issues: list[Issue] = field(default_factory=list)
    insertions: list[str] = field(default_factory=list)
    index: str = ""
    step: str = ""

    def at(self, i: int, step: str) -> ValidationContext:
        prefix = f"{self.index}." if self.index else ""
        return ValidationContext(
            auto_convert=self.auto_convert,
            allow_lossy=self.allow_lossy,
            issues=self.issues,
            insertions=self.insertions,
            index=f"{prefix}{i}",
            step=step,
        )

    def at_branch(self, name: str) -> ValidationContext:
        index = f"{self.index}[{name}]" if self.index else name
        return ValidationContext(
            auto_convert=self.auto_convert,
            allow_lossy=self.allow_lossy,
            issues=self.issues,
            insertions=self.insertions,
            index=index,
            step=self.step,
        )

    def error(self, error: VistralError, message: str | None = None) -> None:
        text = message or (error.message if isinstance(error, SpecMismatchError) else str(error))
        self.issues.append(Issue("error", self.index, self.step, text, error))

    def warn(self, message: str) -> None:
        self.issues.append(Issue("warning", self.index, self.step, message))


@dataclass
class ValidationReport:
    input_spec: ImageSpec
    output_spec: ImageSpec
    issues: list[Issue]
    insertions: list[str]
    pipeline: Pipeline
    state: ValidationState

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> ValidationReport:
        errors = self.errors
        if errors:
            first = errors[0].error
            assert first is not None
            if len(errors) > 1:
                first.add_note(f"{len(errors) - 1} more validation error(s):")
                for issue in errors[1:]:
                    first.add_note(f"  {issue}")
            first.report = self  # type: ignore[attr-defined]
            raise first
        return self

    def __str__(self) -> str:
        lines = [
            f"validation: {len(self.errors)} error(s), {len(self.warnings)} warning(s)",
            f"  input:  {self.input_spec}",
            f"  output: {self.output_spec}",
        ]
        lines += [f"  {i}" for i in self.issues]
        lines += [f"  inserted {s}" for s in self.insertions]
        return "\n".join(lines)


def final_spec(spec: ImageSpec, path: list[Converter]) -> ImageSpec:
    for conv in path:
        spec = conv.result(spec)
    return spec


def convert_kwargs(spec: ImageSpec, path: list[Converter]) -> dict[str, Any]:
    end = final_spec(spec, path)
    return {
        f: getattr(end, f)
        for f in SPEC_FIELDS
        if f != "size" and getattr(end, f) != getattr(spec, f)
    }


def _fmt_kwargs(kwargs: dict[str, Any], lossy: bool = False) -> str:
    parts = [f"{k}={str(v)!r}" for k, v in kwargs.items()]
    if lossy:
        parts.append("allow_lossy=True")
    return ", ".join(parts)


def conversion_suggestion(
    spec: ImageSpec, accepts: Accepts, *, unknown: list[str] | tuple[str, ...] = ()
) -> str:
    """How to fix ``spec`` not satisfying ``accepts``: a conversion if one exists, else a hint."""
    if unknown:
        return (
            f"the upstream image leaves {', '.join(unknown)} unknown; declare it "
            "(e.g. read_image(..., color=...), a stage's produces=...) or insert vt.Convert"
        )
    path = plan(spec, accepts)
    if path:
        return (
            f"insert vt.Convert({_fmt_kwargs(convert_kwargs(spec, path))}) before this stage "
            f"[{describe_path(path)}], or validate(..., auto_convert=True)"
        )
    lossy = plan(spec, accepts, allow_lossy=True)
    if lossy:
        return (
            f"only a lossy conversion exists [{describe_path(lossy)}]; if that loss is acceptable, "
            f"insert vt.Convert({_fmt_kwargs(convert_kwargs(spec, lossy), lossy=True)})"
        )
    violations, _ = accepts.check(spec)
    hints = [v.hint for v in violations if v.hint]
    if hints:
        return "; ".join(hints)
    return (
        "no conversion produces an accepted image; change the upstream stages "
        f"so they output {accepts}"
    )


def infer_stage(
    stage: Stage, state: ValidationState, ctx: ValidationContext
) -> tuple[ValidationState, list[Stage]]:
    """Default static inference for one stage. Combinators override ``Stage.infer``."""
    from vistral.core.stage import Role

    state = state.copy()
    spec, replacement = _check_accepts(stage, state.spec, ctx)
    _check_artifacts(stage, state, ctx)
    reads = _check_results(stage, state, ctx)
    _check_source_coordinates(stage, state, ctx)

    state.spec = stage.produces(spec)
    if stage.changes_geometry:
        state.geometry = (*state.geometry, ctx.step)
    state.artifacts.update(stage.writes_artifacts())
    for key in stage.writes_results():
        if stage.role is Role.POSTPROCESS and reads and reads[0] in state.result_spaces:
            state.result_spaces[key] = state.result_spaces[reads[0]]
        else:
            state.result_spaces[key] = state.coord_space
    return state, replacement


def _check_accepts(
    stage: Stage, spec: ImageSpec, ctx: ValidationContext
) -> tuple[ImageSpec, list[Stage]]:
    from vistral.core.ops import Convert

    replacement: list[Stage] = [stage]
    accepts = stage.accepts
    if accepts is None:
        return spec, replacement

    violations, unknown = accepts.check(spec)
    if unknown:
        ctx.warn(
            f"cannot verify {', '.join(unknown)} (unknown upstream); it will be checked at runtime"
        )
    if not violations:
        return spec, replacement

    path = plan(spec, accepts, allow_lossy=ctx.allow_lossy)
    if ctx.auto_convert and path:
        kwargs = convert_kwargs(spec, path)
        replacement = [Convert(**kwargs, allow_lossy=ctx.allow_lossy), stage]
        ctx.insertions.append(
            f"Convert({_fmt_kwargs(kwargs)}) before stage {ctx.index} '{ctx.step}' "
            f"[{describe_path(path)}]"
        )
        return final_spec(spec, path), replacement

    ctx.error(
        SpecMismatchError(
            f"stage {ctx.index} '{ctx.step}' does not accept {spec}",
            stage=ctx.step,
            index=ctx.index,
            violations=tuple(violations),
            suggestion=conversion_suggestion(spec, accepts),
        )
    )
    return spec, replacement


def _check_artifacts(stage: Stage, state: ValidationState, ctx: ValidationContext) -> None:
    for key in stage.reads_artifacts():
        if key in state.artifacts:
            continue
        ctx.error(
            SpecMismatchError(
                f"stage {ctx.index} '{ctx.step}' reads artifact {key!r}, "
                "but no upstream stage writes it",
                stage=ctx.step,
                index=ctx.index,
                suggestion="add the stage that produces it (e.g. an Encoder before a Model)",
            )
        )


def _check_results(stage: Stage, state: ValidationState, ctx: ValidationContext) -> tuple[str, ...]:
    reads = stage.reads_results()
    for key in reads:
        if key not in state.result_spaces:
            ctx.warn(f"reads result {key!r}, which no upstream stage writes")
    return reads


def _check_source_coordinates(stage: Stage, state: ValidationState, ctx: ValidationContext) -> None:
    for key in stage.requires_source_coords:
        space = state.result_spaces.get(key)
        if space is None or space == "source":
            continue
        ctx.error(
            CoordinateSpaceError(
                f"stage {ctx.index} '{ctx.step}' needs {key!r} in source coordinates, "
                f"but it is in {space} (geometry changed after the source)",
                stage=ctx.step,
                index=ctx.index,
                suggestion="insert vt.ToSourceCoords() before this stage",
            )
        )
