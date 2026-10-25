"""Error hierarchy. Every error Vistral raises derives from :class:`VistralError`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class VistralError(Exception):
    """Base class for all Vistral errors."""


@dataclass(frozen=True)
class Violation:
    """One spec field that does not satisfy a constraint."""

    field: str
    actual: Any
    allowed: Any
    hint: str | None = None

    def __str__(self) -> str:
        text = f"{self.field}={_fmt(self.actual)} (allowed: {_fmt(self.allowed)})"
        return f"{text}; {self.hint}" if self.hint else text


def _fmt(value: Any) -> str:
    if isinstance(value, (set, frozenset)):
        return "{" + ", ".join(sorted(str(v) for v in value)) + "}"
    return "unknown" if value is None else str(value)


class SpecMismatchError(VistralError):
    """An image or result does not satisfy what a stage accepts."""

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        index: int | str | None = None,
        violations: tuple[Violation, ...] = (),
        suggestion: str | None = None,
    ) -> None:
        self.message = message
        self.stage = stage
        self.index = index
        self.violations = tuple(violations)
        self.suggestion = suggestion
        super().__init__(str(self))

    def __str__(self) -> str:
        lines = [self.message]
        lines += [f"  - {v}" for v in self.violations]
        if self.suggestion:
            lines.append(f"  Fix: {self.suggestion}")
        return "\n".join(lines)


class CoordinateSpaceError(SpecMismatchError):
    """A result is in a different coordinate space than a stage requires."""


class ConversionError(VistralError):
    """No conversion path exists, or a conversion failed."""


class StageError(VistralError):
    """Wraps an exception raised inside a stage with the stage and frame that failed."""

    def __init__(
        self,
        stage: str,
        index: int | str,
        cause: BaseException,
        *,
        source: str | None = None,
        input_spec: Any = None,
    ) -> None:
        self.stage = stage
        self.index = index
        self.cause = cause
        self.source = source
        self.input_spec = input_spec
        where = f" on source {source}" if source else ""
        spec = f"\n  input: {input_spec}" if input_spec is not None else ""
        super().__init__(
            f"stage {index} '{stage}' failed{where}: {type(cause).__name__}: {cause}{spec}"
        )


class ModelOutputError(VistralError):
    """A decoder received malformed or unexpected model outputs."""


class ConfigError(VistralError):
    """A stage configuration cannot be serialized or reconstructed."""

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        file: str | None = None,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        self.path = path
        self.file = file
        self.line = line
        self.column = column
        location = ""
        if file:
            location = f"{file}:{line}:{column} " if line is not None else f"{file} "
        key = f"{path}: " if path else ""
        super().__init__(f"{location}{key}{message}")


class MissingDependencyError(VistralError, ImportError):
    """An optional dependency needed by a component is not installed."""

    def __init__(self, package: str, extra: str | None = None) -> None:
        self.package = package
        self.extra = extra
        hint = f"pip install vistral[{extra}]" if extra else f"pip install {package}"
        super().__init__(f"'{package}' is required for this component. Install it with: {hint}")
