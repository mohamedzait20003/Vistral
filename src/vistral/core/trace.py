"""Per-stage tracing: timing, specs, conversions and allocations.

Tracing is active only inside :meth:`Pipeline.run` with ``trace=True``; when inactive, recording
functions are a single context-variable lookup.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversionRecord:
    name: str
    from_spec: str
    to_spec: str
    copied: bool
    nbytes: int
    lossy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TraceRecord:
    index: int | str
    name: str
    role: str
    wall_ns: int = 0
    in_spec: str | None = None
    out_spec: str | None = None
    conversions: list[ConversionRecord] = field(default_factory=list)
    allocated_bytes: int = 0
    results: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    children: list[TraceRecord] = field(default_factory=list)

    @property
    def ms(self) -> float:
        return self.wall_ns / 1e6

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "role": self.role,
            "wall_ns": self.wall_ns,
            "in_spec": self.in_spec,
            "out_spec": self.out_spec,
            "conversions": [c.to_dict() for c in self.conversions],
            "allocated_bytes": self.allocated_bytes,
            "results": dict(self.results),
            "error": self.error,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class Trace:
    records: list[TraceRecord] = field(default_factory=list)
    total_ns: int = 0

    def __iter__(self) -> Iterator[TraceRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key: int | str) -> TraceRecord:
        if isinstance(key, int):
            return self.records[key]
        for rec in self.walk():
            if rec.name == key:
                return rec
        raise KeyError(key)

    def walk(self) -> Iterator[TraceRecord]:
        stack = list(reversed(self.records))
        while stack:
            rec = stack.pop()
            yield rec
            stack.extend(reversed(rec.children))

    @property
    def conversions(self) -> list[ConversionRecord]:
        return [c for rec in self.walk() for c in rec.conversions]

    @property
    def copies(self) -> int:
        return sum(1 for c in self.conversions if c.copied)

    def to_dict(self) -> dict[str, Any]:
        return {"total_ns": self.total_ns, "records": [r.to_dict() for r in self.records]}

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def table(self) -> str:
        header = (
            f"{'#':>5}  {'stage':<28} {'role':<11} {'ms':>9}  {'conv':>4}  {'alloc':>9}  output"
        )
        lines = [header, "-" * len(header)]

        def emit(rec: TraceRecord, depth: int) -> None:
            name = ("  " * depth + rec.name)[:28]
            alloc = _fmt_bytes(rec.allocated_bytes)
            out = (rec.error and f"ERROR {rec.error}") or rec.out_spec or ""
            if rec.results:
                out += "  " + ", ".join(f"{k}={v}" for k, v in rec.results.items())
            lines.append(
                f"{rec.index!s:>5}  {name:<28} {rec.role:<11} {rec.ms:>9.3f}  "
                f"{len(rec.conversions):>4}  {alloc:>9}  {out}"
            )
            for child in rec.children:
                emit(child, depth + 1)

        for rec in self.records:
            emit(rec, 0)
        lines.append(
            f"total {self.total_ns / 1e6:.3f} ms, conversions {len(self.conversions)}, "
            f"copies {self.copies}"
        )
        return "\n".join(lines)


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return str(n)


# Records list of the pipeline level currently executing (None = tracing off).
ACTIVE_RECORDS: ContextVar[list[TraceRecord] | None] = ContextVar("vistral_trace", default=None)
# Conversion sink of the stage currently executing (None = not recording).
ACTIVE_CONVERSIONS: ContextVar[list[ConversionRecord] | None] = ContextVar(
    "vistral_conversions", default=None
)


def record_conversion(record: ConversionRecord) -> None:
    sink = ACTIVE_CONVERSIONS.get()
    if sink is not None:
        sink.append(record)
