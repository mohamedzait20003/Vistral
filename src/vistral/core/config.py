"""Config serialization: every stage reports constructor parameters that rebuild it.

Built-ins are named by short name; anything else by import path ``"package.module:QualName"``.
Import paths are only resolved when they match ``allow_imports`` (fnmatch patterns on the
module name; ``"*"`` allows all). This is the M1 foundation for YAML recipes (M5, D-007).
"""

from __future__ import annotations

import fnmatch
import importlib
import inspect
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from pathlib import PurePath
from typing import Any, TypeVar

import numpy as np

from vistral.core.errors import ConfigError

T = TypeVar("T")

_BUILTINS: dict[str, Any] = {}
_BUILTIN_NAMES: dict[int, str] = {}

CALLABLE_KEY = "$callable"


def register_builtin(name: str) -> Callable[[T], T]:
    """Register a class or factory under a short config name."""

    def deco(obj: T) -> T:
        if name in _BUILTINS and _BUILTINS[name] is not obj:
            raise ValueError(f"built-in name {name!r} is already registered")
        _BUILTINS[name] = obj
        _BUILTIN_NAMES[id(obj)] = name
        return obj

    return deco


def builtin_names() -> list[str]:
    return sorted(_BUILTINS)


def import_path(obj: Any) -> str:
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not module or not qualname:
        raise ConfigError(f"{obj!r} has no importable name")
    if "<locals>" in qualname or "<lambda>" in qualname:
        raise ConfigError(
            f"{qualname} is defined inside a function or is a lambda and cannot be referenced "
            "from a config; define it at module level"
        )
    return f"{module}:{qualname}"


def type_name(obj: Any) -> str:
    """Config name of a class or factory: its built-in name, else its import path."""
    return _BUILTIN_NAMES.get(id(obj)) or import_path(obj)


def _allowed(module: str, allow_imports: Iterable[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(module, pattern) or module == pattern.removesuffix(".*")
        for pattern in allow_imports
    )


def resolve(name: str, allow_imports: Iterable[str] = ()) -> Any:
    """Resolve a built-in name or an allowed import path."""
    if name in _BUILTINS:
        return _BUILTINS[name]
    if ":" not in name:
        raise ConfigError(
            f"unknown component {name!r}; built-ins: {', '.join(builtin_names())}. "
            "Use 'package.module:Name' for custom components"
        )
    module_name, _, qualname = name.partition(":")
    if not _allowed(module_name, allow_imports):
        raise ConfigError(
            f"import of {name!r} is not allowed; "
            f"add {module_name!r} (or a pattern) to allow_imports"
        )
    try:
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"cannot import {name!r}: {exc}") from exc
    return obj


def to_config_value(value: Any, path: str = "") -> Any:
    """Convert a constructor argument to a JSON-compatible value."""
    from vistral.core.stage import Stage

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, Stage):
        return value.get_config()
    if hasattr(value, "to_config") and not isinstance(value, type):
        return value.to_config()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return _to_config_mapping(value, path)
    if isinstance(value, (list, tuple, set, frozenset)):
        return _to_config_sequence(value, path)
    if callable(value):
        return _to_config_callable(value, path)
    raise ConfigError(f"value of type {type(value).__name__} is not config-serializable", path=path)


def _to_config_mapping(value: Mapping[Any, Any], path: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError("mapping keys must be strings", path=path)
        child_path = f"{path}.{key}" if path else key
        out[key] = to_config_value(item, child_path)
    return out


def _to_config_sequence(
    value: list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any], path: str
) -> list[Any]:
    items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
    return [to_config_value(item, f"{path}[{index}]") for index, item in enumerate(items)]


def _to_config_callable(value: Any, path: str) -> dict[str, str]:
    try:
        return {CALLABLE_KEY: import_path(value)}
    except ConfigError as exc:
        raise ConfigError(str(exc), path=path) from exc


def from_config_value(value: Any, allow_imports: Iterable[str] = ()) -> Any:
    """Inverse of :func:`to_config_value` for stages and callables; other values pass through."""
    allow_imports = tuple(allow_imports)
    if isinstance(value, list):
        return [from_config_value(v, allow_imports) for v in value]
    if isinstance(value, Mapping):
        if set(value) == {CALLABLE_KEY}:
            return resolve(value[CALLABLE_KEY], allow_imports)
        if "type" in value and isinstance(value["type"], str) and _is_stage_type(value["type"]):
            return stage_from_config(value, allow_imports)
        return {k: from_config_value(v, allow_imports) for k, v in value.items()}
    return value


def _is_stage_type(name: str) -> bool:
    from vistral.core.stage import FunctionStageFactory, Stage

    obj = _BUILTINS.get(name)
    if obj is None:
        return ":" in name
    return isinstance(obj, FunctionStageFactory) or (
        isinstance(obj, type) and issubclass(obj, Stage)
    )


OPTION_KEYS = ("stage_name", "reads", "writes")


def stage_from_config(config: Mapping[str, Any], allow_imports: Iterable[str] = ()) -> Any:
    """Rebuild a stage from :meth:`Stage.get_config` output."""
    from vistral.core.stage import FunctionStageFactory, Stage

    allow_imports = tuple(allow_imports)
    params = dict(config)
    try:
        name = params.pop("type")
    except KeyError:
        raise ConfigError("stage config needs a 'type' key") from None
    factory = resolve(name, allow_imports)
    accepted = _accepted_params(factory)
    options = {k: params.pop(k) for k in OPTION_KEYS if k in params and k not in accepted}
    kwargs = {k: from_config_value(v, allow_imports) for k, v in params.items()}
    try:
        stage = factory(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"cannot build {name!r}: {exc}") from exc
    if not isinstance(stage, Stage):
        if isinstance(factory, FunctionStageFactory):
            raise ConfigError(f"{name!r} did not produce a stage")
        raise ConfigError(f"{name!r} is not a stage")
    if options:
        stage = stage.with_options(
            name=options.get("stage_name"), reads=options.get("reads"), writes=options.get("writes")
        )
    return stage


def _accepted_params(factory: Any) -> set[str]:
    from vistral.core.stage import FunctionStageFactory

    target = factory.fn if isinstance(factory, FunctionStageFactory) else factory
    try:
        return set(inspect.signature(target).parameters)
    except (TypeError, ValueError):
        return set()
