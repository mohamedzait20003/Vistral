"""Core built-in stages: explicit conversion and mapping results back to source coordinates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vistral.core.config import register_builtin
from vistral.core.convert import DEFAULT_GRAPH
from vistral.core.errors import ConversionError
from vistral.core.frame import Frame
from vistral.core.geometry import to_source_matrix
from vistral.core.results import get_meta, set_meta, transform_result
from vistral.core.spec import Accepts, ImageSpec
from vistral.core.stage import Role, Stage


@register_builtin("Convert")
class Convert(Stage):
    """Convert the image to the given fields using the cheapest conversion path.

    Only lossless conversions are used unless ``allow_lossy=True`` (e.g. to drop alpha, reduce
    bit depth, or convert color to gray). Normalization is never done here; encoders own it.
    """

    role = Role.FILTER

    def __init__(
        self,
        *,
        color: Any = None,
        dtype: Any = None,
        range: Any = None,
        layout: Any = None,
        device: Any = None,
        backend: Any = None,
        allow_lossy: bool = False,
    ) -> None:
        self.target = Accepts(
            color=color, dtype=dtype, range=range, layout=layout, device=device, backend=backend
        )
        if not self.target.constrained():
            raise ValueError("Convert needs at least one target field")
        self.allow_lossy = allow_lossy

    def default_name(self) -> str:
        params = self.config_params().items()
        return f"Convert({', '.join(f'{k}={v}' for k, v in params if v not in (None, False))})"

    def _plan(self, spec: ImageSpec) -> list[Any] | None:
        return DEFAULT_GRAPH.plan(spec, self.target, allow_lossy=self.allow_lossy)

    def produces(self, spec: ImageSpec) -> ImageSpec:
        path = self._plan(spec)
        if path is None:
            return spec
        for conv in path:
            spec = conv.result(spec)
        return spec

    def infer(self, state: Any, ctx: Any) -> tuple[Any, list[Stage]]:
        from vistral.core.validation import infer_stage

        if self._plan(state.spec) is None:
            from vistral.core.errors import SpecMismatchError

            ctx.error(
                SpecMismatchError(
                    f"stage {ctx.index} '{ctx.step}' cannot convert {state.spec} to {self.target}",
                    stage=ctx.step,
                    index=ctx.index,
                    suggestion=None
                    if self.allow_lossy
                    else "pass allow_lossy=True if a lossy conversion is acceptable",
                )
            )
        return infer_stage(self, state, ctx)

    def process(self, frame: Frame) -> Frame:
        frame.image = DEFAULT_GRAPH.convert(frame.image, self.target, allow_lossy=self.allow_lossy)
        return frame


@register_builtin("ToSourceCoords")
class ToSourceCoords(Stage):
    """Map results back to source-image coordinates using their recorded transforms.

    Applies to every result whose metadata says it is not in source coordinates (or only to
    ``keys``). Boxes are re-boxed after rotation; masks are warped with nearest-neighbour.
    """

    role = Role.POSTPROCESS

    def __init__(self, keys: Sequence[str] | None = None) -> None:
        self.keys = tuple(keys) if keys is not None else None

    def process(self, frame: Frame) -> Frame:
        keys = self.keys if self.keys is not None else tuple(frame.results)
        for key in keys:
            if key not in frame.results:
                raise KeyError(f"result {key!r} not found; available: {sorted(frame.results)}")
            result = frame.results[key]
            meta = get_meta(result)
            if meta is None or meta.coord_space == "source" or not meta.transforms:
                continue
            matrix = to_source_matrix(meta.transforms)
            size = meta.transforms[0].src_size
            try:
                mapped = transform_result(result, matrix, size)
            except ConversionError:
                if self.keys is None:
                    continue  # result types without coordinates are left as they are
                raise
            frame.results[key] = set_meta(
                mapped, meta.replace(coord_space="source", image_size=size, transforms=())
            )
        return frame

    def infer(self, state: Any, ctx: Any) -> tuple[Any, list[Stage]]:
        from vistral.core.validation import infer_stage

        state, repl = infer_stage(self, state, ctx)
        for key in self.keys if self.keys is not None else tuple(state.result_spaces):
            if key in state.result_spaces:
                state.result_spaces[key] = "source"
            else:
                ctx.warn(f"result {key!r} is not produced upstream")
        return state, repl
