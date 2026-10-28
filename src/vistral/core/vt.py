"""Vistral's public API namespace. Import it as::

    from vistral.core import vt

    pipe = vt.Pipeline([...])
    pipe.validate(vt.ImageSpec.bgr_u8())

This is the one supported import (D-023). Component namespaces (``vistral.cv``, ``vistral.hf``,
``vistral.qc``) are imported separately as they arrive. Like the rest of ``vistral.core``,
importing this module does not import OpenCV, torch, transformers or supervision.
"""

from vistral import __version__
from vistral import core as _core
from vistral.core import *  # noqa: F403
from vistral.core.stage import filter_stage as filter
from vistral.core.stage import frame_stage as stage
from vistral.core.stage import postprocess_stage as postprocess
from vistral.io.image import read_image

__all__ = ["__version__", "filter", "postprocess", "read_image", "stage"]
__all__ += _core.__all__
