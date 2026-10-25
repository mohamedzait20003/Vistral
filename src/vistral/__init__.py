"""Vistral: validated, inspectable computer-vision pipelines.

M1 provides the core contracts: image specs, frames, stages, pipelines, validation, conversion
and tracing. OpenCV built-ins (M2) and Hugging Face integration (M3) follow.
"""

from vistral import core
from vistral.core import *  # noqa: F403
from vistral.core.stage import filter_stage as filter
from vistral.core.stage import frame_stage as stage
from vistral.core.stage import postprocess_stage as postprocess
from vistral.io import read_image

__version__ = "0.1.0.dev0"

__all__ = ["filter", "postprocess", "read_image", "stage", "__version__"]
__all__ += core.__all__
