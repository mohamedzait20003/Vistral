"""Architecture rules: the public import (D-023) is light (D-011) and is the only entry point."""

import subprocess
import sys

FORBIDDEN = ("cv2", "torch", "transformers", "supervision", "PIL", "pydantic")


def _loaded_after(statement: str) -> list[str]:
    code = f"import sys; {statement}; print(','.join(m for m in {FORBIDDEN!r} if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return [m for m in out.stdout.strip().split(",") if m]


def test_import_vistral_is_light():
    assert _loaded_after("import vistral") == []


def test_public_import_is_light():
    assert _loaded_after("from vistral.core import vt") == []


def test_running_a_numpy_pipeline_stays_light():
    statement = (
        "import numpy as np; from vistral.core import vt; "
        "(vt.Convert(color='RGB') | vt.Convert(layout='CHW'))(np.zeros((4, 4, 3), np.uint8))"
    )
    assert _loaded_after(statement) == []


def test_vt_is_the_only_public_namespace():
    import vistral
    from vistral.core import vt

    assert vistral.__all__ == ["__version__"]
    assert not hasattr(vistral, "Pipeline")  # `import vistral as vt` is intentionally gone
    assert vt.__version__ == vistral.__version__
    for name in (
        "Pipeline",
        "Convert",
        "ToSourceCoords",
        "filter",
        "postprocess",
        "stage",
        "read_image",
        "ImageSpec",
        "Accepts",
        "Branch",
        "When",
    ):
        assert name in vt.__all__ and hasattr(vt, name), name
    assert all(hasattr(vt, name) for name in vt.__all__)
