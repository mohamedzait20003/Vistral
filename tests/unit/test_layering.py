"""Architecture rule (D-011): importing vistral must not import heavy optional dependencies."""

import subprocess
import sys

FORBIDDEN = ("cv2", "torch", "transformers", "supervision", "PIL", "pydantic")


def _loaded_after(statement: str) -> list[str]:
    code = f"import sys; {statement}; print(','.join(m for m in {FORBIDDEN!r} if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return [m for m in out.stdout.strip().split(",") if m]


def test_import_vistral_is_light():
    assert _loaded_after("import vistral") == []


def test_import_core_is_light():
    assert _loaded_after("import vistral.core") == []


def test_running_a_numpy_pipeline_stays_light():
    statement = (
        "import numpy as np, vistral as vt; "
        "(vt.Convert(color='RGB') | vt.Convert(layout='CHW'))(np.zeros((4, 4, 3), np.uint8))"
    )
    assert _loaded_after(statement) == []
