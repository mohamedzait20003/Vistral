"""M1 quickstart: contracts, validation with fixes, conversion, coordinate mapping, tracing.

Run:  python examples/m1_quickstart.py
"""

import numpy as np

from vistral.core import vt
from vistral.testing import stubs


# 1. A custom filter is a decorated function. `produces` declares what it changes.
@vt.filter(
    accepts=vt.Accepts(color="BGR", dtype="uint8"), produces={"color": "GRAY", "layout": "HW"}
)
def green_channel(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[..., 1])


image = vt.read_image(np.zeros((480, 640, 3), np.uint8))  # 3-channel arrays are BGR (OpenCV)
print("input:", image)

# 2. A model that needs RGB (stand-in for a Hugging Face model, which arrives in M3).
pipeline = vt.Pipeline([stubs.RGBModelStub()], name="classify")
try:
    pipeline.validate(image.spec)  # fails before any pixels are processed
except vt.SpecMismatchError as err:
    print("\nvalidation error:\n" + str(err))

# 3. Let Vistral insert the lossless conversion it suggested.
report = pipeline.validate(image.spec, auto_convert=True)
print("\ninserted:", report.insertions)
result = report.pipeline.run(image, trace=True)
print(result.trace.table())

# 4. Coordinates: detections made on a cropped + resized image, mapped back to source pixels.
inspect = vt.Pipeline(
    [
        stubs.CropStub(box=(100, 50, 420, 370)),  # ROI 320x320
        stubs.ResizeStub(size=(160, 160)),  # model input size (H, W)
        stubs.BoxDetectorStub([[10, 10, 30, 20]], labels=["scratch"]),
        vt.ToSourceCoords(),
        stubs.MeasureStub(),  # requires source coordinates; validate() checks it
    ]
)
inspect.validate(image.spec)
frame = inspect(image)
det = frame.results["detections"]
print("\nbox in source pixels:", det.xyxy[0].tolist(), det.data["class_name"][0])
print("coord space:", vt.get_meta(det).coord_space)

# 5. Every stage serializes to a config that rebuilds it (foundation for YAML recipes, M5).
print("\nconfig:", green_channel().get_config())
