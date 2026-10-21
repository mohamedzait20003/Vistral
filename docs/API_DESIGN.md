# API Design

**Status:** Proposed (planning phase, 2026-10-21). The names and signatures here are the M1–M5 targets. The implementation milestones may refine them, and any change will be recorded in [DECISIONS.md](DECISIONS.md). Everything below is a design, not working code.

Conventions used throughout:
- `import vistral as vt`, `from vistral import cv, hf, qc`, `import supervision as sv`
- Sizes are always **(height, width)**. Boxes are absolute **xyxy** float pixels.

---

## 1. Public API surface

| Namespace | Contents |
|---|---|
| `vistral` | `Pipeline`, `Branch`, `When`, `Frame`, `Image`, `ImageSpec`, `Color`, `Layout`, `Range`, `read_image`, `filter`, `postprocess`, `stage`, `Stage`, `Filter`, `Encoder`, `Model`, `Decoder`, `Postprocess`, `Sink`, `FilterDetections`, `MapLabels`, `NMS`, `ToSourceCoords`, `InZone`, `Convert`, errors |
| `vistral.cv` | OpenCV-backed filters, geometry, analysis, annotators, `wrap` (see [BUILTIN_COMPONENTS.md](BUILTIN_COMPONENTS.md)) |
| `vistral.qc` | `ROI`, `Tiled`, `Measure`, `Calibration`, `Verdict`, rule types, `ReferenceDiff`, `TemplateMatch`, `DrawVerdict` |
| `vistral.hf` | `Processor`, `Model`, `PostProcess`, `Classifier`, `Detector`, `ZeroShotDetector`, `SemanticSegmenter`, `Segmenter`, `Embedder`, `Pipeline` |
| `vistral.io` | `ImageFolderSource`, `VideoSource`, `FrameSampler`, `ImageWriter`, `VideoWriter`, `JSONSink`, `CallbackSink` |
| `vistral.config` | `load`, `dump`, CLI entry point `vistral` |
| `vistral.testing` | `check_stage`, `fault_corpus`, synthetic fixtures |

The public API is whatever is listed in each namespace's `__all__`. Everything else is private.

---

## 2. Pipeline examples

### 2.1 Classic image processing
```python
edges = cv.ToGray() | cv.GaussianBlur(ksize=5) | cv.Canny(threshold1=50, threshold2=150)
frame = edges("part.png")              # path, ndarray (assumed BGR u8 unless spec given), or vt.Image
frame.image.data                       # np.ndarray, HW uint8
frame.image.spec                       # ImageSpec(layout=HW, color=GRAY, dtype=uint8, range=U8, size=(H, W))
```

### 2.2 Hugging Face detection, with each piece explicit
```python
detect = vt.Pipeline([
    hf.Processor("facebook/detr-resnet-50"),        # Encoder: BGR→RGB once, native AutoImageProcessor
    hf.Model("facebook/detr-resnet-50", task="object-detection", device="cuda"),
    hf.PostProcess(threshold=0.5),                  # Decoder: processor.post_process_object_detection
], name="detr")                                     #   target_sizes=(H, W) from the Encoder's recorded size
det: sv.Detections = detect("street.jpg").results["detections"]
```

Equivalent composite:
```python
detect = hf.Detector("facebook/detr-resnet-50", threshold=0.5, device="cuda", revision="main")
```

### 2.3 Mixed industrial QC pipeline (the canonical M4 pipeline)
```python
inspect = vt.Pipeline([
    qc.ROI(polygon=[(120, 80), (1800, 80), (1800, 1000), (120, 1000)]),   # records crop transform
    cv.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
    qc.Tiled(hf.Detector("org/defect-detr", revision="3f2a1c", threshold=0.4, device="cuda"),
             tile=(640, 640), overlap=0.2),                               # wraps sv.InferenceSlicer
    vt.FilterDetections(min_score=0.5, min_area_px=25),
    vt.ToSourceCoords(),                                                  # undo ROI crop
    vt.InZone(polygon=[(400, 200), (1500, 200), (1500, 900), (400, 900)], name="critical"),
    qc.Measure(calibration=qc.Calibration(px_per_mm=12.4)),               # adds data["area_mm2"], ...
    qc.Verdict(rules=[
        qc.MaxCount(label="scratch", max=0, zone="critical"),
        qc.MaxCount(label="dent", max=0),
        qc.MaxArea(label="stain", max_mm2=1.5),
    ], review_score_band=(0.4, 0.6)),
    cv.DrawDetections(on="original"), qc.DrawVerdict(),
], name="bracket-inspection")

inspect.validate(vt.ImageSpec.bgr_u8())              # fails fast on contract errors
with inspect:                                        # setup()/close() for models
    res = inspect.run("parts/0001.png", trace=True)
res.frame.results["verdict"].status                  # "PASS" | "FAIL" | "REVIEW"
print(res.trace.table())                             # per-stage ms, conversions, copies
```

### 2.4 Classic golden-reference inspection (no ML)
```python
ref = vt.read_image("golden/bracket.png")
presence = vt.Pipeline([
    qc.TemplateMatch(template="golden/fiducial.png", align=True),         # records alignment transform
    qc.ReferenceDiff(reference=ref, threshold=35),                        # absdiff + threshold → MASK
    cv.Open(kernel=(5, 5)),
    cv.FindContours(min_area_px=40),                                      # results["contours"]
    qc.Verdict(rules=[qc.MaxCount(result="contours", max=0)]),
])
```

### 2.5 Video, batching and streaming
```python
src = vt.io.VideoSource("line3.mp4")                      # cv2.VideoCapture; backend="torchcodec" optional
sampler = vt.io.FrameSampler(every_n=5)
sinks = [vt.io.VideoWriter("out.mp4", fps=6), vt.io.JSONSink("results/{source.stem}.jsonl")]
for frame in inspect.stream(sampler(src), sinks=sinks, on_error="skip"):
    if frame.errors: log.warning(frame.errors)

frames = inspect.map(vt.io.ImageFolderSource("parts/"), batch_size=8)   # batches where stages support it
```

### 2.6 Branching and conditional routing
```python
both = vt.Branch({
    "defects": hf.Detector("org/defect-detr"),
    "label":   hf.Classifier("org/label-quality-vit"),
})                                                   # results["defects.detections"], results["label.classification"]

escalate = vt.When(lambda f: f.results["verdict"].status == "REVIEW",
                   then=hf.Segmenter("org/defect-mask2former"), otherwise=None)
```

### 2.7 Getting at native objects
```python
det_stage = inspect["qc.Tiled"].detector             # stages addressable by name or index
det_stage.model                                      # the transformers PreTrainedModel
det_stage.processor                                  # the native image processor
frame.image.data                                     # raw ndarray / tensor
frame.results["detections"]                          # a real sv.Detections — use any supervision API
```

---

## 3. Custom components

### 3.1 A function filter
```python
@vt.filter(accepts=vt.ImageSpec(color={"BGR", "GRAY"}, dtype="uint8"))   # produces: same as input
def vignette(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    h, w = img.shape[:2]
    k = cv2.getGaussianKernel(w, w * strength) @ cv2.getGaussianKernel(h, h * strength).T
    mask = (k.T / k.max())
    return (img * (mask[..., None] if img.ndim == 3 else mask)).astype(np.uint8)

pipe = cv.ToGray() | vignette(strength=0.7)          # decorated fn → stage factory
```

### 3.2 Wrapping any OpenCV function in one line
```python
detail = cv.wrap(cv2.detailEnhance, sigma_s=10, sigma_r=0.15,
                 accepts=vt.ImageSpec.bgr_u8())      # positional image arg first, kwargs bound
```

### 3.3 A custom postprocessor on `sv.Detections`
```python
@vt.postprocess(reads="detections")
def drop_edge_hits(det: sv.Detections, *, frame: vt.Frame, margin_px: int = 4) -> sv.Detections:
    h, w = frame.image.spec.size
    x1, y1, x2, y2 = det.xyxy.T
    keep = (x1 > margin_px) & (y1 > margin_px) & (x2 < w - margin_px) & (y2 < h - margin_px)
    return det[keep]
```

### 3.4 A custom HF decoder for a model with non-standard outputs
```python
class MyKeypointDecoder(vt.Decoder):
    writes = "keypoints"
    def decode(self, raw_outputs, frame: vt.Frame) -> sv.KeyPoints:
        ...                                          # parse raw_outputs; use frame.image.spec.size
pipe = hf.Processor("org/custom") | hf.Model("org/custom", task=None) | MyKeypointDecoder()
```

### 3.5 A custom model backend (ONNX Runtime)
```python
class OnnxDetector(vt.Model):
    accepts = vt.ImageSpec.rgb_unit_chw()            # validator inserts BGR→RGB, u8→unit, HWC→CHW if auto_convert
    def __init__(self, path: str, providers: list[str] = ("CPUExecutionProvider",)):
        self.path, self.providers = path, list(providers)
    def setup(self):
        import onnxruntime as ort
        self.session = ort.InferenceSession(self.path, providers=self.providers)
    def predict(self, inputs):
        return self.session.run(None, {"images": inputs[None]})
```

### 3.6 A stateful stage
```python
class RunningBackground(vt.Filter):
    def setup(self): self.sub = cv2.createBackgroundSubtractorMOG2()
    def reset(self): self.setup()                    # called at stream boundaries
    def apply(self, img): return self.sub.apply(img)
    def produces(self, spec): return spec.replace(color="MASK", layout="HW")
```

### 3.7 Testing a custom stage
```python
from vistral.testing import check_stage
def test_vignette(): check_stage(vignette(strength=0.7), sample=vt.testing.fixtures.bgr_u8(480, 640))
```

`check_stage` verifies that:
- the declared specs match the actual output,
- the input is not mutated,
- results are deterministic,
- `get_config()` round-trips,
- `close()` is idempotent,
- `process_batch` equals the looped `process`.

---

## 4. Compatibility contracts

| Contract | Rule |
|---|---|
| **Sizes** | Always (H, W) in every Vistral API and metadata field. Conversions to PIL or other (W, H) APIs happen only inside adapters. |
| **Boxes** | `sv.Detections.xyxy`, absolute float pixels, corner-based continuous coordinates (D-009). `metadata["vistral"]["coord_space"]` says which image they refer to. |
| **Color** | Images from `cv2`/`read_image` are BGR. HF encoders accept RGB, so the validator flags a BGR input, or `hf.Processor` converts once when `convert_color=True` (the default) and records it in the trace. |
| **Range / normalization** | Only encoders normalize. Filters must not change `range` unless they declare it. A `NORMALIZED` spec entering an HF encoder is an error, because that would normalize twice. |
| **Mutation** | Stages never modify input arrays in place. Annotators draw on copies. |
| **Ownership** | Output arrays may share memory with inputs (views). Copy with `frame.image.copy()` before mutating them yourself. |
| **Threading** | One pipeline instance per thread or process. Stages may hold models and are not thread-safe. |
| **Determinism** | Built-in CPU stages are deterministic. GPU model stages are deterministic only if the user configures torch for it. |
| **Escape hatches** | Native objects are always reachable (§2.7). Vistral never hides `sv.Detections` behind a proxy type. |

---

## 5. Recipe (YAML) schema, M5

```yaml
vistral: 1                                   # schema version (required)
name: bracket-inspection
input: {color: BGR, dtype: uint8}            # validated with pipeline.validate()
allow_imports: [acme_vision.stages]          # optional; custom import paths allowed
stages:
  - {type: qc.ROI, polygon: [[120,80],[1800,80],[1800,1000],[120,1000]]}
  - {type: cv.CLAHE, clip_limit: 2.0}
  - type: qc.Tiled
    tile: [640, 640]
    overlap: 0.2
    detector: {type: hf.Detector, model: org/defect-detr, revision: 3f2a1c, threshold: 0.4, device: cuda}
  - {type: FilterDetections, min_score: 0.5, min_area_px: 25}
  - {type: ToSourceCoords}
  - {type: InZone, name: critical, polygon: [[400,200],[1500,200],[1500,900],[400,900]]}
  - {type: qc.Measure, calibration: {px_per_mm: 12.4}}
  - type: qc.Verdict
    rules:
      - {type: MaxCount, label: scratch, max: 0, zone: critical}
      - {type: MaxCount, label: dent, max: 0}
      - {type: MaxArea, label: stain, max_mm2: 1.5}
  - {type: acme_vision.stages:CustomCheck, tolerance: 0.2}
sinks:
  - {type: JSONSink, path: "results/{source.stem}.json"}
  - {type: ImageWriter, path: "rejects/{source.stem}.png", annotate: true, when: {result: verdict, equals: FAIL}}
```

- CLI: `vistral run recipe.yaml --source parts/ [--device cuda] [--trace out.json]`, and `vistral validate recipe.yaml`.
- Conditions such as `when:` are structured mappings. Expression strings are never evaluated.

---

## 6. Error messages (target quality)

```
SpecMismatchError: stage 3 'hf.Detector' (Encoder hf.Processor) expects color=RGB but receives color=BGR
  from stage 2 'cv.CLAHE'.
  Fix: insert vt.Convert(color="RGB") before stage 3, or call pipeline.validate(..., auto_convert=True),
       or keep hf.Processor(convert_color=True) (default) which converts once at the boundary.

ConfigError: recipe.yaml:14:7 stages[2].detector.threshold: input should be a valid number (got 'high')

StageError: stage 5 'ToSourceCoords' failed on source parts/0042.png (frame 0):
  detections are in coord_space 'stage:qc.Tiled' but no inverse transform is recorded for it.
```

---

## 7. Versioning and API stability

- **SemVer.** While at 0.x, minor versions may break the API, but only with a deprecation warning for at least one minor release when that's feasible. 1.0 freezes the §1 surface.
- **Signature snapshots.** A snapshot test of the public signatures fails CI on any unintended change.
- **Schema versioning.** The YAML schema (`vistral: N`) is versioned independently of the package, and migrations are documented.
- **Supported dependency ranges** are declared in `pyproject.toml` and tested in the CI matrix ([IMPLEMENTATION_PLAN.md §3](IMPLEMENTATION_PLAN.md)).
