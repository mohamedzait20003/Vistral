# Architecture

**Status:** Proposed (planning phase, 2026-10-21). This describes the **composable component framework** (Option 2), chosen in D-001. Labels: [F] fact, [A] assumption, [R] recommendation, [Q] open question.

---

## 1. Options considered

| Criterion | 1. Thin integration layer | **2. Composable component framework (chosen)** | 3. Full workflow engine |
|---|---|---|---|
| What it is | Helpers plus a few adapters; users write their own glue | Shared `Stage` protocol, `Frame` envelope, explicit specs, sequential `Pipeline` with small combinators, YAML recipes | Adds DAG compiler, scheduler, concurrency, streaming, distributed execution |
| Developer experience | Small gain over raw cv2 + HF | One mental model, validation, tracing, recipes | Powerful, but steep to learn and debug |
| Implementation cost | Low | Medium | Very high |
| Runtime overhead | ~none | Small per stage, measured in M6 | Scheduler and serialization overhead |
| Maintainability | Easy | Moderate; small core | Hard |
| Extensibility | Ad hoc | Uniform (functions, classes, `cv.wrap`) | Uniform, but needs plugin infrastructure |
| Overlap with existing tools | Not enough value added beyond Supervision and HF | Builds on Supervision, HF and cv2; adds contracts and recipes | Duplicates Roboflow Workflows, DeepStream/Savant and G-API |

**Option 2** is the simplest that adds real value over calling the libraries directly. Option 1 doesn't solve the integration-bug problem. Option 3 rebuilds tools that already exist (see [RESEARCH_REVIEW.md §5](RESEARCH_REVIEW.md)).

---

## 2. Layering

```
                ┌───────────────────────────────────────────────┐
  user code ──▶ │ vistral (public API)                          │ ◀── recipe.yaml
                │   Pipeline · Branch · When · stage decorators │     (vistral.config)
                └──────────────┬────────────────────────────────┘
                               │
        ┌──────────────────────┼─────────────────────────────────────┐
        ▼                      ▼                      ▼              ▼
  vistral.core           vistral.cv             vistral.hf      vistral.io
  spec, image, frame,    filters, geometry,     Processor,      read_image, sources,
  stage, pipeline,       morphology, edges,     Model,          sinks, video
  results, convert,      analysis, draw,        PostProcess,    (cv2; torchcodec opt.)
  trace, errors,         qc (Verdict, Measure,  task composites
  testing                ReferenceDiff, Tiled)  (lazy import)
        │                      │                      │              │
        ▼                      ▼                      ▼              ▼
    numpy, pydantic      opencv-python-headless  torch, torchvision,  cv2 / torchcodec
                         supervision             transformers
```

**Dependency rules [R]** (enforced by an import-linter test from M1):
- `vistral.core` imports only numpy, pydantic and the standard library. It **never** imports cv2, torch, transformers or supervision at module import time.
  - Exception: `core.results` imports `supervision` only for type references, inside `TYPE_CHECKING`.
- `vistral.cv` may import cv2 and supervision.
- `vistral.hf` imports torch and transformers **lazily**, inside `setup()`. If the extra isn't installed it raises `MissingDependencyError("pip install vistral[hf]")`.
- `import vistral` must succeed with only the core dependencies installed (D-011).

---

## 3. Data model

### 3.1 `ImageSpec`
A frozen dataclass that describes pixel data. Every field can be `None`, meaning unknown.

| Field | Values | Notes |
|---|---|---|
| `layout` | `HWC`, `CHW`, `HW` (single channel, no channel axis) | Batches add a leading `N` |
| `color` | `BGR`, `RGB`, `GRAY`, `BGRA`, `RGBA`, `MASK` (binary/label map) | Color order can't be inferred from the data, so it has to be declared and tracked |
| `dtype` | `uint8`, `uint16`, `float32`, `float16`, `bool`, `int32` | |
| `range` | `U8` (0–255), `U16` (0–65535), `UNIT` (0–1), `NORMALIZED(mean, std)`, `ANY` | Catches double normalization and 0–1 vs 0–255 confusion |
| `device` | `cpu`, `cuda:N`, `mps` | NumPy data is always `cpu` |
| `size` | `(height, width)` | Always stored as **(H, W)**; never (W, H) anywhere in Vistral |

Named constructors cover common cases: `ImageSpec.bgr_u8()`, `rgb_u8()`, `gray_u8()`, `rgb_unit_chw()`.

### 3.2 `Image`
Holds `data` (an `np.ndarray` or `torch.Tensor`), its `spec`, and `transforms: tuple[GeoTransform, ...]`.
- PIL images are converted at the boundary (`Image.from_pil`) and are never carried internally.
- `Image` is immutable: stages return a new `Image`. The underlying array may be shared when a stage doesn't copy.

### 3.3 `GeoTransform` and coordinate tracking
- Each geometry stage appends a `GeoTransform(name, matrix: 3×3 float64, src_size, dst_size)`, which maps source pixels to destination pixels.
  - Crop, Resize, Pad, Letterbox, Flip and Rotate are affine.
  - Perspective warps are full homographies.
- **Coordinate convention [R] (D-009).** Continuous pixel coordinates: origin at the top-left corner of the top-left pixel, x to the right, y down. Boxes are **absolute xyxy float pixels**. Pixel *i* covers the interval [i, i+1).
  - Under this convention a resize by factor s maps box corners exactly, x' = s·x.
  - This differs from cv2's internal pixel-center sampling, but that sampling doesn't affect box geometry.
- `ToSourceCoords` composes the inverses of every transform after the frame's source image and applies them to the results:
  - `sv.Detections.xyxy`: corners are transformed and re-boxed; for rotations this gives the bounding box of the rotated box, and that loss is documented.
  - `mask`: warped with `cv2.warpPerspective`, using nearest-neighbor interpolation.
  - Keypoints and contours: points are transformed.
  - Afterwards the result's `coord_space` metadata is set to `"source"`.

### 3.4 `Frame` (D-005)
The unit that flows through a pipeline:

```text
Frame
  image:     Image                 # current working image
  original:  Image                 # reference to the source image (not copied)
  results:   dict[str, Result]     # e.g. "detections" -> sv.Detections, "verdict" -> Verdict
  artifacts: dict[str, Any]        # intermediates: "model_inputs", "raw_outputs", ...
  source:    SourceInfo            # uri, stem, frame_index, timestamp, camera_id
  errors:    list[StageFailure]    # filled when on_error="skip"/"collect"
```

- **Why an envelope.** Real pipelines need several values at once: the current image, the original for annotation and coordinate mapping, results from earlier stages, and model intermediates. Passing one bare value from stage to stage fails as soon as an annotator needs both an image and detections.
- **Copy semantics.** `Frame` is copied shallowly at each stage boundary: the dicts are copied, the arrays are not. A stage can't mutate an upstream stage's view of the frame, and no pixel data is copied unless needed.

### 3.5 Results
| Result | Type | Notes |
|---|---|---|
| Detections, instance masks | **`sv.Detections`** (D-004) | Vistral metadata goes in `metadata["vistral"]` (see below). Per-detection extras go in `data`: `class_name`, `zone`, `area_mm2`, … |
| Classification | `sv.Classifications` or `vistral.Classification` | [Q] decided in M1 after checking the sv API |
| Keypoints / pose | `sv.KeyPoints` or `vistral.Keypoints` | [Q] M1 |
| Semantic segmentation | `vistral.SemanticMask(label_map: HW int, id2label)` | |
| Embedding | `vistral.Embedding(vector, model_id, normalized: bool)` | |
| Text (captions, OCR) | `vistral.Text(text, tokens?, scores?)` | M7 |
| Contours | `vistral.Contours(points: list[ndarray], hierarchy)` | from `FindContours` |
| Measurements | `vistral.Measurements(items: list[Measurement(name, value, unit, tolerance, ok)])` | QC |
| Verdict | `vistral.Verdict(status: PASS/FAIL/REVIEW, reasons: list[str], rule_results)` | QC |
| Custom | any object; recommended to subclass `vistral.Result` | |

Every Vistral result carries `meta: ResultMeta`. On `sv.Detections` the same dict is stored under `metadata["vistral"]`. Keys:
- `producer` (stage name)
- `model_id`, `model_revision`
- `coord_space` (`"source"` or `"stage:<name>"`)
- `image_size` (H, W) of the coordinate space
- `created_ns`
- `recipe_hash` (when run from YAML)

---

## 4. Stages

### 4.1 Protocol
```python
class Stage(Protocol):
    name: str
    role: Role                                   # FILTER | ENCODER | MODEL | DECODER | POSTPROCESS | SINK
    accepts: SpecConstraint | None               # None = accepts anything (unchecked)
    def produces(self, spec: ImageSpec) -> ImageSpec   # default: identity
    def process(self, frame: Frame) -> Frame
    def process_batch(self, frames: list[Frame]) -> list[Frame]   # default: loop
    def setup(self) -> None                      # lazy resource acquisition (models)
    def close(self) -> None                      # idempotent
    def reset(self) -> None                      # stateful stages; called at stream boundaries
    def get_config(self) -> dict                 # serializable constructor params (for YAML)
```

`SpecConstraint` is a set of allowed values per `ImageSpec` field, for example `color ∈ {BGR, GRAY}, dtype ∈ {uint8}`.

### 4.2 Roles
The roles separate responsibilities that are often conflated:

| Role | Reads → writes | Examples | Not to be confused with |
|---|---|---|---|
| **Source** | (outside the pipeline) → `Frame`s | `ImageFolderSource`, `VideoSource` | Source is where **media decoding** (JPEG, H.264) happens |
| **Filter** | `frame.image` → `frame.image` | cv ops, geometry | Encoders: filters keep data in image space |
| **Encoder** | `frame.image` → `artifacts["model_inputs"]` | `hf.Processor` | Image preprocessing filters; encoders produce **model-specific tensors** |
| **Model** | `artifacts["model_inputs"]` → `artifacts["raw_outputs"]` | `hf.Model`, ONNX model | Inference only; no pre- or post-processing |
| **Decoder** | `artifacts["raw_outputs"]` → `results[key]` | `hf.PostProcess` (`post_process_*`), text decoding | **Model-output decoding**, distinct from media decoding |
| **Postprocess** | `results[key]` → `results[key]` | `FilterDetections`, `NMS`, `ToSourceCoords`, `Measure`, `Verdict` | Decoders: postprocessors work on structured results |
| **Sink** | `Frame` → side effect | `ImageWriter`, `JSONSink`, `CallbackSink` | Annotators, which are filters that draw on a copy |

The artifact and result keys above are defaults. Every stage takes `reads=` / `writes=` overrides, which lets two models run in one pipeline (`writes="defects"`, `writes="labels"`).

### 4.3 How authors write stages
1. **Function plus decorator.** For pure array operations:
   - `@vt.filter(accepts=..., produces=...)` wraps `fn(ndarray, **params) -> ndarray`.
   - `@vt.postprocess(reads="detections")` wraps `fn(result, **params) -> result`.
2. **`cv.wrap(cv2_fn, **params, accepts=..., produces=...)`** turns any cv2 call into a stage.
3. **Subclass a role base class** (`vt.Filter`, `vt.Model`, …) when the stage has lifecycle (models), state (trackers, background subtractors) or batching.

Decorated functions and subclasses are the same `Stage` at runtime. Decorator parameters become `get_config()`.

---

## 5. Pipeline execution

### 5.1 Pipeline
- `Pipeline(stages, name=, on_error="raise")`. `a | b` builds a pipeline. A `Pipeline` is itself a `Stage`, so pipelines nest as reusable sub-pipelines.
- Calling the pipeline:
  - `pipeline(x)` accepts an `ndarray`, `Image`, path or `Frame` and returns the output `Frame`.
  - `pipeline.run(x, trace=True)` returns `RunResult(frame, trace)`.
- Batches: `pipeline.map(iterable, batch_size=8)`.
  - Consecutive stages that implement `process_batch` receive lists.
  - Other stages are looped.
  - Order is preserved.
- Streams: `pipeline.stream(source, sinks=[...])` is a generator of Frames. `reset()` is called on stateful stages when the source changes (for example a new video file).
- Lifecycle: `with pipeline:` calls `setup()` on all stages, then `close()` on exit. `setup()` also runs lazily on first use.

### 5.2 Combinators
- `Branch({"a": p1, "b": p2}, merge="namespace")` runs sub-pipelines on the same input frame; the image is shared read-only. Results merge as `results["a.detections"]`, …
  - Branch runs sequentially in v1. Parallel execution is an M7 question.
- `When(predicate, then=p, otherwise=q)`. `predicate` is a callable `Frame -> bool`. In YAML it is a structured condition such as `{result: verdict, equals: FAIL}`, never an expression string (D-007).

### 5.3 Validation and spec propagation
`pipeline.validate(input_spec)` runs before any data is processed:

```
spec := input_spec
for stage in stages:
    if stage.accepts is None:            -> spec unknown downstream for unchecked fields (warning)
    elif spec violates stage.accepts:
        path := shortest conversion path in ConversionGraph(spec -> nearest accepted spec)
        strict mode    -> raise SpecMismatchError(stage, spec, accepts, suggestion=path)
        auto_convert   -> insert Convert stages for path; record in validation report
    spec := stage.produces(spec)
```

- **ConversionGraph** nodes are spec states. Edges are registered converters, each with a cost estimate:
  - BGR↔RGB: `cvtColor`
  - GRAY→BGR
  - uint8 U8 → float32 UNIT
  - HWC↔CHW: a view or transpose
  - numpy↔torch: `from_numpy` (zero-copy) or `.numpy()`
  - cpu→cuda
- **Runtime checks.** In debug mode (`VISTRAL_DEBUG=1` or `pipeline.run(check=True)`), each stage's actual output is checked against its declared spec: dtype, ndim, channel count, size and device. Values are cheap to check, but color order can't be detected at runtime. That is why it's tracked as metadata.
- **HF encoders** declare `accepts = RGB, uint8, HWC` and record `size` so the decoder can supply `target_sizes` (H, W).

### 5.4 Conversions and copies
- All conversions go through `vistral.core.convert`. It uses zero-copy where legal:
  - `torch.from_numpy`,
  - transposes returned as views,
  - `np.ascontiguousarray` **only** when cv2 needs contiguous memory.
- Each conversion is recorded in the trace with `copied: bool` and `nbytes`.
- Stages never mutate their input arrays in place (D-013). cv2 drawing functions mutate, so annotators copy first. The contract-test kit checks this.

### 5.5 Errors
```
VistralError
 ├─ SpecMismatchError        (validation; includes suggested fix)
 ├─ ConversionError
 ├─ StageError               (wraps any exception: stage name/index, input spec, frame source; __cause__ preserved)
 ├─ ModelOutputError         (decoder received malformed/unexpected outputs)
 ├─ ConfigError              (YAML: file, line, column, key path)
 └─ MissingDependencyError   (extra not installed; includes pip command)
```
`on_error` controls what happens when a stage fails:
- `"raise"` (default) stops the pipeline.
- `"skip"` attaches a `StageFailure` to `frame.errors`, skips the remaining stages for that frame and continues with the next frame.
- `"collect"` does the same and also returns all failures at the end of `map`/`stream`.

### 5.6 Trace (observability)
Each stage produces a `TraceRecord`:
- name, role
- wall time in ns (with `cuda.synchronize` when the stage reports GPU work and `trace(sync_cuda=True)`)
- input and output spec
- conversions (count, copied bytes)
- new allocations (output `nbytes` when the buffer isn't shared with the input)
- result summaries (for example the detection count)

Available as `trace.table()` (text), `trace.to_dict()` and `trace.to_json()`. With `trace=False` the cost is a single `perf_counter_ns` pair per stage. The overhead is benchmarked in M6.

---

## 6. Model execution

- **Placement.** Device is a constructor parameter (`device="cuda:0"`). Models load lazily in `setup()` and stay cached on the stage instance. `warmup(n=3)` runs dummy inputs.
- **Inference.** Runs under `torch.inference_mode()`. Batching goes through `process_batch` when the model supports it.
- **Threading.** Pipelines and stages are **not thread-safe**; the documented pattern is one pipeline instance per thread or process. cv2 releases the GIL inside most calls [A], which M7 can use for multi-camera threading.
- **No async in v1** (D-012). The protocol allows a future `aprocess`.
- **Remote inference.** A remote model is just a `Model` stage that makes an HTTP or gRPC call, with timeouts, retries and structured `StageError`s. The interface allows it; v1 ships no remote adapters.

---

## 7. Configuration and recipes (D-007, M5)

- **YAML serializes the Python API; it is not a separate DSL.** `Pipeline.from_yaml(path, allow_imports=())`, `pipeline.to_yaml()` and the `vistral run` CLI.
- **Lookup.**
  - Built-ins by short name, from a static dict populated when `vistral` modules are imported.
  - Custom stages by import path `"package.module:ClassOrFunction"`, **only** when they match `allow_imports`.
  - The default is built-ins only.
- **Validation.**
  - `yaml.safe_load`.
  - Each stage's parameters are validated with a pydantic v2 `TypeAdapter` built from the constructor's type-annotated signature. Errors report file, line and column.
  - Then `pipeline.validate(input_spec)` runs.
- **Provenance.**
  - `recipe_hash` = SHA-256 of the canonicalized YAML.
  - Each run's metadata records `recipe_hash`, model ids and revisions, and the versions of vistral, cv2, transformers and supervision.
- **Compatibility.** A top-level `vistral: 1` schema version, and documented migrations when it changes.
- **Constraint from M1 onward:** every stage's constructor parameters must be config-serializable (scalars, lists, dicts, enums, paths). Non-serializable objects, such as a loaded model, are passed by reference (model id or path).

---

## 8. Module layout

```
src/vistral/
  __init__.py          # public re-exports (Pipeline, Branch, When, ImageSpec, read_image, decorators, ...)
  core/   spec.py image.py frame.py geometry.py stage.py pipeline.py combinators.py
          results.py convert.py trace.py errors.py
  cv/     color.py geometry.py filters.py intensity.py morphology.py edges.py analysis.py
          draw.py wrap.py
  qc/     roi.py tiled.py measure.py verdict.py reference.py calibration.py
  hf/     processor.py model.py postprocess.py tasks.py pipeline.py
  io/     image.py sources.py sinks.py video.py
  config/ loader.py schema.py registry.py cli.py
  interop/ (reserved: anomalib, trackers, torchvision in M7)
  testing/ check_stage.py fixtures.py fault_corpus.py
```

---

## 9. Architectural decisions

The rationale for every choice above is in [DECISIONS.md](DECISIONS.md), and the risks are in [RISKS_AND_OPEN_QUESTIONS.md](RISKS_AND_OPEN_QUESTIONS.md).
