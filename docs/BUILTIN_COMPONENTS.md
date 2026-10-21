# Built-in Components

**Status:** Proposed catalog (planning phase, 2026-10-21).

**Principles**
1. Every built-in is a thin layer over a **native call** (cv2, HF, Supervision).
2. Wrapper output must equal the direct call's output, and a test checks it.
3. Keep the catalog small and add to it based on demand. Anything not listed is one line away via `cv.wrap` or `@vt.filter`.

**How built-ins are defined [R].** Each built-in is a small explicit class with a typed `__init__`, declared with a compact dataclass-style helper. The typed signature is what gives:
- IDE discoverability,
- pydantic validation for YAML (D-007),
- `get_config()`.

Runtime-generated classes are avoided because they break all three. Simple operations have no boilerplate at the call site: `cv.GaussianBlur(ksize=5)`.

**Column key**
- **Accepts → Produces:** the spec constraint. "same" means the output spec equals the input spec.
- **M:** the milestone that delivers it.

---

## 1. OpenCV: color (`vistral.cv`)

| Stage | Native call | Accepts → Produces | M |
|---|---|---|---|
| `ToGray()` | `cvtColor(BGR2GRAY / RGB2GRAY / BGRA2GRAY…)` (chosen from the input spec) | BGR/RGB/BGRA/RGBA u8/u16/f32 → GRAY, HW | M2 |
| `ToRGB()` / `ToBGR()` | `cvtColor` (or no-op if already in the target order) | color image → RGB / BGR | M2 |
| `ConvertColor(code)` | `cvtColor(code)` | user-declared | M2 |

## 2. OpenCV: geometry (each records a `GeoTransform`)

| Stage | Native call | Notes | M |
|---|---|---|---|
| `Resize(size=(H,W) \| scale=f \| max_side=n, interpolation="area"\|"linear"\|…)` | `cv2.resize` | Takes (H, W) and converts to cv2's (W, H) internally. Default interpolation is `area` when shrinking and `linear` when enlarging. | M2 |
| `Letterbox(size=(H,W), pad_value=114)` | `resize` + `copyMakeBorder` | Common for detector inputs | M2 |
| `Crop(box=(x1,y1,x2,y2))`, `CenterCrop(size)` | slicing (a view, no copy) | | M2 |
| `Pad(top, bottom, left, right, mode="constant")` | `copyMakeBorder` | | M2 |
| `Rotate(angle, expand=True)` | `rotate` for multiples of 90°, otherwise `warpAffine` | Rotating boxes re-boxes them | M2 |
| `Flip(axis="h"\|"v")` | `flip` | | M2 |
| `Warp(matrix)` | `warpAffine` / `warpPerspective` | Used for alignment results | M4 |

## 3. OpenCV: filtering and intensity

| Stage | Native call | M |
|---|---|---|
| `GaussianBlur(ksize=5, sigma=0)` | `GaussianBlur` (an int `ksize` expands to (k, k)) | M2 |
| `MedianBlur(ksize=5)` | `medianBlur` | M2 |
| `BilateralFilter(d=9, sigma_color=75, sigma_space=75)` | `bilateralFilter` | M2 |
| `Sharpen(amount=1.0, radius=1.0)` | Unsharp mask: `GaussianBlur` + `addWeighted` | M2 |
| `Denoise(h=10, template_window=7, search_window=21)` | `fastNlMeansDenoising` / `…Colored`, chosen from the spec | M2 |
| `NormalizeMinMax(lo=0, hi=255)` | `normalize(NORM_MINMAX)` | M2 |
| `BrightnessContrast(alpha=1.0, beta=0)` | `convertScaleAbs` | M2 |
| `EqualizeHist()` | `equalizeHist` (GRAY; for color, applied to the Y channel of YCrCb) | M2 |
| `CLAHE(clip_limit=2.0, tile_grid_size=(8,8))` | `createCLAHE` (color: L channel of LAB) | M2 |
| `Gamma(gamma=1.0)` | `LUT` | M2 |

## 4. OpenCV: threshold, morphology, edges

| Stage | Native call | Accepts → Produces | M |
|---|---|---|---|
| `Threshold(thresh=127, maxval=255, type="binary", otsu=False)` | `threshold` | GRAY → MASK | M2 |
| `AdaptiveThreshold(block_size=11, C=2, method="gaussian")` | `adaptiveThreshold` | GRAY u8 → MASK | M2 |
| `Morphology(op, kernel=(3,3), shape="rect", iterations=1)` plus aliases `Erode`, `Dilate`, `Open`, `Close` | `morphologyEx` + `getStructuringElement` | same | M2 |
| `Canny(threshold1, threshold2, aperture_size=3, l2_gradient=False)` | `Canny` | GRAY/BGR u8 → MASK | M2 |
| `Sobel(dx, dy, ksize=3)`, `Laplacian(ksize=3)` | `Sobel` / `Laplacian` (output depth CV_32F; optional `abs_u8=True`) | → f32 or u8 | M2 |
| `ApplyMask(mask_from="results:<key>" \| ndarray)` | `bitwise_and` | same | M2 |

## 5. OpenCV: analysis (produces Results)

| Stage | Native call | Writes | M |
|---|---|---|---|
| `FindContours(mode="external", min_area_px=0)` | `findContours`, `contourArea` | `results["contours"]` (`Contours`) | M2 |
| `ContoursToDetections()` | `boundingRect` | `results["detections"]` (`sv.Detections`, with contour index in `data`) | M2 |
| `ConnectedComponents(min_area_px=0)` | `connectedComponentsWithStats` | `sv.Detections` | M4 |

## 6. Annotation

These reuse Supervision annotators (D-015). Each draws on a copy, using `on="current"|"original"`.

| Stage | Backed by | M |
|---|---|---|
| `cv.DrawDetections(labels=True, thickness=2)` | `sv.BoxAnnotator` + `sv.LabelAnnotator` | M2 |
| `cv.DrawMasks(opacity=0.5)` | `sv.MaskAnnotator` | M2 |
| `cv.DrawKeypoints()` | sv keypoint annotators ([Q] API verified in M1) | M3 |
| `cv.DrawContours()` | `cv2.drawContours` | M2 |
| `qc.DrawMeasurements()`, `qc.DrawVerdict()` | `cv2.putText` / `rectangle` (a PASS/FAIL banner) | M4 |

## 7. Detection postprocessing (top-level `vistral`; operates on `sv.Detections`)

| Stage | Backed by | M |
|---|---|---|
| `FilterDetections(min_score, labels, exclude_labels, min_area_px, max_area_px, aspect_range)` | boolean indexing `det[mask]` | M2 |
| `MapLabels(mapping)` | rewrites `data["class_name"]` / `class_id` | M2 |
| `NMS(iou=0.5, class_agnostic=False)` | `Detections.with_nms` | M2 |
| `ToSourceCoords()` | Vistral `GeoTransform` inverses ([ARCHITECTURE.md §3.3](ARCHITECTURE.md)) | M2 |
| `InZone(polygon, name)` | `sv.PolygonZone` ([Q] trigger API verified in M4); tags `data["zone"]` | M4 |
| `Convert(color=, dtype=, range=, layout=, device=)` | `vistral.core.convert` | M1 |

## 8. Industrial / QC (`vistral.qc`), prioritized per D-014

| Stage | Purpose | Backed by | M |
|---|---|---|---|
| `ROI(polygon \| box)` | Restrict inspection to a part region. Crops to the bounding box and masks the outside. | slicing + `fillPoly` | M4 |
| `Tiled(detector, tile=(H,W), overlap=0.2, nms_iou=0.5)` | Small defects on high-resolution images | `sv.InferenceSlicer` ([Q] callback API verified in M4) | M4 |
| `Calibration(px_per_mm=… \| homography=…)` | Pixel → physical units (planar part, fixed camera) | `perspectiveTransform` | M4 |
| `Measure(what=("area","length","width"))` | Dimensions per detection or contour, with optional tolerances | `minAreaRect`, `contourArea` | M4 |
| `Verdict(rules, review_score_band=None)` | PASS/FAIL/REVIEW with reasons | rules: `MaxCount`, `MaxArea`, `MinCount`, `Tolerance(measurement, lo, hi)`, `ScoreAbove` | M4 |
| `ReferenceDiff(reference, threshold, blur=3)` | Golden-image comparison (presence/absence, contamination) | `absdiff` + `threshold` | M4 |
| `TemplateMatch(template, method="ccoeff_normed", align=False, min_score=0.8)` | Fiducial and alignment checks; optionally records an alignment transform | `matchTemplate`, `minMaxLoc` | M4 |

## 9. Hugging Face (`vistral.hf`, extra `[hf]`)

| Stage | Native pieces | Result | M |
|---|---|---|---|
| `Processor(model_id, revision=None, backend=None, device=None, convert_color=True, **overrides)` | `AutoImageProcessor.from_pretrained` | `artifacts["model_inputs"]` | M3 |
| `Model(model_id, task, revision=None, device="cpu", dtype=None)` | `AutoModelFor<Task>.from_pretrained`, `inference_mode` | `artifacts["raw_outputs"]` | M3 |
| `PostProcess(task=None, threshold=…, **kwargs)` | the processor's `post_process_<task>`, with `target_sizes` = (H, W) of the encoded image | task result | M3 |
| `Classifier(model_id, top_k=5)` | `AutoModelForImageClassification` + softmax + `id2label` | Classification | M3 |
| `Detector(model_id, threshold=0.5)` | `AutoModelForObjectDetection` + `post_process_object_detection` → `sv.Detections.from_transformers(results, id2label)` | `sv.Detections` | M3 |
| `ZeroShotDetector(model_id, labels=[...])` | `AutoModelForZeroShotObjectDetection` + its processor's post-processing ([Q] exact method per model) | `sv.Detections` | M3 |
| `SemanticSegmenter(model_id)` | `AutoModelForSemanticSegmentation` + `post_process_semantic_segmentation` | `SemanticMask` | M3 |
| `Segmenter(model_id, mode="instance"\|"panoptic")` | `post_process_instance_segmentation` / `_panoptic_segmentation` | `sv.Detections` with masks | M3 |
| `Embedder(model_id, normalize=True)` | `AutoModel` image features (for example CLIP/SigLIP `get_image_features`, DINOv2 pooled output; [Q] per-family table) | `Embedding` | M3 |
| `Pipeline(task, model_id, **kwargs)` | `transformers.pipeline` (fallback for untyped tasks) | raw pipeline output in `results[task]` | M3 |

**Deferred (M7, each needs a demonstrated need):** pose (VitPose), VLM captioning and VQA (`Text`), timm backbones.

## 10. IO (`vistral.io`)

| Component | Backed by | M |
|---|---|---|
| `read_image(path, spec=None)` | `cv2.imread` (`IMREAD_UNCHANGED` for 16-bit industrial images, then the spec is inferred) | M1 (numpy) / M2 (cv2) |
| `ImageFolderSource(dir, pattern="*.png", sort=True)` | `pathlib` + `read_image` | M2 |
| `VideoSource(path \| index, backend="cv2"\|"torchcodec")` | `cv2.VideoCapture` / TorchCodec `VideoDecoder` | M2 (cv2), M7 (torchcodec) |
| `FrameSampler(every_n \| fps)` | — | M2 |
| `ImageWriter(path_template, annotate=False, when=None)` | `cv2.imwrite` | M2 |
| `VideoWriter(path, fps, fourcc="mp4v")` | `cv2.VideoWriter` | M2 |
| `JSONSink(path_template)` / `CallbackSink(fn)` | stdlib json (results serialized via `to_dict`) | M2 |

## 11. Deferred catalog (M7, optional)

- **Motion and tracking:** `BackgroundSubtractor(MOG2/KNN)`, `OpticalFlow(Farneback)`, `Tracker(algorithm="bytetrack")` via Roboflow `trackers`.
- **Anomaly detection:** `qc.AnomalyDetector` via Anomalib, producing an `AnomalyScore` and heatmap.
- **Calibration:** full camera calibration and undistortion (`calibrateCamera`, `undistort`).
- **Model backends:** ONNX Runtime and OpenVINO `Model` stages.
- **Industrial IO:** GenICam camera source; MQTT and OPC UA sinks.
- **GPU filter chains:** Kornia or OpenCV G-API fusion.
