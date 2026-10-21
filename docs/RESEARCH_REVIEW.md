# Research Review

**Status:** Planning phase. **Research date:** 2026-10-21.
**Labels:**
- **[F]** verified fact, with its source linked
- **[A]** assumption, not yet verified
- **[R]** recommendation
- **[Q]** open question (tracked in [RISKS_AND_OPEN_QUESTIONS.md](RISKS_AND_OPEN_QUESTIONS.md))

This review records what the existing ecosystem already provides, so Vistral reuses it instead of rebuilding it. It also identifies the gaps that would justify Vistral existing. Unless an item says it was verified against a specific version, versions are the ones current on the research date.

---

## 1. Repository state

- **[F]** When work started, the repository was empty: no source, docs, dependencies or git history. Nothing needed to be reused, refactored or preserved.
- **[F]** The development machine has Python 3.14.5 and no CV/ML packages installed. Nothing was installed during planning.

---

## 2. Hugging Face ecosystem

### 2.1 `transformers` (v5.14 current)

**Image processors**
- **[F]** Image processors load with `AutoImageProcessor.from_pretrained(model_id)`. Their configuration comes from the checkpoint's `preprocessor_config.json` and covers size, rescale, normalization mean/std and crop. ([image processors guide](https://huggingface.co/docs/transformers/v5.14.0/image_processors.md))
- **[F]** In v5, image processors use a backend architecture:
  - `TorchvisionBackend` is the default. It is GPU-capable and documented as "up to 33x faster" than PIL on batches of `torch.Tensor`.
  - `PilBackend` is CPU-only and exists only for older models, for exact numerical parity with the original implementation.
  - Select one with `from_pretrained(..., backend="torchvision" | "pil")`.
  - Processing can run on GPU via `processor(images, return_tensors="pt", device="cuda")`. (same source)
- **[F]** In v4.48, fast (torchvision) processors were opt-in via `use_fast=True` ([v4.48.1 docs](https://huggingface.co/docs/transformers/v4.48.1/en/main_classes/image_processor)). v5 made them the default. The upgrade from v4 to v5 changed processor behavior, which shows HF APIs do change.

**Post-processing**
- **[F]** Model-specific post-processing lives **on the image processor**:
  - `post_process_object_detection(outputs, threshold, target_sizes)`
  - `post_process_semantic_segmentation`, `post_process_instance_segmentation`, `post_process_panoptic_segmentation`

  Detection returns `list[dict]` with keys `scores`, `labels` and `boxes`. Boxes are absolute **(top_left_x, top_left_y, bottom_right_x, bottom_right_y)**. `target_sizes` is a list of **(height, width)**. ([DETR docs v5.14](https://huggingface.co/docs/transformers/v5.14.0/model_doc/detr))
- **[F]** The official example passes `target_sizes=torch.tensor([image.size[::-1]])`. PIL's `.size` is (width, height) and has to be reversed, which makes this a common source of silent bugs. (same source)
- **[F]** When adding a model, post-processors return either a list of simple objects or a list of `BatchFeature`-derived output objects, defined in `image_processing_outputs.py`. Post-processing method names and return types therefore **vary by model and task**. ([Add vision processing components](https://huggingface.co/docs/transformers/v5.14.0/add_vision_processing_components))
- **[F]** Video models use `AutoVideoProcessor` / `BaseVideoProcessor`, which handles decoding, frame sampling, resizing and normalization. (same source)

**Pipelines**
- **[F]** `transformers.pipeline(task, model=...)` wraps processor, model and post-processing for tasks such as `object-detection` and `zero-shot-object-detection`. ([DETR docs](https://huggingface.co/docs/transformers/v5.14.0/model_doc/detr), [zero-shot OD guide](https://huggingface.co/docs/transformers/v5.8.0/tasks/zero_shot_object_detection))

**Preprocessing vs augmentation**
- **[F]** HF's docs separate *augmentation* from *preprocessing*. Augmentation is done with torchvision, Albumentations or Kornia, for training. Preprocessing is done with the image processor, to match the pretrained model's expected input. ([image processors guide](https://huggingface.co/docs/transformers/v5.14.0/image_processors.md))

**What this means for Vistral [R]**
- Do not reimplement HF resize, normalize, crop or pad, or any `post_process_*` method. Call the native processor.
- Vistral's job at the HF boundary:
  1. Convert color order once. OpenCV gives BGR; HF expects RGB.
  2. State the channel layout explicitly.
  3. Supply `target_sizes` as (H, W) from tracked image metadata.
  4. Convert HF results into `sv.Detections` via `sv.Detections.from_transformers(...)`.
  5. Map coordinates back to the source image when Vistral geometry stages ran *before* the HF processor.
- Keep the three HF responsibilities separately addressable: the processor (an encoder), the model (inference) and post-processing (a decoder). Some models need custom decoders.
- Use a `transformers.pipeline` fallback for tasks with no typed Vistral adapter.
- Pin `transformers>=5,<6`.

### 2.2 `huggingface_hub`, `datasets`, `timm`, `diffusers`
- **[A]** `huggingface_hub` provides model download, caching and revision pinning through `from_pretrained(..., revision=)`. Vistral relies on it only indirectly through transformers, and records the `revision` in provenance.
- **[A]** `datasets` is useful for evaluation in the benchmark and research track, not at runtime. It is not a dependency.
- **[A] [Q]** `timm` image models are on the Hub. Integrating them as a `hf.TimmClassifier` or `hf.Embedder` backbone is deferred to M7, and the API needs verifying first.
- **[R]** Leave out `diffusers`. Image generation is outside the industrial QC scope.

---

## 3. OpenCV

### 3.1 Versions and packaging
- **[F]** OpenCV **5.0.0** was released in June 2026, the first major release since 4.0 (2018). It brings a rewritten graph-based DNN engine with broader ONNX operator coverage, requires C++17, and removes the legacy C API. ([Wikipedia](https://en.wikipedia.org/wiki/OpenCV), [LearnOpenCV: OD with OpenCV 5](https://learnopencv.com/how-to-run-object-detection-with-opencv-5/))
- **[F]** PyPI has four mutually exclusive wheels: `opencv-python`, `opencv-contrib-python`, `opencv-python-headless` and `opencv-contrib-python-headless`. Only one should be installed per environment. The headless builds avoid GUI and X11 dependencies. ([OpenCV pip tutorial](https://docs.opencv.org/5.x/db/dd1/tutorial_py_pip_install.html))
- **Decision:** support OpenCV **4.10+ and 5.x**. See [DECISIONS.md](DECISIONS.md) D-003.

### 3.2 Capabilities relevant to Vistral
The following are long-standing `cv2` APIs from the [main docs](https://docs.opencv.org/4.x/). They are not re-verified function by function, and M2 confirms each signature on both 4.10 and 5.x.

| Area | cv2 functions | Expose in Vistral as |
|---|---|---|
| Image IO | `imread`, `imwrite`, `imdecode`, `imencode` | `read_image`, `ImageWriter` |
| Video IO | `VideoCapture`, `VideoWriter` | `VideoSource`, `VideoWriter` |
| Color | `cvtColor` | `ToGray`, `ToRGB`, `ToBGR`, `ConvertColor` |
| Geometry | `resize`, slicing, `copyMakeBorder`, `warpAffine`, `warpPerspective`, `rotate`, `flip` | `Resize`, `Crop`, `Pad`, `Letterbox`, `Rotate`, `Flip`, all with recorded transforms |
| Smoothing / denoising | `GaussianBlur`, `medianBlur`, `bilateralFilter`, `fastNlMeansDenoising[Colored]` | built-in stages |
| Sharpening | `addWeighted` (unsharp mask), `filter2D`, `detailEnhance` | `Sharpen`; others via `cv.wrap` |
| Intensity | `convertScaleAbs`, `normalize`, `equalizeHist`, `createCLAHE`, `LUT` | built-in stages |
| Threshold / morphology | `threshold` (incl. Otsu), `adaptiveThreshold`, `morphologyEx`, `getStructuringElement` | built-in stages |
| Edges / gradients | `Canny`, `Sobel`, `Laplacian` | built-in stages |
| Shape analysis | `findContours`, `contourArea`, `minAreaRect`, `boundingRect`, `connectedComponentsWithStats` | `FindContours`, `Measure` |
| Matching | `matchTemplate`, `absdiff` | `TemplateMatch`, `ReferenceDiff` (QC) |
| Motion | `createBackgroundSubtractorMOG2/KNN`, `calcOpticalFlowFarneback` | deferred to M7 |
| Calibration | `calibrateCamera`, `undistort`, `findHomography` | `Calibration` (px/mm or homography) in M4; full calibration in M7 |
| NMS | `cv2.dnn.NMSBoxes` | not used; Supervision's `with_nms` is used instead |

- **[F]** OpenCV has its own graph composition API, **G-API**. It declares a graph of operations over `cv::GMat`, captures it as a `GComputation`, and applies it per frame. Its docs say the API "is volatile" and still in active development. ([G-API docs](https://docs.opencv.org/5.x/d0/d1e/gapi.html))
- **[R]**
  - Wrap individual cv2 operations as thin stages, one per operation, with keyword arguments that match cv2 names where they are clear.
  - Provide `cv.wrap(fn, **kwargs)` so *any* cv2 function, or any `ndarray -> ndarray` function, becomes a stage in one line.
  - Do not use G-API in v1. It may be evaluated in M7 as an optional fusion backend for pure-OpenCV chains.
  - Never reimplement an operation cv2 already provides.

---

## 4. Other technologies

| Technology | Status [F/A] | Recommendation |
|---|---|---|
| **PyTorch** | [F] 3.14 wheels have shipped since 2.10. 2.15 (GA 2026-10-28) drops Python 3.10. ([notice](https://dev-discuss.pytorch.org/t/notice-python-3-10-support-is-being-removed-from-pytorch-2-15-2-14-is-the-last-release-with-3-10-wheels/3440)) | Required only by the `[hf]` extra. Vistral core must import without torch. Python floor ≥3.11. |
| **torchvision v2 / `tv_tensors`** | [F] Typed `Image`, `BoundingBoxes` (format + `canvas_size`), `Mask`, `Video` and keypoints, which transforms keep consistent. ([docs](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html)) | Closest prior art for tensors that carry their own metadata. Vistral's `ImageSpec` and transform tracking apply the same idea to NumPy/OpenCV data. torchvision is pulled in via `[hf]` as HF's default processor backend. |
| **TorchCodec** | [F] PyTorch's video decoder, on CPU or CUDA. torchvision 0.28 removed its video APIs in favor of it. ([releases](https://github.com/meta-pytorch/torchcodec/releases)) | Optional `[video]` backend. `cv2.VideoCapture` stays the default. |
| **Kornia** | [F] Differentiable PyTorch CV library. ([arXiv 1910.02190](https://arxiv.org/abs/1910.02190)) | Optional GPU filter chains, deferred to M7. |
| **ONNX Runtime / OpenVINO / TensorRT** | [A] Not investigated in depth during this pass. | Not in v1. A `Model` stage can wrap any of them (see the custom model example in [API_DESIGN.md](API_DESIGN.md)). Built-in adapters come in M7 if needed. OpenVINO matters because Anomalib exports to it. |

---

## 5. Adjacent frameworks (overlap analysis)

| Tool | What it already provides | Overlap | How Vistral relates |
|---|---|---|---|
| **Supervision** (Roboflow, MIT, v0.31.1) | [F] `Detections` (`xyxy`, `mask`, `confidence`, `class_id`, `tracker_id`, `data`, `metadata`), loaders including `from_transformers(results, id2label)`, NMS/NMM, annotators, zones, `InferenceSlicer`, metrics and dataset formats. Its `requires_dist` has no OpenCV package. ([docs](https://supervision.roboflow.com/latest/), [Detections API](https://supervision.roboflow.com/latest/detection/core/), [PyPI](https://pypi.org/pypi/supervision/json)) | High for results and annotation | **A core dependency.** `sv.Detections` is Vistral's detection type, and annotators, NMS, slicing and zones are reused (D-004). |
| **Roboflow `trackers`** (Apache-2.0) | [F] SORT, ByteTrack and OC-SORT via `tracker.update(detections)`. ([docs](https://trackers.roboflow.com/latest)) | High for tracking | Optional `[track]` backend in M7. Vistral writes no trackers. |
| **Roboflow Inference / Workflows** | [F] Block-based workflow engine (40+ blocks covering models, logic and filters) running on CPU, GPU and Jetson. ([docs](https://docs.roboflow.com/workflows/readme.md)) | High for orchestration | Vistral does **not** build a workflow engine (D-001). |
| **torchvision v2** | See §4 | Medium for typed tensors | Prior art |
| **OpenCV G-API** | See §3 | Medium for composing cv ops | Possible M7 backend |
| **MediaPipe Framework** | [F] Graph and calculator pipelines. Support for the "Legacy Solutions" ended 2023-03-01 in favor of the Tasks API. ([guide](https://developers.google.com/edge/mediapipe/solutions/guide)) | Low; mobile and on-device focus | Shows graph frameworks are costly to adopt and maintain |
| **Savant / NVIDIA DeepStream** | [F] Streaming video pipelines built from YAML blocks over DeepStream/GStreamer for NVIDIA hardware. ([GitHub](https://github.com/insight-platform/savant)) | Low; streaming infrastructure | Out of scope. Its YAML block configuration supports Vistral's recipe approach (D-007). |
| **Anomalib** (Apache-2.0 per its PyPI listing) | [F] Library of deep anomaly-detection models for industrial inspection, with OpenVINO export. ([PyPI](https://pypi.org/project/anomalib), [Akcay et al. 2022, arXiv 2202.08341](https://arxiv.org/abs/2202.08341)) | Complementary | Optional `[anomaly]` adapter in M7. Confirm the license from the repo's LICENSE file. |
| **FiftyOne / Pixeltable** | [F] Dataset curation and multimodal tables with HF integrations. ([FiftyOne HF plugin](https://docs.voxel51.com/plugins/plugins_ecosystem/transformers.html), [Pixeltable models](https://docs.pixeltable.com/integrations/models)) | Low; data tooling | Possible interop later |

### 5.1 Licensing constraints
- **[F]** In mid-2025 Albumentations moved development to **AlbumentationsX**, licensed AGPL-3.0 or commercially. The last MIT release (2.0.8) is frozen. ([license page](https://albumentations.ai/docs/license/))
- **[F]** Ultralytics YOLO is AGPL-3.0 or available under a paid Enterprise license, and that applies to code, models and fine-tuned weights. ([license](https://ultralytics.com/license))
- **[R]** Vistral is Apache-2.0 (D-002). No AGPL package may be a core dependency or a declared extra. Users who choose to install such packages can wrap them themselves with the public extension API.

---

## 6. Research: the cost of composing preprocessing and inference

- **[F]** *Beyond Inference* (2024) measured ViT served through Triton. The share of latency spent on preprocessing grows with image size and reaches **97% (CPU preprocessing) / 88% (GPU preprocessing)** for the largest images. ([arXiv 2403.12981](https://arxiv.org/abs/2403.12981))
- **[F]** Kang et al. (PVLDB 14) found ResNet-50 inference on a T4 has ~9× the throughput of CPU image preprocessing, which makes preprocessing the bottleneck. They proposed optimizing preprocessing and inference jointly. ([paper](http://www.vldb.org/pvldb/vol14/p87-kang.pdf))
- **[F]** On Jetson edge devices, JPEG decoding is ~52–87% of preprocessing time depending on image size. ([arXiv 2608.27655](https://arxiv.org/abs/2608.27655))
- **What this means for Vistral [R]**
  - Abstraction overhead and redundant conversions or copies can matter just as much as model time. Vistral must make them **visible**, through per-stage tracing and conversion and copy counts, and must **measure** them against direct-library baselines ([BENCHMARK_PLAN.md](BENCHMARK_PLAN.md)).
  - Never assume a layer is free.
  - Industrial lines often run high-resolution images under fixed cycle-time budgets. That makes this especially relevant.

---

## 7. Gaps and possible differentiation

What already exists:
- Supervision: typed detections and annotation.
- torchvision: typed tensors.
- `transformers.pipeline`: end-to-end HF inference.
- Roboflow Workflows and Savant: block composition.

**[A] Gap.** In this review I found no library that does all three of the following *before execution*, across a **mixed OpenCV → HF → postprocess** chain:
1. **Checks image-format contracts:** color order, layout (HWC/CHW), dtype, value range and device.
2. **Tracks geometric transforms** (ROI, resize, letterbox, pad, tiling) so model outputs are mapped back to source pixels correctly.
3. **Serializes the whole pipeline as a versioned, reviewable recipe**, together with provenance (model revisions, package versions, recipe hash). This matters for industrial QC audit.

Partial coverage exists: torchvision tracks `canvas_size` for boxes, and HF infers channel layout. This is a weak novelty claim. Treat it as an **engineering contribution** whose value needs to be demonstrated. The research hypothesis and experiment are in [BENCHMARK_PLAN.md §6](BENCHMARK_PLAN.md).

**Not differentiators.** Vistral will not claim:
- "unified detections" (Supervision has this),
- "block-based pipelines" (Workflows has this),
- "GPU preprocessing" (HF and torchvision have this).

---

## 8. Items not verified in this pass

| Item | Verify in |
|---|---|
| `sv.Classifications` and `sv.KeyPoints` APIs, and whether they are adequate as result types | M1 |
| `sv.InferenceSlicer` callback signature; `sv.PolygonZone` trigger API | M4 |
| HF v5 kwarg names for channel format (`input_data_format`?) and model dtype (`dtype` vs `torch_dtype`) | M3 |
| Concrete tiny HF checkpoints for CI | M3 |
| Whether OpenCV 4.10.x wheels support numpy 2, and whether Python 3.14 wheels exist for 4.10 | M1 CI setup |
| Anomalib license (repo LICENSE) and API | M7 |
| docs.opencv.org and learnopencv.com links. Both sites returned HTTP 403 to automated fetches on 2026-10-21; the URLs and contents come from search-index results. Re-check them in a browser. | M2 |
| Industrial camera SDKs (GenICam via `harvesters` or vendor SDKs) | M7 |
| ONNX Runtime / OpenVINO adapter APIs | M7 |
