# Vision

**Status:** Planning phase (2026-10-21). Decisions are referenced as D-xxx; see [DECISIONS.md](DECISIONS.md).

## 1. What Vistral is

Vistral is a small Python library for building **validated, inspectable computer-vision pipelines**. A pipeline is assembled from:
- OpenCV operations,
- Hugging Face models (with their native processors and post-processing),
- Supervision results and annotators,
- your own components.

It doesn't replace any of those libraries. It runs them natively and adds four things on top:

1. **One composition model.** Every step is a `Stage`, and stages chain into a `Pipeline`.
2. **Explicit data contracts.** Color order, layout, dtype, value range, device and coordinate space are declared, checked before execution, and converted explicitly.
3. **Observability.** A trace records per-stage timing, conversions and copies.
4. **Recipes.** A pipeline can be saved as a versioned YAML file with provenance, so it can be reviewed, deployed and audited.

You can always reach the native object underneath: the `np.ndarray`, the `torch.Tensor`, the HF model or processor, the `sv.Detections`.

## 2. Primary domain (D-014)

**Industrial and production inspection, and object detection for quality control.** This means:
- defect detection,
- presence/absence checks,
- dimensional measurement,
- pass/fail verdicts per part,
- under cycle-time budgets,
- with full traceability of which model and settings produced each decision.

The core stays general-purpose. The domain sets priorities for built-ins, examples and benchmarks.

## 3. Target users

| Persona | Needs | What Vistral offers |
|---|---|---|
| **Vision/ML engineer** at a manufacturer or integrator | Combine classic cv2 inspection with HF detectors without hand-writing the glue (BGR/RGB, coordinate mapping, tiling) | Typed stages, `hf.Detector`, `Tiled`, `ToSourceCoords`, validation |
| **Process / automation engineer** | Adjust thresholds, ROIs and tolerances per product without editing Python, and know what changed | YAML recipes, readable errors, recipe hashes |
| **Researcher / prototyper** | Try several models and preprocessing variants quickly and compare cost per stage | Swappable stages, trace, benchmark harness |
| **Component author** | Add a custom filter, decoder or model without forking Vistral | Function decorators, `Stage` subclasses, `cv.wrap`, contract-test kit |

## 4. Use cases, in priority order

| # | Use case | Example pipeline | Milestone |
|---|---|---|---|
| 1 | Classic image processing | `ToGray → GaussianBlur → Canny → FindContours → Measure` | M2 |
| 2 | Golden-reference inspection (no ML) | `Align (TemplateMatch) → ReferenceDiff → Morphology → FindContours → Verdict` | M4 |
| 3 | HF detection with typed output | `hf.Detector → FilterDetections → DrawDetections` | M3 |
| 4 | Mixed QC detection | `ROI → CLAHE → Tiled(hf.Detector) → FilterDetections → InZone → ToSourceCoords → Measure → Verdict → sinks` | M4 |
| 5 | Classification / segmentation / embeddings | `hf.Classifier`, `hf.SemanticSegmenter`, `hf.Embedder` | M3 |
| 6 | Offline video analytics | `VideoSource → FrameSampler → … → VideoWriter` | M2/M4 |
| 7 | Recipe-driven deployment | `vistral run recipe.yaml --source images/` | M5 |
| 8 | Anomaly detection, tracking, industrial cameras | Anomalib, `trackers`, GenICam | M7 (optional) |

## 5. Goals

- **G1 Reuse, not reimplement.** Every built-in maps to a native call (cv2, HF, Supervision). If a built-in is a wrapper, its output must equal the direct call's output (tested).
- **G2 Catch integration errors early.** Common silent bugs fail validation before any pixels are processed: BGR fed to an RGB model, (w,h)/(h,w) swaps, double normalization, CHW/HWC mixups, boxes left in resized coordinates.
- **G3 Little boilerplate.** A one-line function becomes a stage. Simple operations read like `cv.GaussianBlur(ksize=5)`.
- **G4 Measurable overhead.** Framework overhead is benchmarked against direct-library baselines and published. It is never assumed to be zero.
- **G5 Reproducibility.** A result can be traced to its recipe hash, model revisions and package versions.
- **G6 Light core.** `import vistral` needs neither torch nor transformers (D-011).

## 6. Non-goals (v1)

- Training, fine-tuning or data augmentation.
- A workflow engine: scheduling, DAG compilation, distributed or multi-node execution (D-001).
- A streaming server, RTSP fan-out or DeepStream-class throughput.
- A GUI, low-code editor or labeling tool.
- A model zoo. Vistral points to HF Hub checkpoints.
- Rewriting what already exists: trackers (use `trackers`), annotators (use Supervision), HF post-processing.
- Mandatory cloud or remote inference. The interfaces allow it; v1 ships none.
- AGPL dependencies of any kind (D-002).

## 7. Success criteria for 1.0

1. The canonical mixed QC pipeline (use case 4) runs from Python and from YAML with identical results.
2. All faults in the fault-injection corpus are caught before execution or flagged at runtime ([BENCHMARK_PLAN.md §6](BENCHMARK_PLAN.md)).
3. Framework overhead is within the target agreed in M6 and published with methodology.
4. A third-party custom stage passes `vistral.testing.check_stage` without changes to Vistral.
5. CI is green across Python 3.11–3.14, OpenCV 4.10 / latest 4.x / latest 5.x, and the supported HF and Supervision ranges.
