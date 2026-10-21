# Implementation Plan

**Status:** Proposed (2026-10-21). **Approval gate:** each milestone starts only after explicit approval. M0 (this planning set plus the repo skeleton) is the only approved work so far. **M1 is the next milestone to approve.**

---

## 1. Milestones

### M0: Planning and skeleton ✅ (approved, this deliverable)
- **Deliverables:**
  - the 9 documents in `docs/`
  - `pyproject.toml` declaring dependencies (none installed)
  - empty `src/vistral/__init__.py`
  - `README.md`, Apache-2.0 `LICENSE`, `.gitignore`
  - `git init`
- **Exit:** the docs are internally consistent; `pyproject.toml` parses; no implementation code exists.

### M1: Core contracts and image representations
- **Deliverables** (`vistral.core`, `vistral.testing`):
  - `ImageSpec`, `Image`, `GeoTransform`, `Frame`, `SourceInfo`
  - `Result` / `ResultMeta`, Vistral result types, and the `metadata["vistral"]` convention on `sv.Detections`
  - the `Stage` protocol, role base classes, the `@vt.filter` / `@vt.postprocess` / `@vt.stage` decorators, and `get_config()`
  - `Pipeline` (with `|`, naming, nesting, `run`, `map`, `stream`, `on_error`), `Branch`, `When`
  - `convert` with a `ConversionGraph` covering NumPy conversions; torch edges are registered lazily
  - `validate()` with strict mode and `auto_convert` mode
  - `Trace`, the error hierarchy, `read_image` for NumPy/PIL input, `Convert`
  - `check_stage`, and the non-HF part of the fault corpus
  - CI: lint, type check, tests, import-linter
- **Tests:**
  - unit tests for every core type;
  - property tests for transform composition and inversion (`hypothesis`, dev-only);
  - contract tests applying `check_stage` to example stages;
  - fault corpus: BGR→RGB-only stage, CHW/HWC, U8 vs UNIT, double normalization, (W,H) size passed as (H,W), boxes in the wrong `coord_space`.
- **Acceptance criteria:**
  1. A pure-NumPy pipeline of 3 function filters runs and produces a trace.
  2. `validate()` catches 100% of the non-HF fault-corpus cases, and the error message suggests a fix.
  3. `auto_convert=True` inserts the minimal conversion path, which is reported.
  4. `import vistral` works with only core dependencies; the import-linter enforces the layering rules ([ARCHITECTURE.md §2](ARCHITECTURE.md)).
  5. Every stage's `get_config()` serializes to JSON and reconstructs an equal stage.
  6. ≥90% line coverage on `vistral.core`.
  7. Checked and recorded in DECISIONS.md: whether `sv.Classifications` and `sv.KeyPoints` fit as result types.
- **Exit:** CI is green on Python 3.11–3.14 × numpy floor/latest.

### M2: OpenCV integration
- **Deliverables:**
  - `vistral.cv` catalog sections 1–7 of [BUILTIN_COMPONENTS.md](BUILTIN_COMPONENTS.md)
  - `cv.wrap`
  - geometry stages that record `GeoTransform`, plus `ToSourceCoords`
  - `vistral.io` image and video (cv2 backend), sinks
- **Tests:**
  - each stage is bit-identical to the direct cv2 call on fixtures (u8 BGR, u8 GRAY, u16 GRAY, f32);
  - geometry round-trip: forward then `ToSourceCoords` recovers boxes within 1e-6 px for affine transforms;
  - video read/write round-trip on a generated clip;
  - `check_stage` passes for every built-in.
- **Acceptance criteria:**
  1. A custom cv2 function is wrapped in one line, and that's tested.
  2. The catalog works on OpenCV 4.10.x, latest 4.x and latest 5.x (CI matrix).
  3. No built-in mutates its input.
- **Exit:** examples 2.1, 2.4 (without TemplateMatch) and 3.1–3.2 from API_DESIGN run.

### M3: Hugging Face integration (extra `[hf]`)
- **Deliverables:**
  - `hf.Processor`, `hf.Model`, `hf.PostProcess`
  - task composites (`Classifier`, `Detector`, `ZeroShotDetector`, `SemanticSegmenter`, `Segmenter`, `Embedder`)
  - the `hf.Pipeline` fallback
  - lazy imports, `MissingDependencyError`, `warmup`, `revision` pinning and provenance
- **Tests:**
  - **equivalence:** Vistral output == raw `processor → model → post_process` output on the same RGB input, using tiny checkpoints ([Q] concrete model ids chosen at M3 start; prefer `hf-internal-testing` tiny models);
  - BGR input gives the same result as the raw pipeline fed RGB;
  - `target_sizes` is (H, W) for non-square images;
  - malformed outputs → `ModelOutputError`;
  - `-m slow` runs real checkpoints (DETR, ViT, SegFormer, DINOv2/CLIP);
  - `-m gpu` checks device placement.
- **Acceptance criteria:**
  1. Classification, detection, semantic segmentation and embedding pass equivalence.
  2. Detections convert to `sv.Detections` with `class_name` and provenance.
  3. A user can replace `hf.PostProcess` with a custom `Decoder`.
- **Exit:** example 2.2 runs; HF version-matrix CI (`transformers` 5.x floor and latest) is green.

### M4: Mixed pipelines, with industrial QC focus
- **Deliverables:**
  - `vistral.qc` (`ROI`, `Tiled`, `Calibration`, `Measure`, `Verdict` and rules, `ReferenceDiff`, `TemplateMatch`, draw helpers)
  - `InZone`, `Warp`, `ConnectedComponents`
  - the canonical QC example (API_DESIGN §2.3) and the golden-reference example (§2.4)
  - a synthetic defect-image generator in `vistral.testing.fixtures` (parts with scratches, dents and stains at known positions)
- **Tests:**
  - boxes in source coordinates are correct after ROI, Resize, Letterbox and tiling (ground truth from the generator);
  - deterministic verdicts;
  - per-part latency reported via trace;
  - video stream with `on_error="skip"`.
- **Acceptance criteria:**
  1. On the synthetic set, the classic `ReferenceDiff` pipeline gives the expected verdict for 100% of generated cases.
  2. Detection mapping error is ≤0.5 px through ROI + tiling.
  3. Both examples run end to end on CPU.
- **Exit:** `examples/` holds runnable industrial demos with README instructions.

### M5: Config, extensibility and usability
- **Deliverables:**
  - `vistral.config` (load, dump, registry, import allowlist, pydantic validation with file positions)
  - `vistral run` / `vistral validate` CLI
  - recipe hashing and provenance in result metadata
  - docs site (mkdocs-material [A]) with tutorials: custom filter, custom decoder, custom model, recipes
  - error-message review
- **Tests:**
  - YAML round-trip equality for every built-in;
  - every M4 example expressed as YAML produces **identical** outputs;
  - a disallowed import path is rejected;
  - malformed recipes report the correct line and column.
- **Acceptance criteria:** a third-party package defining a custom stage (fixture package in `tests/`) works from Python and from YAML without changes to Vistral.
- **Exit:** the docs build cleanly; the public API snapshot test is in place.

### M6: Performance and reliability
- **Deliverables:** the benchmark suite from [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md), a published results report, fixes for avoidable conversions and copies, a resource-cleanup audit (`close()` paths, video handles, CUDA memory).
- **Acceptance criteria:**
  1. Overhead targets met (proposed: ≤5% end-to-end on 640px mixed pipeline; ≤20 µs fixed per-stage overhead with trace off — **[A]** to be confirmed against baselines), or the misses documented with reasons.
  2. Zero redundant conversions in the canonical pipeline.
  3. No leaked file handles or growing memory over a 10k-frame soak test.
- **Exit:** benchmark results checked in, with a hardware and version manifest.

### M7: Optional advanced capabilities
Candidates, each needing a written justification (user need plus measured benefit) before it starts:
- tracking via `trackers`
- background subtraction and optical flow
- Anomalib adapter
- ONNX Runtime and OpenVINO backends
- TorchCodec video
- GenICam cameras
- parallel `Branch` / threaded multi-camera execution
- async
- remote model adapters
- G-API or Kornia fusion
- VLM/pose/timm adapters

---

## 2. Testing strategy

| Layer | What | Where | Default suite? |
|---|---|---|---|
| Unit | Each type and stage; cv stages vs direct cv2 (bit-exact) | `tests/unit` | yes |
| Contract | `check_stage` on every built-in and on example custom stages | `tests/contract` | yes |
| Validation / faults | Fault-injection corpus (`vistral.testing.fault_corpus`) | `tests/unit/test_faults.py` | yes |
| Compatibility | Input formats (ndarray BGR/GRAY/u16/f32, PIL, torch), color-space round-trips, non-contiguous arrays, (H,W) handling | `tests/unit` | yes |
| HF integration | Equivalence vs raw HF on tiny models | `tests/integration` | yes (tiny models, cached) |
| Mixed pipelines | OpenCV + HF + QC end to end | `tests/integration` | yes (tiny models) |
| Malformed outputs | Fake model outputs: missing keys, wrong shapes | `tests/unit` | yes |
| Large / slow | Real checkpoints | `-m slow` | no |
| GPU | Device placement, CUDA conversions | `-m gpu` | no |
| Regression | Golden images (exact for cv; tolerance for HF), golden JSON results | `tests/regression` | yes |
| Performance | pytest-benchmark vs direct-library baselines | `benchmarks/` | no (scheduled CI) |
| Recipes | YAML round-trip, Python≡YAML outputs | `tests/config` | yes (from M5) |
| API stability | Signature snapshot | `tests/test_public_api.py` | yes (from M5) |

**Fixtures:** synthetic images generated at test time (no binary blobs), plus a few small PNGs checked in under `tests/data` (<1 MB total). HF tiny models are cached in CI.

---

## 3. CI and dependency compatibility

- **Tooling:** `uv` (environments and lock), `ruff` (lint and format), `pyright` (types), `pytest` + `pytest-cov` + `pytest-benchmark`, `hypothesis`, `import-linter`.
- **Matrix:**
  - Python {3.11, 3.14}, plus 3.12/3.13 on Linux only
  - OS {ubuntu, windows}, plus macOS on a schedule
  - OpenCV {4.10.x, latest 4.x, latest 5.x}
  - numpy {floor, latest}
  - transformers {5.x floor, latest}
  - supervision {0.31.x}
  - To keep the matrix manageable, cross-combinations are sampled rather than fully crossed.
- **Lockfiles and updates:** `uv.lock` is committed for development reproducibility. The library declares ranges, not pins. A scheduled weekly "latest deps" job catches upstream breakage early.
- **Upper bounds:** `transformers<6`, `supervision<0.32` (pre-1.0), `opencv-python-headless<6`. Bounds are widened only after the matrix passes.

---

## 4. Priorities

1. **Contracts first (M1):** everything depends on them.
2. **Native correctness (M2, M3):** prove equivalence with the native libraries before adding value on top.
3. **Domain value (M4):** industrial QC demos.
4. **Recipes and usability (M5).**
5. **Performance evidence (M6).**
6. **Optional features (M7):** only on demand.
