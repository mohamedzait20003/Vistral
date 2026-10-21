# Decisions

**Format:** each entry gives the ID, status, decision, alternatives considered, rationale, and the assumptions it depends on.

**Status values**
- **Decided (user):** confirmed by the project owner.
- **Proposed:** recommended in planning, and adopted unless the owner objects.
- **Open:** still to be resolved.

Last updated: 2026-10-21.

---

### D-001: Architecture: composable component framework (Option 2)
- **Status:** Proposed (plan approved 2026-10-21)
- **Decision:**
  - one shared `Stage` protocol, a `Frame` envelope, explicit `ImageSpec` contracts, typed results;
  - a sequential `Pipeline` with `Branch`/`When`;
  - YAML recipes.
  - No scheduler, DAG compiler or distributed execution.
- **Alternatives:**
  - (1) Thin integration layer: too little value over calling Supervision and HF directly.
  - (3) Full workflow engine: duplicates Roboflow Workflows, DeepStream/Savant and OpenCV G-API, at very high cost.
- **Rationale:** the simplest design that fixes the real integration problems (contracts, coordinates, recipes). See [ARCHITECTURE.md §1](ARCHITECTURE.md).
- **Assumption:** sequential execution is fast enough for the industrial QC workloads in v1. M6 validates this.

### D-002: License: Apache-2.0
- **Status:** Decided (user, 2026-10-21)
- **Consequence:** no AGPL dependency in core or in any declared extra. This excludes AlbumentationsX ([license](https://albumentations.ai/docs/license/)) and Ultralytics ([license](https://ultralytics.com/license)). Users may wrap such packages themselves.

### D-003: OpenCV support: 4.10+ and 5.x
- **Status:** Decided (user, 2026-10-21)
- **Decision:**
  - `opencv-python-headless>=4.10,<6` in core;
  - use only APIs present in both 4.10 and 5.x, with 5.x-only features behind version checks;
  - CI covers 4.10.x, latest 4.x and latest 5.x.
- **Open sub-question:** whether 4.10 wheels support numpy 2 and Python 3.14. This decides the numpy floor (see D-016).

### D-004: Detection result type is `sv.Detections`
- **Status:** Decided (user, 2026-10-21)
- **Decision:**
  - Supervision is a core dependency, and `sv.Detections` is used directly with no wrapper.
  - Vistral provenance goes in `metadata["vistral"]`; per-detection extras go in `data`.
  - Supervision's NMS, slicer, zones and annotators are reused.
- **Alternatives:** an own minimal type with interop. Rejected by the owner.
- **Consequences:**
  - Heavier core dependencies: `av`, `matplotlib`, `scipy` come in transitively ([PyPI metadata](https://pypi.org/pypi/supervision/json)).
  - Exposure to pre-1.0 API churn, mitigated by pinning `>=0.31,<0.32` and adding contract tests on the sv features Vistral uses.
  - Supervision declares no OpenCV dependency, so it doesn't conflict with the headless wheel.

### D-005: `Frame` envelope flows between stages
- **Status:** Proposed
- **Decision:**
  - stages exchange a `Frame` (current image, original image, results, artifacts, source info, errors), not bare values;
  - it is copied shallowly at each stage boundary.
- **Alternatives:**
  - Bare value passing: fails as soon as one stage needs two inputs, such as an annotator needing image plus detections.
  - Full DAG with named ports: that's Option 3.
- **Rationale:** the minimum structure mixed pipelines need. Function decorators hide the envelope for simple stages.

### D-006: No forced canonical image format; explicit, tracked specs
- **Status:** Proposed
- **Decision:**
  - Images carry an `ImageSpec` (layout, color, dtype, range, device, size).
  - CPU work defaults to OpenCV-native NumPy HWC u8 BGR.
  - Conversions happen only where a stage declares a need. They go through the conversion graph, use zero-copy where legal, and are traced.
  - PIL is supported only at the edges.
- **Alternatives:**
  - Always torch CHW float: forces conversions on every cv2 op.
  - Always RGB: forces conversions on every cv2 read and write.
- **Rationale:** avoid needless copies (see the research in [RESEARCH_REVIEW.md §6](RESEARCH_REVIEW.md)), and make color order explicit, since it can't be inferred from the data.

### D-007: YAML recipes adopted (M5), Python-first
- **Status:** Decided (user asked for it "if it is better"; assessed as better for industrial QC)
- **Rationale:**
  - per-product and per-line recipes;
  - non-developers can tune parameters;
  - versioning, diff and audit;
  - precedent in Savant's YAML pipelines.
- **Decision:**
  - YAML serializes the Python API, with a round-trip guarantee.
  - Built-ins are looked up by short name; custom stages by import path, only behind an `allow_imports` allowlist.
  - Parsing uses `yaml.safe_load` only, and validation uses pydantic v2.
  - Conditions are structured mappings, **never evaluated expressions**.
  - The schema has a `vistral: N` version key.
- **Consequence:** from M1 onward, every stage's constructor parameters must be config-serializable.

### D-008: No plugin registry beyond name lookup
- **Status:** Proposed
- **Decision:**
  - Python composition needs no registry.
  - YAML uses a static built-in name map plus allowlisted import paths.
  - Entry-point discovery comes only if third-party packages ask for it.
- **Rationale:** avoid premature plugin infrastructure.

### D-009: Coordinate and size conventions
- **Status:** Proposed
- **Decision:**
  - Sizes are always (H, W).
  - Boxes are absolute xyxy float pixels.
  - Coordinates are continuous and corner-based: pixel *i* spans [i, i+1).
  - Geometric transforms are recorded as 3×3 matrices.
  - `ToSourceCoords` maps results back to source coordinates.
- **Rationale:** matches HF `post_process_object_detection` output (absolute xyxy) and `sv.Detections.xyxy`. Removes the PIL (W, H) ambiguity that HF's own examples have to work around (`image.size[::-1]`, [DETR docs](https://huggingface.co/docs/transformers/v5.14.0/model_doc/detr)).

### D-010: Python ≥ 3.11
- **Status:** Proposed
- **Rationale:** PyTorch 2.15 drops 3.10 ([notice](https://dev-discuss.pytorch.org/t/notice-python-3-10-support-is-being-removed-from-pytorch-2-15-2-14-is-the-last-release-with-3-10-wheels/3440)). The development machine runs 3.14. CI covers 3.11–3.14.

### D-011: Core imports without torch/transformers
- **Status:** Proposed
- **Decision:**
  - `vistral.hf` is the extra `[hf]` and imports lazily.
  - An import-linter test enforces the layering rules.
- **Rationale:** classic cv-only inspection deployments shouldn't need a ~GB torch install.

### D-012: Synchronous execution only in v1
- **Status:** Proposed
- **Rationale:** no demonstrated need yet, and async complicates stage authoring. The protocol leaves room for a future `aprocess`. Revisit in M7 with data from M6.

### D-013: Stages never mutate inputs
- **Status:** Proposed
- **Decision:** annotators and other in-place cv2 operations copy first. The contract-test kit checks for mutation.

### D-014: Primary domain: industrial and production inspection, QC object detection
- **Status:** Decided (user, 2026-10-21)
- **Consequences:**
  - `vistral.qc` (ROI, tiling, measurement, calibration, verdicts, golden-reference diff, template matching);
  - synthetic defect fixtures;
  - 16-bit image support in IO;
  - cycle-time-oriented benchmarks (5 MP images);
  - provenance for audit.

### D-015: Reuse Supervision annotators; no own drawing toolkit
- **Status:** Proposed (follows from D-004)
- **Decision:** only QC-specific overlays (verdict banner, measurements) and contours are drawn with cv2 directly.

### D-016: numpy floor
- **Status:** Open
- **Proposal:** `numpy>=1.26`, raised to `>=2` if the OpenCV 4.10 wheels allow it. The M1 CI matrix decides.

### D-017: Classification and keypoint result types
- **Status:** Open
- **Proposal:** use `sv.Classifications` / `sv.KeyPoints` if their APIs fit, otherwise thin Vistral dataclasses. Decided during M1.

### D-018: Packaging and tooling
- **Status:** Proposed
- **Decision:** `uv`, hatchling, `src/` layout, ruff, pyright, pytest. Planning docs live in `docs/`.

---

## Assumptions register

| ID | Assumption | Validated by |
|---|---|---|
| A-1 | Sequential execution meets QC cycle times for typical single-camera stations | M6 benchmarks |
| A-2 | Per-stage overhead can be kept ≤20 µs with tracing off | M6 |
| A-3 | `sv.InferenceSlicer` accepts an arbitrary detector callback that returns `sv.Detections` | M4 |
| A-4 | HF v5 processors accept an explicit channel-format kwarg for NumPy input | M3 |
| A-5 | Tiny HF checkpoints exist for each typed task, for CI | M3 |
| A-6 | Most cv2 calls release the GIL (relevant to future multi-camera threading) | M7 |
