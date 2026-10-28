# Risks and Open Questions

**Status:** Planning phase (2026-10-21). Likelihood (L) and impact (I) are rated H/M/L.

## 1. Technical risks

| # | Risk | L | I | Mitigation | Milestone |
|---|---|---|---|---|---|
| R1 | **HF API churn.** v4→v5 already changed processor backends, and post-processing names and returns vary by model. | H | H | Depend only on public APIs (`AutoImageProcessor`, `AutoModelFor*`, `post_process_*`). Keep a per-task adapter table, a raw-output escape hatch and a custom-`Decoder` path. Pin `transformers>=5,<6` and run a floor/latest CI matrix. | M3 |
| R2 | **Supervision pre-1.0 churn**, now that it's a core dependency (D-004) | M | H | Pin `>=0.31,<0.32`. Keep all sv calls in a few modules. Add contract tests on the sv features Vistral uses (`Detections` fields, `from_transformers`, `with_nms`, `InferenceSlicer`, `PolygonZone`, annotators). | M1–M4 |
| R3 | **Heavy core install** from Supervision's transitive dependencies (`av`, `matplotlib`, `scipy`) | H | M | Document the install size. If edge deployments object, check whether Supervision could make these optional upstream, or revisit D-004. | M1 |
| R4 | **OpenCV wheel conflicts** (headless vs GUI `opencv-python` both installed) | M | M | Document the conflict. Provide a `[gui]` extra. Detect it at import time and warn. | M1 |
| R5 | **OpenCV 4.10 vs 5.x differences** in API behavior and wheel availability (Python 3.14, numpy 2) | M | M | Use only APIs common to both versions. Run the CI matrix. Put version-specific branches in one module. *M1: the core suite passes on 4.10.0 and on 5.0.0.* | M1–M2 |
| R14 | **Pixel-convention mismatches** between Vistral's corner-based coordinates and OpenCV's center-based sampling | M | H | *Found in M1, fixed and generalized before M2:* every OpenCV warp goes through `vistral.core.geometry.warp_array` / `pixel_center_matrix`, which conjugate by a half-pixel shift. Ground-truth tests (`tests/unit/test_pixel_convention.py`) cover 11 transform kinds with 0 mismatched pixels. Guard tests prove the uncorrected call drifts: 126 px on a ×2 upscale, 32 px on a flip or 90° rotation. Crop, pad and integer downscale are unaffected, which is why the original M1 test missed it. **M2 rule:** cv geometry stages must call `warp_array` (or `pixel_center_matrix` for `cv2.resize`-style APIs) and add a case to that test file. | M2 |
| R15 | **Tooling quirks on Windows/Python 3.14.** `--cov=<subpackage>` re-imports numpy ("cannot load module more than once"), and uv hit OS error 1450 under high download concurrency | L | L | Measure `--cov=vistral` and gate core with `coverage report --include`. Lower `UV_CONCURRENT_DOWNLOADS` if installs fail. | M1 |
| R6 | **Hidden overhead.** The abstraction adds latency, copies or conversions. | M | H | Trace counters, zero-copy conversion graph, M6 benchmarks against direct baselines, published results. | M1, M6 |
| R7 | **Spec metadata drifts from the actual data** (a custom stage declares the wrong spec) | M | M | Runtime checks in debug mode. `check_stage` verifies that declared output matches actual output. | M1 |
| R8 | **Coordinate mapping errors** through ROI, tiling, letterbox or rotation | M | H | 3×3 transform composition, property tests (forward then inverse is the identity), ground-truth synthetic fixtures. Rotation re-boxing loss is documented. | M2, M4 |
| R9 | **Recipe security.** Loading by import path executes code. | M | H | Built-ins-only by default, `allow_imports` allowlist, `yaml.safe_load`, no expression evaluation. | M5 |
| R10 | **Recipe schema evolution** breaks deployed lines | M | H | `vistral: N` schema version, migrations, deprecation period, round-trip tests. | M5 |
| R11 | **GPU memory and resource leaks** in long-running stations | M | H | `close()` contract, context managers, 10k-frame soak test. | M6 |
| R12 | **Thread-safety misuse** (sharing one pipeline across threads) | M | M | Document one instance per thread. Optional debug guard that detects concurrent `process` calls. | M1 |
| R13 | **Nondeterministic verdicts** on GPU, which affects QC auditability | L | H | Document the torch determinism settings. Record device and versions in provenance. Make the REVIEW band configurable. | M4 |

## 2. Product and maintenance risks

| # | Risk | L | I | Mitigation |
|---|---|---|---|---|
| P1 | **Not differentiated enough** from Supervision, `transformers.pipeline` and Roboflow Workflows | M | H | Focus on what's verified to be missing (contracts, coordinate tracking, recipes with provenance for QC). Run the H1/H2 experiment ([BENCHMARK_PLAN.md §6](BENCHMARK_PLAN.md)) and publish the results whichever way they go. |
| P2 | **Scope creep** toward a workflow engine or streaming server | H | H | D-001. Every M7 item needs a written need and a measured benefit. Non-goals are in [VISION.md](VISION.md). |
| P3 | **License contamination** from AGPL packages slipping in via examples or extras | L | H | D-002. A CI license check (for example `pip-licenses`) on the core and extras dependency trees. |
| P4 | **Industrial data availability.** Proprietary images can't be committed, and public sets such as MVTec AD are non-commercial. | H | M | Synthetic defect generator. Public datasets used for evaluation only and never redistributed. |
| P5 | **Maintenance load.** Many upstream projects (cv2, HF, sv, torch) | H | M | Small catalog, adapter modules isolated per upstream, scheduled latest-deps CI, clear support policy. |

## 3. Open questions

| # | Question | Default if not answered | Resolve by |
|---|---|---|---|
| Q1 | ~~Are `sv.Classifications` and `sv.KeyPoints` adequate result types?~~ **Resolved in M1:** no, because neither has a metadata field. Vistral types with `.to_supervision()` are used (D-017) | — | ✅ M1 |
| Q2 | ~~numpy floor?~~ **Resolved in M1:** `numpy>=1.26`, verified on Python 3.11 + numpy 1.26.4 + OpenCV 4.10.0 (D-016) | — | ✅ M1 |
| Q3 | Which tiny HF checkpoints to use for CI, per task | `hf-internal-testing` tiny models where they exist | M3 |
| Q4 | HF v5 kwarg names for channel format and model dtype | Verify against the installed v5 source | M3 |
| Q5 | `sv.InferenceSlicer` / `sv.PolygonZone` API details for `Tiled` / `InZone` | Wrap them if they fit; otherwise implement tiling with sv NMS | M4 |
| Q6 | Use MVTec AD (non-commercial) for evaluation? | Evaluation only, opt-in download, never redistributed | M4 |
| Q7 | Which industrial camera SDKs (GenICam via `harvesters`, vendor SDKs) and result sinks (MQTT, OPC UA, PLC) matter to you? | None in v1 | M7 |
| Q8 | Anomalib integration: license confirmation and API fit | Optional `[anomaly]` extra, M7 | M7 |
| Q9 | Docs tooling (mkdocs-material vs Sphinx) | mkdocs-material | M5 |
| Q10 | Is Windows a first-class deployment target for production lines? (It affects CI weight and video backends.) | Yes: Windows and Linux in CI | M1 |
