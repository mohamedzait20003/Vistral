# Benchmark Plan

**Status:** Proposed (2026-10-21). **No measurements exist yet.** Nothing in this document is a measured result. Every target is labeled **[A]** and is confirmed or revised in M6.

**Why this matters.** Prior work shows preprocessing and data movement can dominate vision inference latency: up to 97% of latency on large images in *Beyond Inference* ([arXiv 2403.12981](https://arxiv.org/abs/2403.12981)), with CPU preprocessing as the bottleneck vs ResNet-50 on a T4 ([Kang et al., PVLDB 14](http://www.vldb.org/pvldb/vol14/p87-kang.pdf)). An abstraction layer that adds conversions or copies can make this worse. Vistral therefore has to measure its own cost.

---

## 1. Questions to answer

1. **Q1.** What fixed per-stage overhead does Vistral add over a direct function call, with tracing off and with it on?
2. **Q2.** What end-to-end overhead does Vistral add over hand-written cv2 code and hand-written HF code for the same work?
3. **Q3.** How many conversions and copies does each pipeline perform, compared with the minimum needed?
4. **Q4.** What do the individual conversions cost (BGR↔RGB, numpy↔torch, CPU↔GPU, PIL round-trip) at industrial image sizes?
5. **Q5.** How much do batching and model reuse help, and what do model loading and warm-up cost?
6. **Q6.** How do memory use and the error rate behave over long streams (soak test)?

## 2. Baselines

| ID | Baseline | Compared with |
|---|---|---|
| B1 | Hand-written cv2 chain: `cvtColor → GaussianBlur → Canny → findContours` | the same chain as a Vistral `Pipeline` |
| B2 | Raw HF: `AutoImageProcessor → AutoModelForObjectDetection → post_process_object_detection` (+ manual BGR→RGB) | `hf.Detector` |
| B3 | `transformers.pipeline("object-detection")` | `hf.Detector` |
| B4 | Hand-written mixed QC: cv2 CLAHE + HF detector + `sv.InferenceSlicer` + manual coordinate mapping + `sv` annotators | canonical Vistral QC pipeline ([API_DESIGN.md §2.3](API_DESIGN.md)) |
| B5 | Empty Python function call | an empty Vistral stage (isolates framework overhead) |

Each baseline is written idiomatically, the way a competent engineer would write it, with no deliberately slow code. Each one lives in `benchmarks/baselines/` and is reviewed for that.

## 3. Workloads

| Workload | Image size(s) (H×W) | Notes |
|---|---|---|
| W1 Classic chain | 480×640, 1080×1920, 2048×2448 (5 MP, typical industrial camera) | u8 BGR and u16 GRAY |
| W2 HF detection | 480×640, 1080×1920 | tiny model (overhead focus) and DETR-R50 (realistic) |
| W3 Mixed QC with tiling | 2048×2448, 640 tiles, 0.2 overlap | synthetic defect parts |
| W4 Conversion matrix | 640×640, 2048×2448 | each conversion in isolation |
| W5 Stream soak | 10,000 frames, 1080×1920 | video file source, `on_error="skip"`, injected 1% faulty frames |

## 4. Metrics

| Metric | How it's measured |
|---|---|
| Latency p50 / p95 / p99 per stage and end to end | `time.perf_counter_ns`; `torch.cuda.synchronize()` around GPU stages |
| Throughput (frames/s), single-stream and batched | wall time over N frames |
| Framework overhead | (Vistral − baseline) / baseline, per workload |
| Conversions and copies | Vistral `Trace` counters, cross-checked with an instrumented baseline |
| Peak memory | `tracemalloc` (Python), RSS via `psutil` (dev-only), `torch.cuda.max_memory_allocated` |
| Model load / warm-up time | cold process vs warm cache; first-call vs steady-state latency |
| Error behavior | failed-frame rate under fault injection; whether errors are attributed to the right stage and frame |

## 5. Method

- **Harness:** `pytest-benchmark` for micro and stage benchmarks, plus a small runner script for end-to-end runs and soak tests. Results go to `benchmarks/results/<date>-<host>.json`.
- **Protocol:**
  - discard ≥10 warm-up iterations;
  - run ≥200 timed iterations, or ≥30 s;
  - fix CPU frequency or set power plan to high performance where possible;
  - run with no other heavy processes;
  - `OMP_NUM_THREADS` and `cv2.setNumThreads` recorded and held fixed.
- **Reporting:**
  - a manifest with CPU and GPU models, RAM, OS, driver/CUDA, and the versions of Python, numpy, cv2, torch, transformers, supervision and vistral;
  - each result as median with 95% CI (bootstrap);
  - every claim in documentation links to a results file.
- **Correctness first:** a benchmark counts only if the Vistral and baseline outputs match (bit-exact for cv chains, tolerance for HF).
- **CI:** benchmarks don't gate PRs. A scheduled job runs B1, B5 and W4 on fixed runners and alerts on regressions above 10% [A].

## 6. Research experiment: are contract checks worth having?

This is kept separate from the product requirements. It tests the differentiation claim in [RESEARCH_REVIEW.md §7](RESEARCH_REVIEW.md).

- **Hypotheses**
  - **H1.** Declaring and propagating specs before execution (color order, layout, dtype, range, device, coordinate space) catches most common mixed-pipeline integration faults. Without it, those faults produce *silently wrong* outputs.
  - **H2.** Spec-aware conversion planning removes redundant conversions and copies compared with typical hand-written glue, at negligible validation cost.
- **Fault corpus** (`vistral.testing.fault_corpus`, ≥20 cases across these classes):
  1. BGR to an RGB model
  2. (W,H) passed as (H,W) for `target_sizes`
  3. double normalization
  4. 0–1 float fed where 0–255 is expected
  5. CHW/HWC swap
  6. boxes not mapped back after resize, ROI or tiling
  7. GRAY fed to a 3-channel model
  8. mask resolution mismatch
  9. wrong `id2label` mapping
  10. 16-bit image truncated to 8-bit
- **Procedure:**
  - Apply each fault to B2, B3 and B4, written as idiomatic hand code, and to the Vistral equivalent.
  - Classify each outcome as one of: *caught before execution*, *runtime error*, *silently wrong* (output differs from ground truth beyond tolerance), or *no effect*.
- **Metrics:**
  - fraction caught before execution or at runtime vs silently wrong, per system;
  - number of conversions and copies vs the theoretical minimum;
  - validation time compared with end-to-end latency.
- **Success criteria [A]:**
  - H1 is supported if Vistral turns at least 80% of the faults that are *silently wrong* for the baselines into caught errors.
  - H2 is supported if Vistral uses no more conversions than the minimum, and fewer than at least one baseline.
- **Reporting:** results are published whether they support the hypotheses or not. If they don't, the docs describe Vistral as a convenience library and make no claim of better correctness.

## 7. Targets [A] (proposals, to be revisited in M6)

| Target | Proposed value |
|---|---|
| Fixed per-stage overhead, trace off | ≤ 20 µs |
| Trace-on additional overhead | ≤ 10 µs per stage |
| End-to-end overhead vs B2/B4 at 640px | ≤ 5% |
| Redundant conversions in canonical QC pipeline | 0 |
| Soak test (W5) memory growth | none beyond warm-up plateau |
| Soak test faulty-frame attribution | 100% of injected faults attributed to the correct stage and frame |
