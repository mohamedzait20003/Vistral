# Vistral

**Status: M1 (core contracts) complete.** The OpenCV built-ins (M2) and Hugging Face integration (M3) are next. See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Try M1

```bash
py -m uv sync --extra dev          # or: uv sync --extra dev
.venv/Scripts/python examples/m1_quickstart.py   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest
```

M1 provides:
- image specs (color order, layout, dtype, value range, device, size as (H, W));
- stages and sequential pipelines with `Branch`/`When`;
- **validation before execution** with suggested fixes and optional lossless auto-conversion;
- coordinate tracking (`ToSourceCoords`);
- per-stage tracing;
- config round-trips;
- a contract-test kit for your own components (`vistral.testing.check_stage`).

```python
import vistral as vt
from vistral.testing import stubs   # stand-ins until real OpenCV/HF stages land in M2/M3

pipe = vt.Pipeline([stubs.RGBModelStub()])
pipe.validate(vt.ImageSpec.bgr_u8())
# SpecMismatchError: stage 0 'RGBModelStub' does not accept BGR uint8 HWC U8 ...
#   - color=BGR (allowed: {RGB})
#   Fix: insert vistral.Convert(color='RGB') before this stage [swap_rb], ...
```

## Target API (later milestones)

Vistral is a planned Python library for building **validated, inspectable computer-vision pipelines** from:
- OpenCV operations,
- Hugging Face models (using their native processors and post-processing),
- Supervision results and annotators,
- your own components.

Its main focus is **industrial and production inspection and QC**.

```python
# Design only — cv, hf and qc arrive in M2–M4
import vistral as vt
from vistral import cv, hf, qc

inspect = vt.Pipeline([
    qc.ROI(polygon=[(120, 80), (1800, 80), (1800, 1000), (120, 1000)]),
    cv.CLAHE(clip_limit=2.0),
    qc.Tiled(hf.Detector("org/defect-detr", threshold=0.4), tile=(640, 640), overlap=0.2),
    vt.FilterDetections(min_score=0.5),
    vt.ToSourceCoords(),
    qc.Verdict(rules=[qc.MaxCount(label="scratch", max=0)]),
])
```

## Planning documents

| Document | Contents |
|---|---|
| [docs/RESEARCH_REVIEW.md](docs/RESEARCH_REVIEW.md) | Existing libraries, research, capabilities, gaps |
| [docs/VISION.md](docs/VISION.md) | Scope, users, use cases, non-goals |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, data model, execution, validation |
| [docs/API_DESIGN.md](docs/API_DESIGN.md) | Public API, examples, custom components, contracts, YAML |
| [docs/BUILTIN_COMPONENTS.md](docs/BUILTIN_COMPONENTS.md) | Initial OpenCV / HF / QC / IO catalog |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Milestones, acceptance criteria, testing, CI |
| [docs/BENCHMARK_PLAN.md](docs/BENCHMARK_PLAN.md) | Baselines and methodology |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decisions, alternatives, assumptions |
| [docs/RISKS_AND_OPEN_QUESTIONS.md](docs/RISKS_AND_OPEN_QUESTIONS.md) | Risks and open questions |

## License

Apache-2.0. See [LICENSE](LICENSE).
