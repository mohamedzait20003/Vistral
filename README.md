# Vistral

**Status: planning.** There is no implementation yet.

Vistral is a planned Python library for building **validated, inspectable computer-vision pipelines** from:
- OpenCV operations,
- Hugging Face models (using their native processors and post-processing),
- Supervision results and annotators,
- your own components.

Its main focus is **industrial and production inspection and QC**.

```python
# Target API (design only — not implemented)
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
