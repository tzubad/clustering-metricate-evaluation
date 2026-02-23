# Implementation Plan: Metricate

**Branch**: `001-metricate-product` | **Date**: 2026-02-23 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-metricate-product/spec.md`

## Summary

Metricate is a clustering evaluation toolkit that calculates 34 quality metrics on clustering results (CSV with cluster labels + embeddings), compares two clusterings to determine a winner, and generates degraded datasets with HTML visualizations. The implementation extracts and refactors existing code from `clustering_metrics_evaluation.ipynb` and `degradation_toolkit.py`.

## Technical Context

**Language/Version**: Python 3.10+ (match existing .venv)  
**Primary Dependencies**: pandas, numpy, scikit-learn, plotly (existing), click (new for CLI), flask (new for web UI P4)  
**Storage**: File-based (CSV input/output, HTML visualization output)  
**Testing**: pytest with fixtures using existing degraded_datasets/ as test data  
**Target Platform**: macOS/Linux CLI, Python module, simple Flask web UI (P4)  
**Project Type**: Single Python package  
**Performance Goals**: <30s for 10k rows (SC-001), auto-skip O(n²) metrics >50k rows  
**Constraints**: Memory proportional to dataset size; pairwise distance matrix O(n²) is the bottleneck  
**Scale/Scope**: 10k-100k row datasets typical; 34 metrics; 19 degradation types × 4 severity levels

## Constitution Check

*No constitution.md file found - no gate violations to check.*

## Project Structure

### Documentation (this feature)

```text
specs/001-metricate-product/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (API contracts)
│   └── api.md
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
metricate/
├── __init__.py          # Package entry, exports evaluate(), compare(), degrade()
├── core/
│   ├── __init__.py
│   ├── metrics.py       # 34 metric implementations (extracted from notebook)
│   ├── loader.py        # CSV loading, embedding parsing, validation (FR-001, FR-004)
│   ├── reference.py     # Metric ranges, directions, O(n²) flags
│   └── evaluator.py     # Main evaluate() function (FR-002, FR-003)
├── comparison/
│   ├── __init__.py
│   └── compare.py       # Two-clustering comparison logic (FR-008-013)
├── degradation/
│   ├── __init__.py
│   ├── toolkit.py       # Refactored from degradation_toolkit.py (FR-014-019)
│   └── visualize.py     # HTML visualization generation (FR-016-018)
├── cli/
│   ├── __init__.py
│   └── main.py          # Click-based CLI (FR-022-024)
├── web/                 # P4 - lowest priority
│   ├── __init__.py
│   ├── app.py           # Flask app (FR-025-027)
│   └── templates/
│       └── index.html
└── output/
    ├── __init__.py
    ├── formatters.py    # Table, JSON, CSV output (FR-021a, FR-024)
    └── report.py        # Metric result with ranges/directions

tests/
├── conftest.py          # Fixtures using existing test data
├── test_metrics.py      # Unit tests for each metric
├── test_loader.py       # CSV parsing tests
├── test_evaluator.py    # Integration tests
├── test_comparison.py   # Comparison mode tests
├── test_cli.py          # CLI smoke tests
└── test_data/           # Symlink to existing degraded_datasets/
```

**Structure Decision**: Single Python package with submodules for separation of concerns. CLI and Web UI are optional entry points over the core module API.

## Complexity Tracking

> No Constitution violations to justify.

## Phase 0: Research ✅ COMPLETE

**Output**: [research.md](./research.md)

### Unknowns Resolved

| Unknown | Resolution | Source |
|---------|------------|--------|
| Metric implementations | Extract from notebook cells 5, 44 (Part 6) | clustering_metrics_evaluation.ipynb |
| Helper functions needed | `compute_cluster_stats`, `compute_scatter_matrices`, `compute_concordance_pairs`, `pairwise_distances` | notebook Part 1 |
| Existing degradation code | `ClusteringDegrader` class (708 lines) ready to import | degradation_toolkit.py |
| O(n²) metrics list | Gamma, Tau, Point-Biserial, G-plus, McClain-Rao, NIVA | notebook analysis |
| Visualization approach | Plotly for interactive HTML (existing pattern) | existing visualizations |

### Technology Decisions

| Decision | Rationale | Alternatives Rejected |
|----------|-----------|----------------------|
| Click for CLI | Lightweight, Pythonic, decorator-based | argparse (verbose), typer (extra dep) |
| Flask for web UI (P4) | Minimal, fits simple upload use case | FastAPI (overkill), Streamlit (different paradigm) |
| pandas DataFrame output | Matches existing notebook workflow | dict (less ergonomic), custom class (overhead) |
| Plotly HTML | Existing pattern in degradation visualizations | matplotlib (not interactive) |

### Code Reuse Map

| Existing Code | Target Location | Modifications |
|--------------|-----------------|---------------|
| `calculate_all_metrics()` (notebook) | `metricate/core/evaluator.py` | Add exclusion list, auto-skip logic |
| 34 metric functions (notebook Part 6) | `metricate/core/metrics.py` | Extract as standalone functions |
| `compute_*` helpers (notebook Part 1) | `metricate/core/metrics.py` | Extract, add type hints |
| `ClusteringDegrader` (degradation_toolkit.py) | `metricate/degradation/toolkit.py` | Import directly, thin wrapper |
| `ALL_METRIC_DIRECTIONS` dict | `metricate/core/reference.py` | Add ranges, O(n²) flags |
| Plotly visualization code (notebook) | `metricate/degradation/visualize.py` | Parameterize for CLI/API use |

## Phase 1: Design ✅ COMPLETE

### Output: data-model.md ✅

See [data-model.md](./data-model.md) - Defines Clustering, MetricResult, ComparisonReport, DegradationSuite entities with validation rules and state transitions.

### Output: contracts/api.md ✅

See [contracts/api.md](./contracts/api.md) - Complete API specification for Python module, CLI, and Web UI interfaces.

### Output: quickstart.md ✅

See [quickstart.md](./quickstart.md) - Installation, usage examples, and troubleshooting guide.

### Agent Context Updated ✅

Updated `.github/agents/copilot-instructions.md` with project technologies.

---

## Implementation Phases Summary

| Phase | Priority | Deliverables | Dependencies |
|-------|----------|--------------|--------------|
| 1 | P1 | Core module: evaluate() API, 34 metrics, DataFrame output | None |
| 2 | P1 | CSV loader with validation, embedding parsing | Phase 1 |
| 3 | P2 | Metric exclusion, auto-skip O(n²) | Phase 2 |
| 4 | P2 | CLI with evaluate command | Phase 3 |
| 5 | P2 | Comparison mode (compare() API + CLI) | Phase 4 |
| 6 | P3 | Degradation module (degrade() API + CLI) | Phase 2 |
| 7 | P4 | Web UI | Phase 5 |

## Files to Create/Modify

### New Files
- `metricate/__init__.py`
- `metricate/core/__init__.py`
- `metricate/core/metrics.py` (extract ~400 lines from notebook)
- `metricate/core/loader.py`
- `metricate/core/reference.py`
- `metricate/core/evaluator.py`
- `metricate/comparison/__init__.py`
- `metricate/comparison/compare.py`
- `metricate/degradation/__init__.py`
- `metricate/degradation/toolkit.py`
- `metricate/degradation/visualize.py`
- `metricate/cli/__init__.py`
- `metricate/cli/main.py`
- `metricate/output/__init__.py`
- `metricate/output/formatters.py`
- `metricate/output/report.py`
- `pyproject.toml` or `setup.py`
- `README.md` (with 50k threshold documentation per FR-004d)
- `tests/conftest.py`
- `tests/test_*.py` (6 test files)

### Files to Reuse (import/refactor)
- `degradation_toolkit.py` → import into `metricate/degradation/toolkit.py`
- Metric functions from notebook → extract to `metricate/core/metrics.py`
