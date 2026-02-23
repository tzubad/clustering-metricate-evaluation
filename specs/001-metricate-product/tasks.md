# Tasks: Metricate

**Input**: Design documents from `/specs/001-metricate-product/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/api.md ✅

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4, US5)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and package structure

- [X] T001 Create metricate package structure with all __init__.py files per plan.md
- [X] T002 Create pyproject.toml with dependencies: pandas>=2.0, numpy>=1.24, scikit-learn>=1.3, plotly>=5.0, click>=8.0
- [X] T003 [P] Create README.md with installation instructions and 50k threshold documentation (FR-004d)
- [X] T004 [P] Setup pytest configuration in pyproject.toml or pytest.ini
- [X] T005 [P] Create tests/conftest.py with fixtures pointing to existing degraded_datasets/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 Create METRIC_REFERENCE dict with all 34 metrics (ranges, directions, tiers, complexity, skip_large flags) in metricate/core/reference.py
- [X] T007 [P] Extract compute_cluster_stats() helper from notebook to metricate/core/metrics.py
- [X] T008 [P] Extract compute_scatter_matrices() helper from notebook to metricate/core/metrics.py
- [X] T009 [P] Extract compute_concordance_pairs() helper from notebook to metricate/core/metrics.py
- [X] T010 Extract all 34 metric functions from notebook Part 6 to metricate/core/metrics.py
- [X] T011 Create MetricValue and EvaluationResult dataclasses in metricate/output/report.py
- [X] T012 [P] Create CSV loader with validation (V-001 through V-008) in metricate/core/loader.py
- [X] T013 [P] Create column auto-detection (detect_columns function) in metricate/core/loader.py
- [X] T014 Create MetricateError exception hierarchy in metricate/core/exceptions.py

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Single Clustering Evaluation (Priority: P1) 🎯 MVP

**Goal**: Calculate all 34 clustering metrics for a single CSV and return results with ranges/directions

**Independent Test**: `metricate.evaluate("clustering.csv")` returns EvaluationResult with all metrics, ranges, directions

### Implementation for User Story 1

- [X] T015 [US1] Implement calculate_all_metrics() orchestrator with precomputation in metricate/core/evaluator.py
- [X] T016 [US1] Implement evaluate() public API function in metricate/core/evaluator.py
- [X] T017 [P] [US1] Create table formatter (to_table method) in metricate/output/formatters.py
- [X] T018 [P] [US1] Create DataFrame formatter (to_dataframe method) in metricate/output/formatters.py
- [X] T019 [US1] Export evaluate() from metricate/__init__.py
- [X] T020 [US1] Handle noise points (cluster_id=-1) exclusion in evaluator

**Checkpoint**: User Story 1 complete - `metricate.evaluate()` works independently

---

## Phase 4: User Story 2 - Metric Exclusion (Priority: P2)

**Goal**: Allow users to exclude specific metrics from calculation

**Independent Test**: `metricate.evaluate("clustering.csv", exclude=["Gamma", "Tau"])` skips those metrics

### Implementation for User Story 2

- [X] T021 [US2] Add exclude parameter handling in evaluate() in metricate/core/evaluator.py
- [X] T022 [US2] Implement O(n²) auto-skip logic for datasets >50k rows (FR-004a) in metricate/core/evaluator.py
- [X] T023 [US2] Add force_all parameter to override auto-skip (FR-004c) in metricate/core/evaluator.py
- [X] T024 [US2] Add skip_reason field population when metrics are excluded in metricate/output/report.py
- [X] T025 [US2] Warn about unrecognized metric names in exclusion list (FR-007) in metricate/core/evaluator.py

**Checkpoint**: User Story 2 complete - metric exclusion and auto-skip work

---

## Phase 5: User Story 3 - Two Clustering Comparison (Priority: P2)

**Goal**: Compare two clusterings and determine overall winner by metric count

**Independent Test**: `metricate.compare("v1.csv", "v2.csv")` returns ComparisonResult with winner

### Implementation for User Story 3

- [X] T026 [US3] Create ComparisonResult dataclass in metricate/output/report.py
- [X] T027 [US3] Implement determine_winner() function for per-metric comparison in metricate/comparison/compare.py
- [X] T028 [US3] Implement compare() public API function in metricate/comparison/compare.py
- [X] T029 [US3] Add row count validation with warning (FR-009, FR-009a) in metricate/comparison/compare.py
- [X] T030 [US3] Add dimension mismatch validation in metricate/comparison/compare.py
- [X] T031 [P] [US3] Create comparison table formatter (to_table method) in metricate/output/formatters.py
- [X] T032 [US3] Export compare() from metricate/__init__.py

**Checkpoint**: User Story 3 complete - comparison mode works independently

---

## Phase 6: User Story 2+3 - CLI Interface (Priority: P2)

**Goal**: Provide CLI commands for evaluate and compare

**Independent Test**: `metricate evaluate clustering.csv --format json` outputs JSON

### Implementation for CLI

- [X] T033 Create Click CLI application entry point in metricate/cli/main.py
- [X] T034 [P] Create JSON output formatter in metricate/output/formatters.py
- [X] T035 [P] Create CSV output formatter in metricate/output/formatters.py
- [X] T036 Implement `metricate evaluate` command with --exclude, --force-all, --format, -o options in metricate/cli/main.py
- [X] T037 Implement `metricate compare` command with same options in metricate/cli/main.py
- [X] T038 [P] Implement `metricate list metrics` and `metricate list degradations` commands in metricate/cli/main.py
- [X] T039 Add CLI entry point to pyproject.toml [project.scripts]

**Checkpoint**: CLI works for evaluate and compare commands

---

## Phase 7: User Story 4 - Degradation Generation Module (Priority: P3)

**Goal**: Generate degraded datasets with HTML visualizations

**Independent Test**: `metricate.degrade("clustering.csv", "./output/")` creates 76 CSVs + HTMLs

### Implementation for User Story 4

- [X] T040 [US4] Create DegradationConfig and DegradationResult dataclasses in metricate/degradation/toolkit.py
- [X] T041 [US4] Import and wrap ClusteringDegrader from degradation_toolkit.py in metricate/degradation/toolkit.py
- [X] T042 [US4] Implement degrade() public API function in metricate/degradation/toolkit.py
- [X] T043 [US4] Generate manifest.json with complete metadata (FR-017) in metricate/degradation/toolkit.py
- [X] T044 [P] [US4] Create Plotly visualization generator (per-degradation HTMLs) in metricate/degradation/visualize.py
- [X] T045 [P] [US4] Create index.html generator linking all visualizations (FR-018) in metricate/degradation/visualize.py
- [X] T046 [US4] Implement `metricate degrade` CLI command with --levels, --types, --no-visualize in metricate/cli/main.py
- [X] T047 [US4] Export degrade() from metricate/__init__.py

**Checkpoint**: User Story 4 complete - degradation module generates CSVs and HTMLs

---

## Phase 8: User Story 5 - Web UI (Priority: P4)

**Goal**: Simple browser-based evaluation interface

**Independent Test**: Start Flask server, upload CSV via browser, see formatted results

### Implementation for User Story 5

- [X] T048 [US5] Create Flask application in metricate/web/app.py
- [X] T049 [P] [US5] Create HTML template with file upload form in metricate/web/templates/index.html
- [X] T050 [US5] Implement POST /api/evaluate endpoint in metricate/web/app.py
- [X] T051 [US5] Implement POST /api/compare endpoint in metricate/web/app.py
- [X] T052 [US5] Add formatted table display in web UI template
- [X] T053 [US5] Add `metricate web` CLI command to start server in metricate/cli/main.py

**Checkpoint**: User Story 5 complete - web UI accessible via browser

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, cleanup, and validation

- [X] T054 [P] Add docstrings to all public functions following Google style
- [X] T055 [P] Update README.md with complete usage examples from quickstart.md
- [X] T056 Run quickstart.md validation - test all documented examples work
- [X] T057 [P] Add type hints to all public API functions
- [X] T058 Create simple unit tests for core metrics in tests/test_metrics.py
- [X] T059 Create integration tests using existing degraded_datasets/ in tests/test_evaluator.py

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational - Core MVP
- **US2 (Phase 4)**: Depends on US1 (extends evaluate())
- **US3 (Phase 5)**: Depends on US1 (reuses evaluate())
- **CLI (Phase 6)**: Depends on US1, US2, US3
- **US4 (Phase 7)**: Depends on Foundational only - can parallel with US2/US3
- **US5 (Phase 8)**: Depends on US1, US3
- **Polish (Phase 9)**: Depends on all desired stories

### User Story Dependencies

```
         ┌─────────────────────────────────────────────────────────────┐
         │                  Phase 2: Foundational                      │
         └─────────────────────────────────────────────────────────────┘
                   │                    │                    │
                   ▼                    │                    ▼
         ┌─────────────────┐            │           ┌─────────────────┐
         │ US1: Evaluate   │◄───────────┘           │ US4: Degrade    │
         │ (P1) 🎯 MVP     │                        │ (P3)            │
         └─────────────────┘                        └─────────────────┘
                   │
         ┌────────┴────────┐
         ▼                 ▼
┌─────────────────┐ ┌─────────────────┐
│ US2: Exclusion  │ │ US3: Compare    │
│ (P2)            │ │ (P2)            │
└─────────────────┘ └─────────────────┘
         │                 │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │ CLI (P2)        │
         └─────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ US5: Web UI     │
         │ (P4)            │
         └─────────────────┘
```

### Parallel Opportunities

**Within Setup**:
```bash
# Run T003, T004, T005 in parallel
```

**Within Foundational**:
```bash
# Run T007, T008, T009 in parallel (helper functions)
# Run T012, T013, T014 in parallel (loader, detection, exceptions)
```

**Within User Story 1**:
```bash
# Run T017, T018 in parallel (formatters)
```

**Across Stories (after Foundational)**:
```bash
# US4 can run in parallel with US2, US3, CLI
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: `metricate.evaluate("clustering.csv")` works
5. Deploy/demo if ready - core product is functional

### Incremental Delivery

| Increment | Stories | Value Delivered |
|-----------|---------|-----------------|
| MVP | US1 | Single clustering evaluation with 34 metrics |
| v1.1 | US1 + US2 | Add metric exclusion and auto-skip |
| v1.2 | US1 + US2 + US3 | Add comparison mode |
| v1.3 | + CLI | Command-line interface |
| v1.4 | + US4 | Degradation generation |
| v2.0 | + US5 | Web UI |

### Suggested MVP Scope

**Complete US1 only** - this delivers:
- ✅ 34 clustering metrics calculated
- ✅ Ranges and directions displayed
- ✅ Table/DataFrame output
- ✅ CSV validation and error handling

This is sufficient for data scientists to evaluate clustering quality programmatically.

---

## Notes

- All paths are relative to repository root
- Extract code from `clustering_metrics_evaluation.ipynb` cells and `degradation_toolkit.py`
- Existing test data in `degraded_datasets/` and `degraded_datasets_17clusters/`
- Total tasks: 59
- Commit after each task or logical group
- Verify each story checkpoint before proceeding
