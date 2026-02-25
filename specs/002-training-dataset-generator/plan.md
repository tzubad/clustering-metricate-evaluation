# Implementation Plan: Training Dataset Generator

**Feature**: `002-training-dataset-generator`  
**Spec**: [spec.md](spec.md)  
**Status**: Planning

---

## Technical Context

### Existing Dependencies (RESOLVED)

| Component | Location | Usage |
|-----------|----------|-------|
| `metricate.evaluate()` | `metricate/core/evaluator.py` | Calculate 30 internal metrics |
| `metricate.degrade()` | `metricate/degradation/toolkit.py` | Generate degraded CSVs |
| `metricate.core.loader` | `metricate/core/loader.py` | Load clustering CSVs |
| `METRIC_REFERENCE` | `metricate/core/reference.py` | Metric metadata (ranges, directions) |
| `EvaluationResult` | `metricate/output/report.py` | Metric result container |

### External Dependencies

| Package | Purpose | Status |
|---------|---------|--------|
| `pandas` | DataFrame manipulation | ✅ Already installed |
| `numpy` | Percentile rank calculation | ✅ Already installed |
| `pyarrow` | Parquet export | ⚠️ Optional, may need install |

### Key Design Decisions

1. **Reuse existing modules**: Don't reimplement degradation or evaluation
2. **Percentile normalization**: Apply after all records collected (requires full dataset)
3. **Save degraded CSVs**: Use `metricate.degrade()` which already saves files
4. **Exclude external metrics**: ARI, Van Dongen, VI, Omega require ground truth

---

## Constitution Check

No constitution file found. Proceeding without project-level constraints.

---

## Phase 0: Research (COMPLETE)

### Findings

1. **`metricate.evaluate()` returns `EvaluationResult`** with:
   - `metrics: list[MetricValue]` where each has `.metric` (name) and `.value` (float|None)
   - `metadata: dict` with `n_samples`, `n_clusters`, `n_features`
   
2. **`metricate.degrade()` returns `DegradationResult`** with:
   - `csv_files: list[str]` - paths to generated degraded CSVs
   - `degradations: list[DegradationEntry]` - metadata per degradation
   
3. **Degradation levels map to quality_score**:
   - `5pct` → 0.95
   - `10pct` → 0.90
   - `25pct` → 0.75
   - `50pct` → 0.50

4. **External metrics to exclude**: `["ARI", "Van Dongen", "VI", "Omega"]`

---

## Phase 1: Data Model & Result Container

### Task 1.1: Create TrainingDataResult class
**File**: `metricate/training/result.py`

```python
@dataclass
class TrainingDataResult:
    records: list[dict]
    metadata: dict
    warnings: list[str]
    errors: list[str]
    
    def to_dataframe(self) -> pd.DataFrame
    def to_csv(self, path: str | Path) -> None
    def to_parquet(self, path: str | Path) -> None
    def summary(self) -> str
```

### Task 1.2: Implement normalization logic
**File**: `metricate/training/normalize.py`

- Compute percentile rank for each metric column
- Handle NaN values (exclude from ranking, keep as NaN in output)
- Account for metric direction (higher-is-better vs lower-is-better)

---

## Phase 2: Core Generation Logic

### Task 2.1: Implement single-file generator
**File**: `metricate/training/generator.py`

```python
def generate_training_data(
    csv_path: str | Path,
    output_dir: str | Path,
    types: list[str] | None = None,
    levels: list[str] | None = None,
    exclude: list[str] | None = None,
    force_all: bool = False,
    topic: str | None = None,
    random_seed: int = 42,
) -> TrainingDataResult
```

**Logic**:
1. Load original CSV
2. Evaluate original → create record (quality=1, quality_score=1.0)
3. Call `metricate.degrade()` to generate degraded CSVs
4. For each degraded CSV:
   - Evaluate → create record (quality=0, quality_score=level_score)
5. After all records collected: apply percentile normalization
6. Return TrainingDataResult

### Task 2.2: Implement batch generator
**File**: `metricate/training/generator.py`

```python
def generate_training_data_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    topic_mapping: dict[str, str] | None = None,
    **kwargs,
) -> TrainingDataResult
```

**Logic**:
1. List all CSVs in input_dir
2. For each CSV: call single-file generator, collect records
3. Combine all records
4. Apply percentile normalization across entire dataset
5. Return combined TrainingDataResult

---

## Phase 3: API Integration

### Task 3.1: Export from metricate package
**File**: `metricate/__init__.py`

- Add `generate_training_data` and `generate_training_data_batch` to public API
- Update `__all__` list

### Task 3.2: Create training module __init__
**File**: `metricate/training/__init__.py`

- Export public functions

---

## Phase 4: CLI Support (P3 - Optional)

### Task 4.1: Add train-data command
**File**: `metricate/cli/main.py`

```bash
metricate train-data <input> --output <output.csv> [--types ...] [--levels ...]
```

---

## File Changes Summary

```
metricate/
├── __init__.py                    # MODIFY: Add exports
├── training/                      # NEW FOLDER
│   ├── __init__.py               # NEW: Module exports
│   ├── generator.py              # NEW: Core generation logic
│   ├── result.py                 # NEW: TrainingDataResult class
│   └── normalize.py              # NEW: Percentile normalization
└── cli/
    └── main.py                   # MODIFY: Add train-data command (optional)
```

---

## Estimated Effort

| Phase | Description | Est. Hours |
|-------|-------------|------------|
| Phase 1 | Data Model & Result | 2-3 |
| Phase 2 | Core Generation | 4-5 |
| Phase 3 | API Integration | 1 |
| Phase 4 | CLI (optional) | 1-2 |
| **Total** | | **8-11** |

---

## Acceptance Checklist

- [ ] Single CSV generates 77 rows (1 original + 19 types × 4 levels)
- [ ] Batch processing combines multiple CSVs correctly
- [ ] `quality` column is binary (0/1)
- [ ] `quality_score` column has correct values (1.0, 0.95, 0.90, 0.75, 0.50)
- [ ] All raw metric columns present
- [ ] All `_norm` columns present with values in [0, 1]
- [ ] Degraded CSVs saved to output_dir
- [ ] `to_csv()` and `to_parquet()` work correctly
- [ ] Random seed produces reproducible results
