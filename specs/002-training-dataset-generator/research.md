# Research: Training Dataset Generator

## Phase 0 Findings

### 1. Existing metricate.evaluate() Behavior

**Research Task**: Understand return structure of `metricate.evaluate()`

**Finding**: Returns `EvaluationResult` dataclass with:
```python
@dataclass
class EvaluationResult:
    metrics: list[MetricValue]  # Each has .metric (name) and .value (float|None)
    metadata: dict              # {"n_samples": int, "n_clusters": int, "n_features": int}
    warnings: list[str]
```

**Integration**: 
- Extract metric values via `[m.value for m in result.metrics]`
- Get metric names via `[m.metric for m in result.metrics]`
- Access cluster count via `result.metadata["n_clusters"]`

---

### 2. Existing metricate.degrade() Behavior

**Research Task**: Understand how degradation generates files

**Finding**: Returns `DegradationResult` with:
```python
@dataclass
class DegradationResult:
    output_dir: str
    degradations: list[DegradationEntry]  # Metadata per degradation
    csv_files: list[str]                  # Paths to generated CSVs
    manifest_path: str
```

Each `DegradationEntry` has:
- `type`: Degradation type name
- `level`: e.g., "10pct"
- `filepath`: Path to generated CSV

**Integration**:
- Call `metricate.degrade()` once per input file
- Iterate `result.csv_files` to get paths for evaluation
- Match degradation metadata via `result.degradations`

---

### 3. External Metrics to Exclude

**Research Task**: Identify metrics requiring ground truth

**Finding**: 4 external metrics require ground truth labels:
- `ARI` (Adjusted Rand Index)
- `Van Dongen`
- `VI` (Variation of Information)
- `Omega`

**Decision**: Always exclude these in training data generation since we're degrading clusterings without external ground truth.

**Implementation**: Pass `exclude=["ARI", "Van Dongen", "VI", "Omega"]` to `metricate.evaluate()`

---

### 4. Quality Score Mapping

**Research Task**: Define mapping from degradation level to quality_score

**Decision**: Use linear degradation from 1.0:

| Level | quality_score | Rationale |
|-------|---------------|-----------|
| Original | 1.00 | Perfect quality |
| 5pct | 0.95 | 5% degradation |
| 10pct | 0.90 | 10% degradation |
| 25pct | 0.75 | 25% degradation |
| 50pct | 0.50 | 50% degradation |

**Implementation**:
```python
LEVEL_TO_SCORE = {
    None: 1.00,      # Original
    "5pct": 0.95,
    "10pct": 0.90,
    "25pct": 0.75,
    "50pct": 0.50,
}
```

---

### 5. Percentile Rank Normalization

**Research Task**: Best approach for normalizing unbounded metrics

**Options Evaluated**:
| Method | Pros | Cons |
|--------|------|------|
| Min-max | Simple | Sensitive to outliers |
| Z-score | Statistical | Unbounded output |
| Sigmoid | Deterministic | Requires hyperparameters |
| **Percentile rank** | Robust, no params | Dataset-dependent |

**Decision**: Percentile rank
- Produces uniform distribution in [0, 1]
- No hyperparameters to tune
- Robust to outliers
- Works well with ML models

**Implementation**:
```python
from scipy.stats import rankdata

def percentile_normalize(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Convert to percentile ranks in [0, 1]."""
    ranks = rankdata(values, method='average', nan_policy='omit')
    normalized = (ranks - 1) / (len(values) - 1)
    if not higher_is_better:
        normalized = 1.0 - normalized
    return normalized
```

---

### 6. Dataset Size Estimates

**Research Task**: Estimate output dataset size

**Calculation** (per input file):
- 1 original clustering
- 19 degradation types × 4 levels = 76 degraded versions
- **Total: 77 rows per input file**

**Column count**:
- 8 metadata columns
- 30 raw metric columns
- 30 normalized metric columns
- **Total: ~68 columns**

**For 10 input files**: 770 rows × 68 columns = ~52,360 cells

---

### 7. Metric Column Names

**Research Task**: Standardize column naming

**Decision**: Use exact metric names from `METRIC_REFERENCE`, adding `_norm` suffix for normalized:

```python
# Raw columns
"Silhouette", "Davies-Bouldin", "Calinski-Harabasz", ...

# Normalized columns
"Silhouette_norm", "Davies-Bouldin_norm", "Calinski-Harabasz_norm", ...
```

**Note**: Hyphens preserved in column names (pandas handles this fine).

---

## Alternatives Considered

### Alternative 1: Generate Degradations In-Memory

**Rejected**: The `metricate.degrade()` function already handles degradation correctly and saves CSVs. Reimplementing would duplicate code and risk inconsistencies.

### Alternative 2: Store Normalized Values Only

**Rejected**: Raw values are useful for debugging and understanding actual metric behavior. Storage overhead is minimal.

### Alternative 3: Multi-Class Quality Labels

**Rejected**: Binary classification is simpler for initial model training. `quality_score` provides continuous signal if needed.

---

## Dependencies Confirmed

| Package | Version | Usage |
|---------|---------|-------|
| pandas | ≥1.0 | DataFrame operations |
| numpy | ≥1.20 | Percentile calculations |
| scipy | ≥1.7 | `rankdata` for percentile rank |
| pyarrow | ≥8.0 | Parquet export (optional) |

All are already installed in the project environment except potentially pyarrow.
