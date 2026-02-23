# Data Model: Metricate

**Feature**: 001-metricate-product  
**Version**: 1.0  
**Status**: Draft

## Entity Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│   Clustering    │────▶│   MetricResult   │────▶│ ComparisonReport  │
│   (Input)       │     │   (Output)       │     │    (Compare)      │
└─────────────────┘     └──────────────────┘     └───────────────────┘
        │
        ▼
┌─────────────────┐     ┌──────────────────┐
│DegradationSuite │────▶│ DegradedDataset  │
│   (Generate)    │     │   (Output)       │
└─────────────────┘     └──────────────────┘
```

---

## 1. Clustering (Input Schema)

### DataFrame Schema

| Column | Type | Required | Constraints | Description |
|--------|------|----------|-------------|-------------|
| `cluster_id` | int64 | ✓ | ≥ -1 | Cluster assignment (-1 for noise) |
| `embedding_*` | float64 | ✓ (1+) | numeric | Embedding dimensions |

### Validation Rules

| Rule ID | Rule | Error Type |
|---------|------|------------|
| V-001 | File must exist and be readable | FileNotFoundError |
| V-002 | File must be valid CSV format | ParserError |
| V-003 | `cluster_id` column must exist | ColumnNotFoundError |
| V-004 | At least 1 `embedding_*` column | ColumnNotFoundError |
| V-005 | `cluster_id` must be numeric | TypeError |
| V-006 | Embedding columns must be numeric | TypeError |
| V-007 | At least 2 unique clusters (excluding -1) | InsufficientClustersError |
| V-008 | No NaN in embedding columns | ValueError |

### Auto-Detection

```python
def detect_columns(df: pd.DataFrame) -> tuple[str, list[str]]:
    """Auto-detect label and embedding columns."""
    # Label column priority: cluster_id > label > cluster > first int column
    # Embedding columns: all float columns or columns matching 'embedding_*', 'dim_*', 'feature_*'
```

---

## 2. MetricResult (Output Schema)

### DataFrame Schema

| Column | Type | Description |
|--------|------|-------------|
| `metric` | str | Metric name (e.g., "Silhouette") |
| `value` | float64 | Calculated metric value |
| `range` | str | Valid range (e.g., "[-1, 1]") |
| `direction` | str | "↑ Higher is better" or "↓ Lower is better" |
| `tier` | str | "Original", "Tier 1", "Tier 2", "Tier 3", "External" |
| `computed` | bool | True if computed, False if skipped |
| `skip_reason` | str | Reason if skipped (e.g., "Dataset >50k rows") |

### Python Dataclass

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class MetricValue:
    """Single metric result."""
    metric: str
    value: Optional[float]  # None if skipped
    range: str
    direction: str
    tier: str
    computed: bool = True
    skip_reason: Optional[str] = None

@dataclass
class EvaluationResult:
    """Complete evaluation output."""
    metrics: list[MetricValue]
    metadata: dict  # n_points, n_clusters, n_dimensions, computation_time
    warnings: list[str]  # e.g., "6 O(n²) metrics skipped for dataset >50k rows"
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame format."""
        
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        
    def to_table(self) -> str:
        """Render as formatted table (default CLI output)."""
```

---

## 3. ComparisonReport (Compare Output)

### DataFrame Schema

| Column | Type | Description |
|--------|------|-------------|
| `metric` | str | Metric name |
| `value_a` | float64 | Value for clustering A |
| `value_b` | float64 | Value for clustering B |
| `winner` | str | "A", "B", or "tie" |
| `direction` | str | Used to determine winner |
| `range` | str | Valid range for reference |

### Python Dataclass

```python
@dataclass
class ComparisonResult:
    """Comparison of two clusterings."""
    results: pd.DataFrame  # Per-metric comparison
    summary: dict  # wins_a, wins_b, ties, overall_winner
    clustering_a_meta: dict  # n_points, n_clusters, path
    clustering_b_meta: dict
    warnings: list[str]  # e.g., "Row counts differ: A=1000, B=950"
    
    def get_overall_winner(self) -> str:
        """Return 'A', 'B', or 'tie' based on metric wins."""
        
    def to_table(self) -> str:
        """Render comparison table with winner indicators."""
```

### Winner Determination Logic

```python
def determine_winner(value_a: float, value_b: float, direction: str) -> str:
    """
    Determine winner based on metric direction.
    
    Args:
        value_a: Metric value for clustering A
        value_b: Metric value for clustering B
        direction: "higher" or "lower"
    
    Returns:
        "A", "B", or "tie"
    """
    if np.isnan(value_a) or np.isnan(value_b):
        return "N/A"
    
    if abs(value_a - value_b) < 1e-10:
        return "tie"
    
    if direction == "higher":
        return "A" if value_a > value_b else "B"
    else:  # direction == "lower"
        return "A" if value_a < value_b else "B"
```

---

## 4. DegradationSuite (Degradation Config)

### Configuration Schema

```python
@dataclass
class DegradationConfig:
    """Configuration for degradation generation."""
    levels: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.25, 0.50])
    degradation_types: list[str] = field(default_factory=lambda: [
        'random_removal', 'core_removal', 'boundary_reassignment',
        'label_swap_random', 'label_swap_neighboring', 'label_swap_distant',
        'noise_injection', 'embedding_perturbation', 'centroid_displacement',
        'merge_nearest', 'merge_farthest', 'merge_random',
        'split_largest', 'split_loosest', 'split_random',
        'remove_smallest_clusters', 'remove_largest_clusters', 'remove_tightest_clusters'
    ])
    generate_visualizations: bool = True
    output_format: str = 'csv'  # csv | parquet
```

### Available Degradation Types

| Category | Type | Description |
|----------|------|-------------|
| Point Removal | `random_removal` | Remove random points |
| | `core_removal` | Remove points closest to centroids |
| Label Modification | `boundary_reassignment` | Reassign boundary points to neighbors |
| | `label_swap_random` | Swap random point labels |
| | `label_swap_neighboring` | Swap labels between adjacent clusters |
| | `label_swap_distant` | Swap labels between far clusters |
| Embedding Noise | `noise_injection` | Add Gaussian noise to embeddings |
| | `embedding_perturbation` | Perturb embedding values |
| | `centroid_displacement` | Shift points toward other centroids |
| Cluster Merge | `merge_nearest` | Merge closest clusters |
| | `merge_farthest` | Merge most distant clusters |
| | `merge_random` | Merge random clusters |
| Cluster Split | `split_largest` | Split largest cluster |
| | `split_loosest` | Split most dispersed cluster |
| | `split_random` | Split random cluster |
| Cluster Remove | `remove_smallest_clusters` | Remove smallest clusters |
| | `remove_largest_clusters` | Remove largest clusters |
| | `remove_tightest_clusters` | Remove most compact clusters |

---

## 5. DegradedDataset (Degradation Output)

### Output Directory Structure

```
<output_dir>/
├── manifest.json           # Metadata for all generated files
├── <type>_<level>.csv     # Degraded datasets
├── visualizations/
│   ├── index.html         # Overview page
│   └── <type>.html        # Per-type interactive plot
```

### Manifest Schema

```json
{
  "source_file": "clustering.csv",
  "timestamp": "2024-01-15T10:30:00Z",
  "original_stats": {
    "n_points": 10000,
    "n_clusters": 17,
    "n_dimensions": 50
  },
  "degradations": [
    {
      "file": "random_removal_10pct.csv",
      "type": "random_removal",
      "level": 0.10,
      "n_points_after": 9000,
      "n_clusters_after": 17
    }
  ],
  "visualizations_generated": true
}
```

### Python Dataclass

```python
@dataclass
class DegradationResult:
    """Result of degradation suite generation."""
    output_dir: Path
    manifest: dict
    files_generated: list[str]
    visualizations_generated: list[str]
    errors: list[str]  # Any degradations that failed
```

---

## State Transitions

### Evaluation Flow

```
[CSV File] 
    │ validate()
    ▼
[Clustering] 
    │ detect_columns()
    ▼
[Validated Clustering]
    │ evaluate()
    ▼
[MetricResult]
    │ format()
    ▼
[Output: table/json/csv]
```

### Comparison Flow

```
[CSV A] ─────┬───────────────────┐
             │ validate_pair()   │
[CSV B] ─────┘                   ▼
                         [Two Clusterings]
                                 │ compare()
                                 ▼
                         [ComparisonReport]
                                 │ format()
                                 ▼
                         [Output: table/json/csv]
```

### Degradation Flow

```
[Source CSV]
    │ validate()
    ▼
[Clustering]
    │ DegradationConfig
    ▼
[DegradationSuite]
    │ generate()
    ▼
[DegradedDataset × N]
    │ visualize()
    ▼
[HTML Visualizations]
```

---

## Validation Summary

| Entity | Key Validations |
|--------|-----------------|
| Clustering | File exists, valid CSV, required columns, numeric types, ≥2 clusters |
| MetricResult | All 34 metrics present (computed or skipped), valid ranges |
| ComparisonReport | Both clusterings valid, same embedding dimensions |
| DegradationConfig | Valid degradation types, levels in (0, 1) |
| DegradedDataset | All files written, manifest complete |
