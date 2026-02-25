# API Contract: Training Dataset Generator

## Public Functions

### `metricate.generate_training_data()`

Generate training data from a single clustering CSV.

```python
def generate_training_data(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    types: list[str] | None = None,
    levels: list[str] | None = None,
    exclude: list[str] | None = None,
    force_all: bool = False,
    topic: str | None = None,
    random_seed: int = 42,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
) -> TrainingDataResult:
    """
    Generate a training dataset from a single clustering CSV.
    
    Args:
        csv_path: Path to input clustering CSV.
        output_dir: Directory for degraded CSVs and outputs.
        types: Degradation types to apply (None = all 19).
        levels: Degradation levels (None = ["5pct", "10pct", "25pct", "50pct"]).
        exclude: Metrics to exclude from calculation.
        force_all: Compute O(n²) metrics on large datasets.
        topic: Manual topic assignment (None = extract from filename).
        random_seed: Seed for reproducibility.
        label_col: Cluster label column (auto-detected if None).
        embedding_cols: Embedding columns (auto-detected if None).
    
    Returns:
        TrainingDataResult with records for original + all degraded versions.
    
    Raises:
        FileNotFoundError: If csv_path doesn't exist.
        InvalidCSVError: If CSV is malformed.
        InsufficientClustersError: If clustering has < 2 clusters.
    """
```

### `metricate.generate_training_data_batch()`

Generate training data from multiple clustering CSVs.

```python
def generate_training_data_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    topic_mapping: dict[str, str] | None = None,
    types: list[str] | None = None,
    levels: list[str] | None = None,
    exclude: list[str] | None = None,
    force_all: bool = False,
    random_seed: int = 42,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
) -> TrainingDataResult:
    """
    Generate training data from all CSVs in a directory.
    
    Args:
        input_dir: Directory containing clustering CSVs.
        output_dir: Directory for degraded CSVs and outputs.
        topic_mapping: Dict mapping filename to topic.
        (other args same as generate_training_data)
    
    Returns:
        TrainingDataResult with combined records from all files.
    """
```

---

## TrainingDataResult

```python
@dataclass
class TrainingDataResult:
    """Container for training dataset generation results."""
    
    records: list[dict]
    """List of record dictionaries, one per clustering version."""
    
    metadata: dict
    """Generation metadata: timestamp, parameters, file counts."""
    
    warnings: list[str]
    """Non-fatal warnings encountered during generation."""
    
    errors: list[str]
    """Files that failed to process."""
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to pandas DataFrame with proper dtypes."""
    
    def to_csv(self, path: str | Path) -> None:
        """Export to CSV file."""
    
    def to_parquet(self, path: str | Path) -> None:
        """Export to Parquet file (requires pyarrow)."""
    
    def summary(self) -> str:
        """Human-readable summary of generation results."""
    
    @property
    def n_originals(self) -> int:
        """Count of original clusterings processed."""
    
    @property
    def n_degraded(self) -> int:
        """Count of degraded versions generated."""
```

---

## Record Schema

Each record in `TrainingDataResult.records`:

```python
{
    # Identity
    "clustering_name": str,       # e.g., "narrative_17clusters"
    "topic": str,                 # e.g., "narrative"
    
    # Structure
    "n_clusters": int,            # e.g., 17
    "n_samples": int,             # e.g., 1247
    
    # Labels
    "quality": int,               # 1 = original, 0 = degraded
    "quality_score": float,       # 1.0, 0.95, 0.90, 0.75, 0.50
    "degradation_type": str | None,
    "degradation_level": str | None,
    
    # Raw metrics (30 columns)
    "Silhouette": float | None,
    "Davies-Bouldin": float | None,
    "Calinski-Harabasz": float | None,
    # ... (27 more)
    
    # Normalized metrics (30 columns)
    "Silhouette_norm": float | None,
    "Davies-Bouldin_norm": float | None,
    # ... (28 more)
    
    # Metadata
    "metrics_computed": int,
    "metrics_failed": str,        # Comma-separated list
}
```

---

## Error Handling

| Exception | Condition | Recovery |
|-----------|-----------|----------|
| `FileNotFoundError` | Input file doesn't exist | Fatal |
| `InvalidCSVError` | Malformed CSV | Skip file in batch mode |
| `InsufficientClustersError` | < 2 clusters | Skip with warning |
| `ColumnNotFoundError` | Missing required columns | Skip file in batch mode |

---

## Normalization

Normalized columns (`*_norm`) use **percentile rank** within the generated dataset:

```python
# For "higher is better" metrics:
norm_value = percentile_rank(raw_value)  # 0.0 to 1.0

# For "lower is better" metrics:
norm_value = 1.0 - percentile_rank(raw_value)  # Inverted
```

This ensures all `*_norm` columns have:
- Range: [0.0, 1.0]
- Direction: Higher is always better
- Distribution: Uniform (robust to outliers)
