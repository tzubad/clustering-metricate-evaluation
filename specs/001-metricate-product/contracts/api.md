# API Contracts: Metricate

**Feature**: 001-metricate-product  
**Version**: 1.0  
**Status**: Draft

---

## Python Module API (P1)

### Core Functions

#### `metricate.evaluate()`

Evaluate a single clustering and return all metric scores.

```python
def evaluate(
    csv_path: str | Path,
    *,
    exclude: list[str] | None = None,
    force_all: bool = False,
    label_col: str = "cluster_id",
    embedding_cols: list[str] | None = None,  # Auto-detect if None
) -> EvaluationResult:
    """
    Calculate all clustering metrics for a dataset.
    
    Args:
        csv_path: Path to CSV file with cluster_id and embedding columns.
        exclude: List of metric names to skip (case-insensitive).
        force_all: If True, compute O(n²) metrics even for large datasets.
        label_col: Name of the cluster label column.
        embedding_cols: List of embedding column names. Auto-detected if None.
    
    Returns:
        EvaluationResult containing:
        - metrics: List[MetricValue] with 34 metric results
        - metadata: dict with n_points, n_clusters, n_dimensions, time
        - warnings: list[str] with any warnings
    
    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If CSV is invalid or missing required columns.
        InsufficientClustersError: If fewer than 2 clusters (excluding noise).
    
    Example:
        >>> result = metricate.evaluate("clustering.csv")
        >>> print(result.to_table())
        >>> df = result.to_dataframe()
    """
```

**Behavior**:
- Auto-detects embedding columns if not specified (columns matching `embedding_*`, `dim_*`, or all float columns)
- Skips 6 O(n²) metrics (Gamma, Tau, Point-Biserial, G-plus, McClain-Rao, NIVA) if n > 50,000 unless `force_all=True`
- Excludes noise points (cluster_id = -1) from calculations
- Returns all 34 metrics with computed values or skip reasons

---

#### `metricate.compare()`

Compare two clusterings and determine the winner.

```python
def compare(
    csv_path_a: str | Path,
    csv_path_b: str | Path,
    *,
    exclude: list[str] | None = None,
    force_all: bool = False,
    label_col: str = "cluster_id",
    embedding_cols: list[str] | None = None,
) -> ComparisonResult:
    """
    Compare two clusterings and determine which is better.
    
    Args:
        csv_path_a: Path to first clustering CSV.
        csv_path_b: Path to second clustering CSV.
        exclude: List of metric names to skip.
        force_all: If True, compute O(n²) metrics even for large datasets.
        label_col: Name of the cluster label column (same for both).
        embedding_cols: Embedding columns (auto-detected if None).
    
    Returns:
        ComparisonResult containing:
        - results: DataFrame with per-metric comparison
        - summary: dict with wins_a, wins_b, ties, overall_winner
        - warnings: list with any warnings (e.g., different row counts)
    
    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If CSVs are invalid or incompatible.
        DimensionMismatchError: If embedding dimensions don't match.
    
    Example:
        >>> result = metricate.compare("clustering_v1.csv", "clustering_v2.csv")
        >>> print(f"Winner: {result.summary['overall_winner']}")
        >>> print(result.to_table())
    """
```

**Behavior**:
- Validates both files independently
- Warns (does not error) if row counts differ
- Requires same number of embedding dimensions
- Determines winner per-metric based on direction (higher/lower is better)
- Overall winner = clustering with more metric wins

---

#### `metricate.degrade()`

Generate degraded versions of a clustering dataset.

```python
def degrade(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    levels: list[float] | None = None,  # Default: [0.05, 0.10, 0.25, 0.50]
    types: list[str] | None = None,  # Default: all 19 types
    visualize: bool = True,
    label_col: str = "cluster_id",
    embedding_cols: list[str] | None = None,
) -> DegradationResult:
    """
    Generate degraded versions of a clustering dataset.
    
    Args:
        csv_path: Path to source clustering CSV.
        output_dir: Directory for output files.
        levels: Degradation levels (fractions). Default: [0.05, 0.10, 0.25, 0.50].
        types: Degradation types to apply. Default: all 19 types.
        visualize: If True, generate HTML visualizations.
        label_col: Cluster label column name.
        embedding_cols: Embedding columns (auto-detected if None).
    
    Returns:
        DegradationResult containing:
        - output_dir: Path to output directory
        - manifest: dict with metadata
        - files_generated: list of CSV paths
        - visualizations_generated: list of HTML paths
    
    Raises:
        FileNotFoundError: If source file doesn't exist.
        ValueError: If invalid degradation type specified.
    
    Example:
        >>> result = metricate.degrade(
        ...     "clustering.csv",
        ...     "degraded_output/",
        ...     levels=[0.10, 0.25],
        ...     types=["random_removal", "label_swap_random"]
        ... )
        >>> print(f"Generated {len(result.files_generated)} files")
    """
```

**Behavior**:
- Creates output directory if it doesn't exist
- Generates `<type>_<level>pct.csv` files for each combination
- Writes `manifest.json` with complete metadata
- If `visualize=True`, creates `visualizations/` subdirectory with Plotly HTMLs

---

### Utility Functions

#### `metricate.list_metrics()`

```python
def list_metrics(
    include_reference: bool = False,
) -> list[str] | pd.DataFrame:
    """
    List all available metrics.
    
    Args:
        include_reference: If True, return DataFrame with ranges/directions.
    
    Returns:
        List of metric names, or DataFrame with full reference.
    """
```

#### `metricate.list_degradations()`

```python
def list_degradations() -> list[str]:
    """Return list of all available degradation types."""
```

---

## CLI API (P2)

### `metricate evaluate`

```bash
metricate evaluate <csv_path> [OPTIONS]

Arguments:
  csv_path          Path to clustering CSV file

Options:
  --exclude TEXT    Metric names to exclude (comma-separated or multiple flags)
  --force-all       Compute O(n²) metrics even for large datasets
  --label-col TEXT  Cluster label column name [default: cluster_id]
  --format TEXT     Output format: table|json|csv [default: table]
  -o, --output PATH Write output to file instead of stdout
  -q, --quiet       Suppress warnings
  -v, --verbose     Show computation details

Examples:
  metricate evaluate clustering.csv
  metricate evaluate clustering.csv --exclude Gamma,Tau --format json
  metricate evaluate clustering.csv --force-all -o results.csv --format csv
```

**Exit Codes**:
- 0: Success
- 1: File not found or invalid
- 2: Insufficient clusters
- 3: Invalid options

---

### `metricate compare`

```bash
metricate compare <csv_a> <csv_b> [OPTIONS]

Arguments:
  csv_a             Path to first clustering CSV
  csv_b             Path to second clustering CSV

Options:
  --exclude TEXT    Metric names to exclude
  --force-all       Compute O(n²) metrics for large datasets
  --format TEXT     Output format: table|json|csv [default: table]
  -o, --output PATH Write output to file
  -v, --verbose     Show detailed comparison

Examples:
  metricate compare v1.csv v2.csv
  metricate compare baseline.csv improved.csv --format json
```

**Output** (table format):
```
Clustering Comparison
=====================
File A: v1.csv (10,000 points, 15 clusters)
File B: v2.csv (10,000 points, 17 clusters)

┌─────────────────┬──────────┬──────────┬────────┬───────────┐
│ Metric          │ A        │ B        │ Winner │ Direction │
├─────────────────┼──────────┼──────────┼────────┼───────────┤
│ Silhouette      │ 0.4521   │ 0.5234   │ B      │ ↑ Higher  │
│ Davies-Bouldin  │ 1.2341   │ 0.9876   │ B      │ ↓ Lower   │
│ ...             │ ...      │ ...      │ ...    │ ...       │
└─────────────────┴──────────┴──────────┴────────┴───────────┘

Summary: A wins 12, B wins 18, Ties 4
Overall Winner: B
```

---

### `metricate degrade`

```bash
metricate degrade <csv_path> <output_dir> [OPTIONS]

Arguments:
  csv_path          Path to source clustering CSV
  output_dir        Directory for output files

Options:
  --levels TEXT     Degradation levels (comma-separated) [default: 0.05,0.10,0.25,0.50]
  --types TEXT      Degradation types (comma-separated or 'all')
  --no-visualize    Skip HTML visualization generation
  --label-col TEXT  Cluster label column name [default: cluster_id]
  -v, --verbose     Show progress

Examples:
  metricate degrade clustering.csv ./degraded/
  metricate degrade clustering.csv ./out/ --levels 0.10,0.25 --types random_removal,label_swap_random
  metricate degrade clustering.csv ./out/ --no-visualize
```

**Output**:
```
Generating degradations for clustering.csv
├── Levels: 5%, 10%, 25%, 50%
├── Types: 19 degradation types
└── Output: ./degraded/

[████████████████████████████████████████] 76/76 complete

Generated:
├── 76 degraded datasets
├── 19 HTML visualizations
└── manifest.json

Open ./degraded/visualizations/index.html to view results.
```

---

### `metricate list`

```bash
metricate list [WHAT] [OPTIONS]

Arguments:
  WHAT              What to list: metrics|degradations [default: metrics]

Options:
  --reference       Show full reference (ranges, directions, tiers)

Examples:
  metricate list metrics
  metricate list metrics --reference
  metricate list degradations
```

---

## Error Responses

### Standard Error Format

```python
class MetricateError(Exception):
    """Base exception for Metricate errors."""
    code: str
    message: str
    details: dict

class FileNotFoundError(MetricateError):
    code = "FILE_NOT_FOUND"
    
class InvalidCSVError(MetricateError):
    code = "INVALID_CSV"
    
class ColumnNotFoundError(MetricateError):
    code = "COLUMN_NOT_FOUND"
    details = {"missing": ["column_name"]}
    
class InsufficientClustersError(MetricateError):
    code = "INSUFFICIENT_CLUSTERS"
    details = {"found": 1, "required": 2}
    
class DimensionMismatchError(MetricateError):
    code = "DIMENSION_MISMATCH"
    details = {"dimensions_a": 50, "dimensions_b": 100}
```

### CLI Error Display

```
Error: FILE_NOT_FOUND
Could not find file: nonexistent.csv
```

---

## Response Schemas

### JSON Output: evaluate

```json
{
  "metrics": [
    {
      "metric": "Silhouette",
      "value": 0.4521,
      "range": "[-1, 1]",
      "direction": "higher",
      "tier": "Original",
      "computed": true,
      "skip_reason": null
    },
    {
      "metric": "Gamma",
      "value": null,
      "range": "[-1, 1]",
      "direction": "higher",
      "tier": "Tier 2",
      "computed": false,
      "skip_reason": "Dataset >50k rows (use --force-all)"
    }
  ],
  "metadata": {
    "n_points": 55000,
    "n_clusters": 17,
    "n_dimensions": 50,
    "computation_time_seconds": 2.34,
    "metrics_computed": 28,
    "metrics_skipped": 6
  },
  "warnings": [
    "6 O(n²) metrics skipped for dataset >50k rows. Use --force-all to compute."
  ]
}
```

### JSON Output: compare

```json
{
  "comparison": [
    {
      "metric": "Silhouette",
      "value_a": 0.4521,
      "value_b": 0.5234,
      "winner": "B",
      "direction": "higher",
      "range": "[-1, 1]"
    }
  ],
  "summary": {
    "wins_a": 12,
    "wins_b": 18,
    "ties": 4,
    "overall_winner": "B"
  },
  "clustering_a": {
    "path": "v1.csv",
    "n_points": 10000,
    "n_clusters": 15
  },
  "clustering_b": {
    "path": "v2.csv",
    "n_points": 10000,
    "n_clusters": 17
  },
  "warnings": []
}
```

---

## Web UI API (P4)

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Dashboard / upload page |
| `POST` | `/api/evaluate` | Evaluate uploaded CSV |
| `POST` | `/api/compare` | Compare two uploaded CSVs |
| `GET` | `/api/metrics` | List all metrics with reference |
| `GET` | `/api/degradations` | List degradation types |

### POST /api/evaluate

**Request**: `multipart/form-data`
```
file: <clustering.csv>
exclude: ["Gamma", "Tau"]  (optional JSON array)
force_all: false  (optional boolean)
```

**Response**: Same as JSON output for `evaluate()`

### POST /api/compare

**Request**: `multipart/form-data`
```
file_a: <v1.csv>
file_b: <v2.csv>
exclude: []
force_all: false
```

**Response**: Same as JSON output for `compare()`

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-01-15 | Initial API design |
