# Feature Specification: Metricate Training Dataset Generator

**Feature Branch**: `002-training-dataset-generator`  
**Created**: February 25, 2026  
**Status**: Draft  
**Input**: User description: "Module to automatically create a dataset of good and bad clusterings with their metricate scores for training a meta-model to learn metric weights"

## Problem Statement

To compound all 34 clustering metrics into a single quality score, we need to determine the relative importance/informativeness of each metric. This requires training a model on a labeled dataset of clusterings (good vs. degraded) with their metric values.

**Goal**: Create a module that takes good clustering CSVs, systematically degrades them, calculates all metrics for both good and degraded versions, and outputs a structured training dataset.

## Clarifications

### Session 2026-02-25

- Q: What metadata should accompany each clustering? → A: Clustering name (filename), topic (extracted from filename or provided), number of clusters, degradation type (if degraded), degradation level (if degraded)
- Q: How should the output dataset be structured? → A: A single CSV/DataFrame with one row per clustering (both good and degraded), containing all metric values as columns plus label column
- Q: Should we support batch processing of multiple input files? → A: Yes, the module should accept a directory or list of CSV files
- Q: What label values for good vs degraded? → A: Binary label: 1 = good (original), 0 = degraded; also include degradation_type and degradation_level columns for analysis
- Q: Should we calculate external metrics? → A: No - external metrics (ARI, Van Dongen, VI, Omega) require ground truth and aren't applicable for this use case
- Q: How to handle metrics that fail to compute? → A: Store NaN for failed metrics; include a metadata column listing which metrics failed
- Q: Should degraded CSVs be saved to disk or only processed in memory? → A: Save to disk in `output_dir` alongside the training dataset for debugging and reuse
- Q: Should metric values be normalized for ML training? → A: Include both raw columns AND normalized columns (e.g., `Silhouette`, `Silhouette_norm`) for maximum flexibility
- Q: How should unbounded metrics be normalized to [0, 1]? → A: Use percentile rank within the generated dataset (robust to outliers, no hyperparameters, creates uniform distribution)
- Q: Should the module support incremental/resumable generation? → A: No - keep it simple; degraded CSVs are saved so metrics can be recalculated manually if needed
- Q: Should quality label be binary or encode severity? → A: Both - include `quality` (binary: 1=good, 0=degraded) AND `quality_score` (continuous: 1.0=original, 0.95=5%, 0.90=10%, 0.75=25%, 0.50=50%)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Single Clustering Training Data Generation (Priority: P1)

A researcher has a single good clustering CSV and wants to generate training data. They provide the file, and the module degrades it at multiple levels using all degradation types, calculates metrics for each version, and returns a DataFrame ready for model training.

**Why this priority**: Core functionality - enables immediate use for training data generation.

**Independent Test**: Provide a single clustering CSV, verify that degraded versions are generated and all metrics are calculated.

**Acceptance Scenarios**:

1. **Given** a user provides a clustering CSV, **When** they call the generate function, **Then** the system generates degraded versions at all levels (5%, 10%, 25%, 50%) for all applicable degradation types and calculates metrics for each.

2. **Given** the original clustering is processed, **When** metrics are calculated, **Then** the original clustering is labeled as `quality=1` (good) and all degraded versions are labeled as `quality=0` (degraded).

3. **Given** training data is generated, **When** the user requests output, **Then** each row contains: `clustering_name`, `topic`, `n_clusters`, `n_samples`, all 34 metric columns, `quality` (0/1), `degradation_type` (None for originals), `degradation_level` (None for originals).

---

### User Story 2 - Batch Processing Multiple Clusterings (Priority: P1)

A researcher has a directory of good clustering CSVs (from different topics/experiments) and wants to generate a comprehensive training dataset. They provide the directory path, and the module processes all files.

**Why this priority**: Essential for creating large-scale training datasets from multiple clustering experiments.

**Independent Test**: Provide a directory with multiple CSVs, verify all are processed and combined into one training dataset.

**Acceptance Scenarios**:

1. **Given** a user provides a directory path, **When** they call the batch generate function, **Then** all CSV files in the directory are processed and combined into a single training dataset.

2. **Given** multiple clustering files are processed, **When** generating output, **Then** each row includes the source filename as `clustering_name` and optionally extracted topic information.

3. **Given** some files fail to process, **When** batch processing completes, **Then** the system logs warnings but continues processing other files, returning partial results with a summary of successes/failures.

---

### User Story 3 - Custom Degradation Configuration (Priority: P2)

A researcher wants to focus on specific degradation types or levels (e.g., only label manipulations at 10% and 25% levels) to generate a targeted training dataset.

**Why this priority**: Allows researchers to control experimental conditions and reduce dataset size when needed.

**Independent Test**: Configure specific degradation types and levels, verify only those are applied.

**Acceptance Scenarios**:

1. **Given** a user specifies `types=["label_swap_random", "merge_nearest"]`, **When** generating training data, **Then** only those degradation types are applied.

2. **Given** a user specifies `levels=["10pct", "25pct"]`, **When** generating training data, **Then** only those degradation levels are used.

3. **Given** custom configuration is applied, **When** viewing output, **Then** the resulting dataset reflects only the specified degradation types and levels.

---

### User Story 4 - Export to Multiple Formats (Priority: P2)

A researcher needs the training dataset in various formats for different ML frameworks (pandas DataFrame, CSV file, Parquet for large datasets).

**Why this priority**: Flexibility in output format enables integration with various ML pipelines.

**Independent Test**: Generate training data and export to CSV, Parquet, and return as DataFrame.

**Acceptance Scenarios**:

1. **Given** training data is generated, **When** the user calls `to_dataframe()`, **Then** a pandas DataFrame is returned with proper column types.

2. **Given** training data is generated, **When** the user calls `to_csv(path)`, **Then** a CSV file is saved with all data.

3. **Given** training data is generated, **When** the user calls `to_parquet(path)`, **Then** a Parquet file is saved (efficient for large datasets).

---

### User Story 5 - Topic Extraction/Assignment (Priority: P3)

A researcher wants to associate each clustering with a topic for stratified analysis (e.g., to ensure the model learns across different domains).

**Why this priority**: Nice-to-have for advanced analysis; not required for basic training data generation.

**Acceptance Scenarios**:

1. **Given** filenames contain topic information (e.g., `politics_clustering.csv`), **When** auto-extraction is enabled, **Then** topic is extracted from filename.

2. **Given** a user provides a topic mapping dict `{"file1.csv": "politics", "file2.csv": "sports"}`, **When** generating training data, **Then** topics are assigned according to the mapping.

3. **Given** no topic information is available, **When** generating training data, **Then** topic column contains the filename or "unknown".

---

### Edge Cases

- What happens when a clustering has only 1 cluster? → Skip clustering with warning; some metrics require ≥2 clusters.
- What happens when degradation reduces clusters to <2? → Label that degraded version with metrics=NaN and include warning column.
- How to handle very large files (>100k rows)? → Auto-skip O(n²) metrics with warning; user can override with `force_all=True`.
- What if all metrics fail for a clustering? → Include row with all NaN values and set `metrics_computed=0`.
- What if input CSV lacks required columns? → Skip file with clear error message listing missing columns.

## Requirements *(mandatory)*

### Functional Requirements

#### Core Generation (P1)

- **FR-001**: System MUST accept a single CSV file path or a directory of CSV files as input.
- **FR-002**: System MUST apply all 19 degradation types at configurable levels (default: 5%, 10%, 25%, 50%) to each input clustering.
- **FR-003**: System MUST calculate all 30 internal metrics (excluding 4 external metrics that require ground truth) for both original and degraded clusterings.
- **FR-004**: System MUST output a structured dataset with these columns:
  - `clustering_name`: Source filename (without extension)
  - `topic`: Topic/category (extracted or provided)
  - `n_clusters`: Number of clusters in this version
  - `n_samples`: Number of data points
  - `quality`: Binary label (1 = good/original, 0 = degraded)
  - `quality_score`: Continuous score (1.0 = original, 0.95 = 5% degraded, 0.90 = 10%, 0.75 = 25%, 0.50 = 50%)
  - `degradation_type`: Type of degradation applied (None for originals)
  - `degradation_level`: Intensity level (None for originals)
  - `[metric_name]`: One column per metric with raw values (34 columns)
  - `[metric_name]_norm`: Normalized version of each metric scaled to [0, 1] via percentile rank (34 columns)
  - `metrics_computed`: Count of successfully computed metrics
  - `metrics_failed`: Comma-separated list of failed metric names

#### Configuration (P2)

- **FR-005**: System MUST allow users to specify which degradation types to apply.
- **FR-006**: System MUST allow users to specify which degradation levels to use.
- **FR-007**: System MUST allow users to exclude specific metrics from calculation.
- **FR-008**: System MUST support `force_all=True` to compute O(n²) metrics on large datasets.

#### Output (P2)

- **FR-009**: System MUST provide `to_dataframe()` method returning pandas DataFrame.
- **FR-010**: System MUST provide `to_csv(path)` method for CSV export.
- **FR-011**: System MUST provide `to_parquet(path)` method for Parquet export.
- **FR-012**: System MUST provide `summary()` method with generation statistics.

#### Metadata (P3)

- **FR-013**: System SHOULD support topic extraction from filename patterns.
- **FR-014**: System SHOULD support user-provided topic mapping dictionary.
- **FR-015**: System SHOULD track generation timestamp and parameters for reproducibility.

### Non-Functional Requirements

- **NFR-001**: Processing a single clustering file with 19 degradation types × 4 levels should complete in < 5 minutes for datasets up to 10,000 rows.
- **NFR-002**: System should log progress for batch processing.
- **NFR-003**: System MUST save degraded CSVs to `output_dir` for debugging and allowing metric recalculation without re-degrading.
- **NFR-004**: Random seed should be configurable for reproducibility.

## Data Model

### TrainingDataResult

```python
@dataclass
class TrainingDataResult:
    """Result container for training dataset generation."""
    
    records: list[dict]  # Raw records (one per clustering)
    metadata: dict  # Generation metadata
    warnings: list[str]
    errors: list[str]
    
    def to_dataframe(self) -> pd.DataFrame
    def to_csv(self, path: str | Path) -> None
    def to_parquet(self, path: str | Path) -> None
    def summary(self) -> str
```

### Record Schema

Each record (row) in the output dataset:

```python
{
    "clustering_name": str,      # e.g., "narrative_dataset_17clusters"
    "topic": str,                # e.g., "narrative" or "unknown"
    "n_clusters": int,           # e.g., 17
    "n_samples": int,            # e.g., 1000
    "quality": int,              # 1 = good, 0 = degraded
    "degradation_type": str | None,   # e.g., "label_swap_random" or None
    "degradation_level": str | None,  # e.g., "10pct" or None
    
    # Metrics (34 columns)
    "Silhouette": float | None,
    "Davies-Bouldin": float | None,
    "Calinski-Harabasz": float | None,
    # ... all other metrics ...
    
    # Metadata
    "metrics_computed": int,     # Count of non-null metrics
    "metrics_failed": str,       # Comma-separated failed metric names
}
```

## API Design

### Python Module API

```python
import metricate

# Single file generation
result = metricate.generate_training_data(
    csv_path="clustering.csv",
    output_dir="./training_data",  # Optional: save degraded CSVs
    types=None,                    # None = all 19 types
    levels=None,                   # None = ["5pct", "10pct", "25pct", "50pct"]
    exclude=None,                  # Metrics to exclude
    force_all=False,               # Force O(n²) metrics
    topic=None,                    # Manual topic assignment
    random_seed=42,
)

# Batch generation from directory
result = metricate.generate_training_data_batch(
    input_dir="./clusterings/",
    output_dir="./training_data",
    topic_mapping={"file1.csv": "politics"},  # Optional
    **kwargs,
)

# Access results
df = result.to_dataframe()
result.to_csv("training_dataset.csv")
print(result.summary())
```

### CLI Interface (Optional, P3)

```bash
# Single file
metricate train-data clustering.csv --output training_data.csv

# Batch processing
metricate train-data ./clusterings/ --output training_data.csv --levels 10pct,25pct
```

## Implementation Notes

### Integration with Existing Modules

- **Degradation**: Use `metricate.degrade()` to generate degraded versions
- **Evaluation**: Use `metricate.evaluate()` to calculate metrics
- **Shared code**: Reuse `metricate.core.loader` for CSV loading

### Workflow

```
Input CSVs
    │
    ├── For each good clustering:
    │       │
    │       ├── Calculate metrics → add record (quality=1)
    │       │
    │       └── For each degradation type × level:
    │               │
    │               ├── Apply degradation
    │               ├── Calculate metrics
    │               └── Add record (quality=0)
    │
    └── Combine all records → TrainingDataResult
```

### Metric Handling

- External metrics (ARI, Van Dongen, VI, Omega) are **excluded** by default since they require ground truth labels
- For degraded clusterings, we're comparing against the embedding space quality, not original labels
- Include both absolute metric values and normalized versions (0-1 scale) for better ML training

## Success Criteria

1. ✅ Generate training dataset from single clustering CSV
2. ✅ Batch process multiple clustering files
3. ✅ Export to DataFrame, CSV, and Parquet
4. ✅ Reproducible with random seed
5. ✅ Clear labeling of good (1) vs degraded (0) clusterings
6. ✅ All relevant metadata captured for analysis
7. ✅ Handles edge cases gracefully with warnings

## Resolved Questions

1. ~~Should we include metric direction indicators in the output?~~ → Not needed; direction info available via `metricate.list_metrics(include_reference=True)`
2. ~~Should we normalize metric values to [0, 1] range?~~ → Yes, include both raw and `_norm` columns using percentile rank
3. ~~Should we include cluster size distribution statistics?~~ → Future enhancement; not in initial scope
4. ~~Should we support resumption for interrupted batch processing?~~ → No; keep simple, rely on saved degraded CSVs

## Dependencies

- `metricate.core.evaluator` - Metric calculation
- `metricate.degradation.toolkit` - Degradation generation
- `metricate.core.loader` - CSV loading
- `pandas` - Data manipulation
- `pyarrow` (optional) - Parquet export
