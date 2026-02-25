# Quickstart: Training Dataset Generator

## Installation

No additional installation required - uses existing metricate dependencies.

For Parquet export (optional):
```bash
pip install pyarrow
```

## Basic Usage

### Single File

```python
import metricate

# Generate training data from one clustering
result = metricate.generate_training_data(
    csv_path="datasets/good_clustering.csv",
    output_dir="./training_output",
)

# Export to CSV
result.to_csv("training_dataset.csv")

# Or get DataFrame directly
df = result.to_dataframe()
print(df.shape)  # (77, ~80) - 1 original + 76 degraded × ~80 columns
```

### Batch Processing

```python
# Process all clusterings in a directory
result = metricate.generate_training_data_batch(
    input_dir="./good_clusterings/",
    output_dir="./training_output",
)

# Summary
print(result.summary())
```

### Custom Configuration

```python
# Only specific degradation types and levels
result = metricate.generate_training_data(
    csv_path="clustering.csv",
    output_dir="./output",
    types=["label_swap_random", "merge_nearest"],  # Only 2 types
    levels=["10pct", "25pct"],                      # Only 2 levels
)

# Result: 1 original + 2 types × 2 levels = 5 rows
```

## Output Structure

```
training_output/
├── degraded/                    # Degraded CSV files
│   ├── label_swap_random_5pct.csv
│   ├── label_swap_random_10pct.csv
│   └── ...
├── training_dataset.csv         # Main output (if to_csv called)
└── manifest.json                # Generation metadata
```

## Output Columns

| Category | Columns |
|----------|---------|
| Identity | `clustering_name`, `topic` |
| Structure | `n_clusters`, `n_samples` |
| Labels | `quality` (0/1), `quality_score` (0-1), `degradation_type`, `degradation_level` |
| Raw Metrics | `Silhouette`, `Davies_Bouldin`, ... (30 columns) |
| Normalized | `Silhouette_norm`, `Davies_Bouldin_norm`, ... (30 columns) |
| Metadata | `metrics_computed`, `metrics_failed` |

## Use for ML Training

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load training data
df = result.to_dataframe()

# Features: normalized metric columns
feature_cols = [c for c in df.columns if c.endswith('_norm')]
X = df[feature_cols]

# Target: quality label
y = df['quality']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
