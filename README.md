# Metricate

A comprehensive clustering evaluation toolkit that calculates 34 quality metrics, compares clusterings, and generates degraded datasets for testing metric robustness.

## Installation

```bash
pip install metricate
```

Or install from source:

```bash
git clone <repo>/metricate.git
cd metricate
pip install -e .
```

### Requirements

- Python ≥ 3.10
- pandas ≥ 2.0
- numpy ≥ 1.24
- scikit-learn ≥ 1.3
- plotly ≥ 5.0
- click ≥ 8.0

## Quick Start

### Evaluate a Single Clustering

```python
import metricate

# Evaluate and get all 34 metrics
result = metricate.evaluate("clustering.csv")

# Print formatted table
print(result.to_table())

# Output:
# Clustering Evaluation: clustering.csv
# =====================================
# Points: 10,000 | Clusters: 17 | Dimensions: 50
#
# ┌─────────────────────┬──────────┬────────────┬───────────┐
# │ Metric              │ Value    │ Range      │ Direction │
# ├─────────────────────┼──────────┼────────────┼───────────┤
# │ Silhouette          │ 0.4521   │ [-1, 1]    │ ↑ Higher  │
# │ Davies-Bouldin      │ 1.2341   │ [0, ∞)     │ ↓ Lower   │
# │ Calinski-Harabasz   │ 1523.45  │ [0, ∞)     │ ↑ Higher  │
# │ ...                 │ ...      │ ...        │ ...       │
# └─────────────────────┴──────────┴────────────┴───────────┘

# Get as DataFrame for further analysis
df = result.to_dataframe()
print(df[["metric", "value", "direction"]])

# Export to JSON or CSV
result.to_json()  # Returns JSON string
result.to_csv()   # Returns CSV string
```

### Compare Two Clusterings

```python
import metricate

# Compare two clusterings
comparison = metricate.compare(
    "baseline.csv", 
    "improved.csv",
    name_a="Baseline",
    name_b="Improved"
)

# See the winner
print(f"Overall winner: {comparison.winner}")
print(f"Wins: {comparison.wins}")  # {'Baseline': 12, 'Improved': 18, 'Tie': 4}

# Print comparison table
print(comparison.to_table())

# Output:
# ┌─────────────────┬──────────┬──────────┬────────┐
# │ Metric          │ Baseline │ Improved │ Winner │
# ├─────────────────┼──────────┼──────────┼────────┤
# │ Silhouette      │ 0.4521   │ 0.5234   │ Improved │
# │ Davies-Bouldin  │ 1.2341   │ 0.9876   │ Improved │
# │ ...             │ ...      │ ...      │ ...    │
# └─────────────────┴──────────┴──────────┴────────┘
```

### Generate Degraded Datasets

```python
import metricate

# Generate all 19 degradation types at 4 intensity levels (76 total)
result = metricate.degrade(
    "clustering.csv",
    output_dir="./degraded/",
    visualize=True  # Generate HTML visualizations
)

print(result.summary())
# Output:
# Degradation Generation Complete
# ========================================
# Output directory: ./degraded/
# Total degradations: 76
# Manifest: ./degraded/manifest.json
# Index HTML: ./degraded/index.html

# Generate specific degradations only
result = metricate.degrade(
    "clustering.csv",
    output_dir="./output/",
    types=["label_swap_random", "noise_injection"],
    levels=["10pct", "25pct"],
    visualize=False  # Skip visualizations for speed
)
```

### List Available Options

```python
import metricate

# List all 34 metrics
metrics = metricate.list_metrics()
print(f"{len(metrics)} metrics available")

# Get full reference with directions and ranges
df = metricate.list_metrics(include_reference=True)
print(df[["metric", "direction", "complexity"]])

# List all 19 degradation types by category
degradations = metricate.list_degradations()
for category, types in degradations.items():
    print(f"{category}: {types}")
```

## Command Line Interface

```bash
# Evaluate a clustering (table output)
metricate evaluate clustering.csv

# JSON output for scripting
metricate evaluate clustering.csv --format json

# Save to file
metricate evaluate clustering.csv --format csv -o results.csv

# Exclude expensive metrics
metricate evaluate clustering.csv --exclude Gamma,Tau,G-plus

# Compare two clusterings
metricate compare baseline.csv improved.csv
metricate compare v1.csv v2.csv --name-a "Version 1" --name-b "Version 2"

# Generate degradations
metricate degrade clustering.csv ./output/

# Specific types and levels
metricate degrade clustering.csv ./output/ \
  --types label_swap_random,noise_injection \
  --levels 10pct,25pct

# Skip visualizations (faster)
metricate degrade clustering.csv ./output/ --no-visualize

# List available options
metricate list metrics
metricate list degradations
```

## Input Format

Your CSV must have:
- `cluster_id` column: Cluster assignments (integer, -1 for noise points)
- Embedding columns: Named `embedding_*`, `dim_*`, or any float columns

Example:
```csv
cluster_id,embedding_0,embedding_1,embedding_2
0,0.123,0.456,0.789
0,0.234,0.567,0.890
1,0.345,0.678,0.901
-1,0.111,0.222,0.333
```

## 34 Clustering Metrics

### Internal Original (6)
- Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn Index, SSE, NCI

### Tier 1 CVIs (6)
- Ball-Hall, Ratkowsky-Lance, Ray-Turi, RMSSTD, R-squared, Wemmert-Gancarski

### Tier 2 CVIs (14)
- CS index, COP, S_Dbw, Det Ratio, Gamma, Generalized Dunn, G-plus, I-index (PBM), Log_Det_Ratio, McClain-Rao, Point-Biserial, SD validity, Tau, Trace_WiB, Ksq_DetW

### Tier 3 CVIs (4)
- Banfield-Raftery, Negentropy, NIVA, Score Function, Scott-Symons

### External (4)
- Adjusted Rand Index, Van Dongen, Variation of Information, Omega

## Large Dataset Handling (>50,000 rows)

For datasets exceeding 50,000 rows, the following O(n²) metrics are automatically skipped to ensure reasonable computation time:

- **Gamma**
- **Tau**
- **Point-Biserial**
- **G-plus**
- **McClain-Rao**
- **NIVA**

### Override Auto-Skip

To compute all metrics regardless of dataset size:

```python
# Python API
result = metricate.evaluate("large_data.csv", force_all=True)
```

```bash
# CLI
metricate evaluate large_data.csv --force-all
```

**Warning**: Computing O(n²) metrics on large datasets may take several minutes and require significant memory.

## Excluding Metrics
Reasons to exclude metrics:
- Your clustering picked the number of K clusters by optimising for Tau? then you shouldn't look at Tau as a metric
- Shorten runtime for the comparison\evaluation
Skip specific metrics to save computation time:

```python
result = metricate.evaluate(
    "clustering.csv",
    exclude=["Gamma", "Tau", "S_Dbw"]
)
```

```bash
metricate evaluate clustering.csv --exclude Gamma,Tau,S_Dbw
```

## 19 Degradation Types

| Category | Types |
|----------|-------|
| Point Removal | random_removal, core_removal |
| Label Modification | boundary_reassignment, label_swap_random, label_swap_neighboring, label_swap_distant |
| Embedding Noise | noise_injection, embedding_perturbation, centroid_displacement |
| Cluster Merge | merge_nearest, merge_farthest, merge_random |
| Cluster Split | split_largest, split_loosest, split_random |
| Cluster Remove | remove_smallest_clusters, remove_largest_clusters, remove_tightest_clusters |

## License

MIT
