# Quickstart: Metricate

**Feature**: 001-metricate-product  
**Version**: 1.0

---

## Installation

### From PyPI (when published)

```bash
pip install metricate
```

### From Source

```bash
git clone <repo>/metricate.git
cd metricate
pip install -e .
```

### Dependencies

- Python ≥ 3.10
- pandas ≥ 2.0
- numpy ≥ 1.24
- scikit-learn ≥ 1.3
- plotly ≥ 5.0
- click ≥ 8.0 (for CLI)

---

## Quick Examples

### Python Module

#### Evaluate a Single Clustering

```python
import metricate

# Basic evaluation
result = metricate.evaluate("clustering.csv")

# Print formatted table
print(result.to_table())

# Get as DataFrame
df = result.to_dataframe()
print(df[["metric", "value", "direction"]])
```

**Output**:
```
Clustering Evaluation: clustering.csv
=====================================
Points: 10,000 | Clusters: 17 | Dimensions: 50

┌─────────────────────┬──────────┬────────────┬───────────┬──────────┐
│ Metric              │ Value    │ Range      │ Direction │ Tier     │
├─────────────────────┼──────────┼────────────┼───────────┼──────────┤
│ Silhouette          │ 0.4521   │ [-1, 1]    │ ↑ Higher  │ Original │
│ Davies-Bouldin      │ 1.2341   │ [0, ∞)     │ ↓ Lower   │ Original │
│ Calinski-Harabasz   │ 1523.45  │ [0, ∞)     │ ↑ Higher  │ Original │
│ ...                 │ ...      │ ...        │ ...       │ ...      │
└─────────────────────┴──────────┴────────────┴───────────┴──────────┘

34 metrics computed in 2.34s
```

#### Compare Two Clusterings

```python
# Compare two clusterings
comparison = metricate.compare("baseline.csv", "improved.csv")

# See the winner
print(f"Overall winner: {comparison.summary['overall_winner']}")
print(f"A wins: {comparison.summary['wins_a']}")
print(f"B wins: {comparison.summary['wins_b']}")

# Print comparison table
print(comparison.to_table())
```

**Output**:
```
Clustering Comparison
=====================
A: baseline.csv (10,000 points, 15 clusters)
B: improved.csv (10,000 points, 17 clusters)

┌─────────────────┬──────────┬──────────┬────────┐
│ Metric          │ A        │ B        │ Winner │
├─────────────────┼──────────┼──────────┼────────┤
│ Silhouette      │ 0.4521   │ 0.5234   │ B      │
│ Davies-Bouldin  │ 1.2341   │ 0.9876   │ B      │
│ Calinski-Harabasz│ 1523.45 │ 1876.32  │ B      │
│ ...             │ ...      │ ...      │ ...    │
└─────────────────┴──────────┴──────────┴────────┘

Summary: A=12, B=18, Ties=4
Overall Winner: B ✓
```

#### Generate Degraded Datasets

```python
# Generate degradations with visualizations
result = metricate.degrade(
    "clustering.csv",
    output_dir="./degraded/",
    levels=[0.10, 0.25, 0.50],
    visualize=True
)

print(f"Generated {len(result.files_generated)} degraded datasets")
print(f"Visualizations: {result.output_dir}/visualizations/index.html")
```

#### Exclude Specific Metrics

```python
# Skip expensive O(n²) metrics
result = metricate.evaluate(
    "large_dataset.csv",
    exclude=["Gamma", "Tau", "Point-Biserial"]
)

# Force all metrics even for large datasets
result = metricate.evaluate(
    "large_dataset.csv",
    force_all=True  # Computes all 34 metrics regardless of size
)
```

---

### Command Line Interface

#### Evaluate

```bash
# Basic evaluation (table output)
metricate evaluate clustering.csv

# JSON output
metricate evaluate clustering.csv --format json

# Save to file
metricate evaluate clustering.csv --format csv -o results.csv

# Exclude metrics
metricate evaluate clustering.csv --exclude Gamma,Tau,G-plus
```

#### Compare

```bash
# Compare two clusterings
metricate compare baseline.csv improved.csv

# JSON output for scripting
metricate compare baseline.csv improved.csv --format json
```

#### Degrade

```bash
# Generate all degradations
metricate degrade clustering.csv ./output/

# Specific levels and types
metricate degrade clustering.csv ./output/ \
  --levels 0.10,0.25 \
  --types random_removal,label_swap_random

# Skip visualizations (faster)
metricate degrade clustering.csv ./output/ --no-visualize
```

#### List Available Options

```bash
# List all metrics
metricate list metrics

# With full reference info
metricate list metrics --reference

# List degradation types
metricate list degradations
```

---

## Input Format

### Required CSV Structure

```csv
cluster_id,embedding_0,embedding_1,...,embedding_49
0,0.123,0.456,...,0.789
0,0.234,0.567,...,0.890
1,0.345,0.678,...,0.901
-1,0.111,0.222,...,0.333  # -1 = noise (excluded from metrics)
```

### Column Requirements

| Column | Required | Description |
|--------|----------|-------------|
| `cluster_id` | ✓ | Cluster assignment (int, -1 for noise) |
| `embedding_*` | ✓ (1+) | Embedding dimensions (float) |

### Auto-Detection

Metricate auto-detects columns:
- **Label column**: `cluster_id` > `label` > `cluster` > first int column
- **Embedding columns**: `embedding_*` > `dim_*` > `feature_*` > all float columns

---

## Metric Reference

### Quick Reference

| Category | Count | Examples |
|----------|-------|----------|
| Internal Original | 6 | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| Tier 1 CVIs | 6 | Ball-Hall, Ratkowsky-Lance, R-squared |
| Tier 2 CVIs | 14 | Gamma, Tau, Generalized Dunn, S_Dbw |
| Tier 3 CVIs | 4 | Banfield-Raftery, NIVA, Score Function |
| External | 4 | Adjusted Rand Index, Variation of Information |

### O(n²) Metrics (auto-skipped for n > 50,000)

These metrics are skipped by default for large datasets:
- Gamma
- Tau
- Point-Biserial
- G-plus
- McClain-Rao
- NIVA

Use `--force-all` or `force_all=True` to compute them anyway.

---

## Common Workflows

### 1. Evaluate Clustering Quality

```python
import metricate

# Load and evaluate
result = metricate.evaluate("my_clustering.csv")

# Check key metrics
df = result.to_dataframe()
silhouette = df[df.metric == "Silhouette"]["value"].values[0]
print(f"Silhouette Score: {silhouette:.4f}")

# Good clustering typically has:
# - Silhouette > 0.5
# - Davies-Bouldin < 1.0
# - Calinski-Harabasz: higher is better
```

### 2. Compare Algorithm Variants

```python
import metricate

# Run your clustering algorithms, save results
# Then compare:
result = metricate.compare("kmeans_k15.csv", "kmeans_k20.csv")

if result.summary["overall_winner"] == "A":
    print("K=15 performs better overall")
else:
    print("K=20 performs better overall")

# See which metrics favor which
df = result.results
print(df[df.winner == "A"]["metric"].tolist())  # Metrics favoring K=15
```

### 3. Test Metric Robustness

```python
import metricate

# Generate degraded versions
metricate.degrade("baseline.csv", "./degraded/")

# Evaluate each degradation
import glob
for csv_file in sorted(glob.glob("./degraded/*.csv")):
    result = metricate.evaluate(csv_file)
    print(f"{csv_file}: Silhouette = {result.get_metric('Silhouette'):.4f}")
```

---

## Troubleshooting

### "InsufficientClustersError: Found 1 cluster, need at least 2"

Your data only has one unique cluster_id (excluding -1 noise). Clustering metrics require at least 2 clusters.

### "6 metrics skipped for dataset >50k rows"

This is expected behavior. For datasets with > 50,000 points, O(n²) metrics are skipped by default. Use `--force-all` to compute them (may take several minutes).

### "Column 'cluster_id' not found"

Specify your label column explicitly:
```python
metricate.evaluate("data.csv", label_col="my_label_column")
```

### Comparison shows "Warning: Row counts differ"

This is a warning, not an error. Metricate will still compare the clusterings, but be aware they may represent different data subsets.

---

## Next Steps

- See [API Reference](contracts/api.md) for complete function signatures
- See [Data Model](data-model.md) for detailed schemas
- See [Metric Reference](../research.md) for metric formulas and interpretations
