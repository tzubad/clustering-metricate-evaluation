# Data Model: Training Dataset Generator

## Output Schema

### Training Dataset Columns

#### Identity Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `clustering_name` | str | Source filename (without extension) | `"narrative_17clusters"` |
| `topic` | str | Topic/category of the clustering | `"narrative"`, `"politics"` |

#### Structural Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `n_clusters` | int | Number of clusters in this version | `17` |
| `n_samples` | int | Number of data points | `1000` |

#### Label Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `quality` | int | Binary label: 1=good, 0=degraded | `1` or `0` |
| `quality_score` | float | Continuous score based on degradation level | `1.0`, `0.90` |
| `degradation_type` | str \| None | Type applied (None for originals) | `"label_swap_random"` |
| `degradation_level` | str \| None | Intensity level (None for originals) | `"10pct"` |

### Quality Score Mapping

| Level | quality | quality_score |
|-------|---------|---------------|
| Original | 1 | 1.00 |
| 5pct | 0 | 0.95 |
| 10pct | 0 | 0.90 |
| 25pct | 0 | 0.75 |
| 50pct | 0 | 0.50 |

### Metric Columns (34 total)

#### Internal Original (6)
| Column | Range | Direction |
|--------|-------|-----------|
| `Silhouette` | [-1, 1] | Higher ↑ |
| `Dunn_Index` | [0, ∞) | Higher ↑ |
| `Davies_Bouldin` | [0, ∞) | Lower ↓ |
| `Calinski_Harabasz` | [0, ∞) | Higher ↑ |
| `SSE` | [0, ∞) | Lower ↓ |
| `NCI` | [-1, 1] | Higher ↑ |

#### Tier 1 CVIs (6)
| Column | Range | Direction |
|--------|-------|-----------|
| `Ball_Hall` | [0, ∞) | Lower ↓ |
| `Ratkowsky_Lance` | [0, 1] | Higher ↑ |
| `Ray_Turi` | [0, ∞) | Lower ↓ |
| `RMSSTD` | [0, ∞) | Lower ↓ |
| `R_squared` | [0, 1] | Higher ↑ |
| `Wemmert_Gancarski` | [0, 1] | Higher ↑ |

#### Tier 2 CVIs (14)
| Column | Range | Direction |
|--------|-------|-----------|
| `CS_Index` | [0, ∞) | Lower ↓ |
| `COP` | [0, ∞) | Lower ↓ |
| `S_Dbw` | [0, ∞) | Lower ↓ |
| `Det_Ratio` | [0, ∞) | Higher ↑ |
| `Gamma` | [-1, 1] | Higher ↑ |
| `Generalized_Dunn` | [0, ∞) | Higher ↑ |
| `G_plus` | [0, 1] | Lower ↓ |
| `I_Index` | [0, ∞) | Higher ↑ |
| `Log_Det_Ratio` | (-∞, ∞) | Higher ↑ |
| `McClain_Rao` | [0, ∞) | Lower ↓ |
| `Point_Biserial` | [-1, 1] | Higher ↑ |
| `SD_Validity` | [0, ∞) | Lower ↓ |
| `Tau` | [-1, 1] | Higher ↑ |
| `Trace_WiB` | [0, ∞) | Higher ↑ |
| `Ksq_DetW` | [0, ∞) | Lower ↓ |

#### Tier 3 CVIs (5)
| Column | Range | Direction |
|--------|-------|-----------|
| `Banfield_Raftery` | (-∞, ∞) | Lower ↓ |
| `Negentropy` | [0, ∞) | Higher ↑ |
| `NIVA` | [0, ∞) | Lower ↓ |
| `Score_Function` | [0, ∞) | Higher ↑ |
| `Scott_Symons` | [0, ∞) | Lower ↓ |

#### External Metrics (4) - *Excluded by default*
| Column | Range | Direction |
|--------|-------|-----------|
| `ARI` | [-1, 1] | Higher ↑ |
| `Van_Dongen` | [0, 1] | Lower ↓ |
| `VI` | [0, log(n)] | Lower ↓ |
| `Omega` | [0, 1] | Lower ↓ |

### Metadata Columns

| Column | Type | Description |
|--------|------|-------------|
| `metrics_computed` | int | Count of successfully computed metrics |
| `metrics_failed` | str | Comma-separated list of failed metric names |

## Example Output

```csv
clustering_name,topic,n_clusters,n_samples,quality,quality_score,degradation_type,degradation_level,Silhouette,Silhouette_norm,Davies_Bouldin,Davies_Bouldin_norm,...,metrics_computed,metrics_failed
narrative_17clusters,narrative,17,1247,1,1.0,,,0.452,0.85,0.831,0.72,...,30,
narrative_17clusters,narrative,17,1247,0,0.95,label_swap_random,5pct,0.438,0.78,0.912,0.65,...,30,
narrative_17clusters,narrative,17,1247,0,0.90,label_swap_random,10pct,0.421,0.71,0.983,0.58,...,30,
narrative_17clusters,narrative,16,1247,0,0.90,merge_nearest,10pct,0.395,0.62,1.021,0.52,...,30,
...
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Input CSVs                               │
│  (clustering_id, embeddings, [optional: other cols])        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  generate_training_data()                   │
│                                                             │
│  1. Load CSV                                                │
│  2. Evaluate original → record(quality=1)                   │
│  3. For each degradation:                                   │
│     a. Apply degradation                                    │
│     b. Evaluate degraded → record(quality=0)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 TrainingDataResult                          │
│                                                             │
│  - records: list[dict]                                      │
│  - to_dataframe() → pd.DataFrame                            │
│  - to_csv(path)                                             │
│  - to_parquet(path)                                         │
│  - summary() → str                                          │
└─────────────────────────────────────────────────────────────┘
```

## Dataset Statistics (Expected)

For a single clustering file with default settings:

| Component | Count |
|-----------|-------|
| Original | 1 |
| Degradation types | 19 |
| Levels per type | 4 |
| Degraded versions | 76 |
| **Total rows** | **77** |

For batch processing N files:
- **Total rows** = N × 77

## Column Naming Convention

- Metric names are converted to valid Python/DataFrame column names
- Spaces → underscores: `"Davies-Bouldin"` → `"Davies_Bouldin"`
- Hyphens → underscores: `"Ray-Turi"` → `"Ray_Turi"`
- Original casing preserved where possible
