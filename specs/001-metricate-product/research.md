# Research: Metricate Implementation

**Feature**: 001-metricate-product  
**Date**: 2026-02-23  
**Status**: Complete

## Executive Summary

All implementation unknowns have been resolved through codebase analysis. The existing `clustering_metrics_evaluation.ipynb` contains all 34 metric implementations ready for extraction. The `degradation_toolkit.py` provides the complete degradation functionality. No external research was needed.

## Research Tasks Completed

### 1. Metric Implementation Extraction

**Task**: Identify all metric functions and their dependencies in the notebook.

**Findings**:
- **Location**: Cells in Part 1 (helpers) and Part 6 (CVIs)
- **Total Functions**: 34 metric functions + 4 helper functions
- **Helper Functions Required**:
  - `compute_cluster_stats()` - centroids, sizes, WGSS/BGSS/TSS
  - `compute_scatter_matrices()` - W, B, T matrices
  - `compute_concordance_pairs()` - S+, S-, Nw, Nb for Gamma-family
  - `pairwise_distances()` - from sklearn.metrics

**Code Pattern** (extracted from notebook):
```python
def calculate_all_metrics(df, embedding_cols, label_col='cluster_id', original_labels=None):
    """Calculate ALL clustering metrics for a dataset."""
    X_all = df[embedding_cols].values
    labels_all = df[label_col].values
    
    # Filter out noise points (label = -1)
    non_noise_mask = labels_all != -1
    X = X_all[non_noise_mask]
    labels = labels_all[non_noise_mask]
    
    # Shared precomputation
    centroids, sizes, wgss_pc, wgss, bgss, tss, gmean = compute_cluster_stats(X, labels)
    W, B, T, W_pc, _ = compute_scatter_matrices(X, labels)
    dm = pairwise_distances(X)
    Sp, Sm, Nw, Nb = compute_concordance_pairs(dm, labels)
    
    pre = dict(centroids=centroids, sizes=sizes, ...)
    
    # Calculate each metric with precomputed values
    metrics['Silhouette'] = silhouette_score(X, labels)
    metrics['Ball-Hall'] = ball_hall(X, labels, **pre)
    # ... etc
```

**Decision**: Extract all functions preserving the precomputation pattern for efficiency.

---

### 2. O(n²) Metric Identification

**Task**: Identify which metrics have O(n²) complexity for auto-skip logic.

**Findings** (from notebook complexity column):

| Metric | Complexity | Reason |
|--------|-----------|--------|
| Silhouette | O(n²) | Pairwise distances |
| Dunn Index | O(n²) | Min inter-cluster distance |
| Gamma | O(n²) | All pairwise comparisons |
| Tau | O(n²) | All pairwise comparisons |
| Point-Biserial | O(n²) | Distance-label correlation |
| G-plus | O(n²) | Discordant pairs |
| McClain-Rao | O(n²) | Mean within/between distances |
| NIVA | O(n²) | Nearest neighbor ratios |
| CS index | O(n²) | Max nearest intra |
| COP | O(n²) | Per-point centroid vs max |
| S_Dbw | O(n²) | Inter-cluster density |
| Generalized Dunn | O(n²) | Centroid-based variant |

**Decision**: Auto-skip the 6 most expensive (Gamma, Tau, Point-Biserial, G-plus, McClain-Rao, NIVA) above 50k rows as specified in FR-004a. Keep Silhouette (sklearn optimized) and others with --force-all.

---

### 3. Degradation Toolkit Analysis

**Task**: Assess reusability of existing degradation_toolkit.py.

**Findings**:
- **Class**: `ClusteringDegrader` (708 lines)
- **Methods**: 19 degradation types all implemented
- **Output**: CSV files + manifest
- **Visualization**: Not included (separate script)

**Available Degradation Methods**:
```python
# Label modifications
label_swap(fraction, swap_type='random'|'neighboring'|'distant')
boundary_reassignment(fraction)

# Cluster structure
merge_clusters(method='nearest'|'farthest'|'random')
split_cluster(method='largest'|'loosest'|'random')

# Point removal  
random_removal(fraction)
core_removal(fraction)  # removes points closest to centroids
remove_tight_clusters(n_clusters, criteria='tightest'|'smallest'|'largest')

# Embedding perturbation
noise_injection(fraction, std_factor)
embedding_perturbation(fraction, magnitude)
centroid_displacement(fraction, magnitude)
```

**Decision**: Import `ClusteringDegrader` directly, add thin wrapper for CLI integration.

---

### 4. Visualization Pattern Analysis

**Task**: Determine visualization approach for degradation HTMLs.

**Findings** (from existing code):
- Uses Plotly for all interactive visualizations
- 2D scatter plots with UMAP/PCA reduced embeddings
- Color-coded by cluster or degradation type
- Dropdown menus for metric/degradation selection
- Index.html with links to individual visualizations

**Existing Pattern** (from create_degradation_visualizations.py):
```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    mode='markers',
    marker=dict(color=cluster_colors, size=5),
    text=hover_text,
    hoverinfo='text'
))
fig.write_html(output_path)
```

**Decision**: Reuse Plotly pattern, parameterize for different degradation types.

---

### 5. Output Format Analysis

**Task**: Design output format for metric results.

**Findings** (from notebook):
- Current output: dict → DataFrame
- Existing column structure: metric_name as key, value as float
- Direction info stored separately in `ALL_METRIC_DIRECTIONS`

**Proposed DataFrame Structure**:
```
| metric_name       | value    | range      | direction | tier     |
|-------------------|----------|------------|-----------|----------|
| Silhouette        | 0.4521   | [-1, 1]    | ↑ Higher  | Original |
| Davies-Bouldin    | 1.2341   | [0, ∞)     | ↓ Lower   | Original |
| ...               | ...      | ...        | ...       | ...      |
```

**Decision**: Return pandas DataFrame with columns: metric_name, value, range, direction, tier. Formatters convert to table/JSON/CSV.

---

## Metric Reference Data

Complete reference for `metricate/core/reference.py`:

```python
METRIC_REFERENCE = {
    # Internal Original (6)
    'Silhouette': {'range': '[-1, 1]', 'direction': 'higher', 'tier': 'Original', 'complexity': 'O(n²)', 'skip_large': False},
    'Davies-Bouldin': {'range': '[0, ∞)', 'direction': 'lower', 'tier': 'Original', 'complexity': 'O(n)', 'skip_large': False},
    'Calinski-Harabasz': {'range': '[0, ∞)', 'direction': 'higher', 'tier': 'Original', 'complexity': 'O(n)', 'skip_large': False},
    'Dunn Index': {'range': '[0, ∞)', 'direction': 'higher', 'tier': 'Original', 'complexity': 'O(n²)', 'skip_large': False},
    'SSE': {'range': '[0, ∞)', 'direction': 'lower', 'tier': 'Original', 'complexity': 'O(n)', 'skip_large': False},
    'NCI': {'range': '[-1, 1]', 'direction': 'higher', 'tier': 'Original', 'complexity': 'O(n)', 'skip_large': False},
    
    # Tier 1 CVIs (6)
    'Ball-Hall': {'range': '[0, ∞)', 'direction': 'lower', 'tier': 'Tier 1', 'complexity': 'O(n·d)', 'skip_large': False},
    'Ratkowsky-Lance': {'range': '[0, 1]', 'direction': 'higher', 'tier': 'Tier 1', 'complexity': 'O(n·d)', 'skip_large': False},
    'Ray-Turi': {'range': '[0, ∞)', 'direction': 'lower', 'tier': 'Tier 1', 'complexity': 'O(n·d)', 'skip_large': False},
    'RMSSTD': {'range': '[0, ∞)', 'direction': 'lower', 'tier': 'Tier 1', 'complexity': 'O(n·d)', 'skip_large': False},
    'R-squared': {'range': '[0, 1]', 'direction': 'higher', 'tier': 'Tier 1', 'complexity': 'O(n·d)', 'skip_large': False},
    'Wemmert-Gancarski': {'range': '[0, 1]', 'direction': 'higher', 'tier': 'Tier 1', 'complexity': 'O(n·k·d)', 'skip_large': False},
    
    # Tier 2 CVIs (14) - 6 have skip_large=True
    'CS index': {'range': '[0, ∞)', 'direction': 'lower', 'tier': 'Tier 2', 'complexity': 'O(n²)', 'skip_large': False},
    'COP': {'range': '[0, ∞)', 'direction': 'lower', 'tier': 'Tier 2', 'complexity': 'O(n²)', 'skip_large': False},
    'S_Dbw': {'range': '[0, ∞)', 'direction': 'lower', 'tier': 'Tier 2', 'complexity': 'O(n²)', 'skip_large': False},
    'Det Ratio': {'range': '[0, ∞)', 'direction': 'higher', 'tier': 'Tier 2', 'complexity': 'O(n·d²)', 'skip_large': False},
    'Gamma': {'range': '[-1, 1]', 'direction': 'higher', 'tier': 'Tier 2', 'complexity': 'O(n²)', 'skip_large': True},
    'Generalized Dunn': {'range': '[0, ∞)', 'direction': 'higher', 'tier': 'Tier 2', 'complexity': 'O(n²)', 'skip_large': False},
    'G-plus': {'range': '[0, 1]', 'direction': 'lower', 'tier': 'Tier 2', 'complexity': 'O(n²)', 'skip_large': True},
    'I-index (PBM)': {'range': '[0, ∞)', 'direction': 'higher', 'tier': 'Tier 2', 'complexity': 'O(n·d)', 'skip_large': False},
    'Log_Det_Ratio': {'range': '(-∞, ∞)', 'direction': 'higher', 'tier': 'Tier 2', 'complexity': 'O(n·d²)', 'skip_large': False},
    'McClain-Rao': {'range': '[0, ∞)', 'direction': 'lower', 'tier': 'Tier 2', 'complexity': 'O(n²)', 'skip_large': True},
    'Point-Biserial': {'range': '[-1, 1]', 'direction': 'higher', 'tier': 'Tier 2', 'complexity': 'O(n²)', 'skip_large': True},
    'SD validity': {'range': '[0, ∞)', 'direction': 'lower', 'tier': 'Tier 2', 'complexity': 'O(n·d)', 'skip_large': False},
    'Tau': {'range': '[-1, 1]', 'direction': 'higher', 'tier': 'Tier 2', 'complexity': 'O(n²)', 'skip_large': True},
    'Trace_WiB': {'range': '[0, ∞)', 'direction': 'higher', 'tier': 'Tier 2', 'complexity': 'O(n·d²)', 'skip_large': False},
    'Ksq_DetW': {'range': '[0, ∞)', 'direction': 'lower', 'tier': 'Tier 2', 'complexity': 'O(n·d²)', 'skip_large': False},
    
    # Tier 3 CVIs (5)
    'Banfield-Raftery': {'range': '(-∞, ∞)', 'direction': 'lower', 'tier': 'Tier 3', 'complexity': 'O(n·d²)', 'skip_large': False},
    'Negentropy': {'range': '(-∞, ∞)', 'direction': 'lower', 'tier': 'Tier 3', 'complexity': 'O(n·d²)', 'skip_large': False},
    'NIVA': {'range': '[0, ∞)', 'direction': 'lower', 'tier': 'Tier 3', 'complexity': 'O(n²)', 'skip_large': True},
    'Score Function': {'range': '[0, 1]', 'direction': 'higher', 'tier': 'Tier 3', 'complexity': 'O(n·d)', 'skip_large': False},
    'Scott-Symons': {'range': '(-∞, ∞)', 'direction': 'lower', 'tier': 'Tier 3', 'complexity': 'O(n·d²)', 'skip_large': False},
    
    # External (4)
    'Adjusted Rand Index': {'range': '[-1, 1]', 'direction': 'higher', 'tier': 'External', 'complexity': 'O(n)', 'skip_large': False},
    'Van Dongen': {'range': '[0, 1]', 'direction': 'lower', 'tier': 'External', 'complexity': 'O(n)', 'skip_large': False},
    'Variation of Information': {'range': '[0, log(n)]', 'direction': 'lower', 'tier': 'External', 'complexity': 'O(n)', 'skip_large': False},
    'Omega': {'range': '[0, 1]', 'direction': 'lower', 'tier': 'External', 'complexity': 'O(n)', 'skip_large': False},
}

# Metrics to auto-skip above 50k rows (FR-004a)
LARGE_DATASET_SKIP = ['Gamma', 'Tau', 'Point-Biserial', 'G-plus', 'McClain-Rao', 'NIVA']

# Redundant metrics (excluded by default)
REDUNDANT_METRICS = ['Trace_W', 'Baker-Hubert Gamma', 'Sym-index', 'Rand Index', 'Log_SS_Ratio']
```

## Conclusion

All research complete. No blockers identified. Ready to proceed to Phase 1 design and implementation.
