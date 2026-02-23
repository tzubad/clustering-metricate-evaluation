"""Metric reference data: ranges, directions, tiers, and complexity information."""

# Complete reference for all 34 active metrics
METRIC_REFERENCE = {
    # ========== Internal Original (6) ==========
    "Silhouette": {
        "range": "[-1, 1]",
        "direction": "higher",
        "tier": "Original",
        "complexity": "O(n²)",
        "skip_large": False,
        "description": "Average silhouette coefficient across all samples",
    },
    "Davies-Bouldin": {
        "range": "[0, ∞)",
        "direction": "lower",
        "tier": "Original",
        "complexity": "O(n)",
        "skip_large": False,
        "description": "Average similarity ratio of each cluster with its most similar cluster",
    },
    "Calinski-Harabasz": {
        "range": "[0, ∞)",
        "direction": "higher",
        "tier": "Original",
        "complexity": "O(n)",
        "skip_large": False,
        "description": "Ratio of between-cluster dispersion to within-cluster dispersion",
    },
    "Dunn Index": {
        "range": "[0, ∞)",
        "direction": "higher",
        "tier": "Original",
        "complexity": "O(n²)",
        "skip_large": False,
        "description": "Ratio of min inter-cluster distance to max intra-cluster diameter",
    },
    "SSE": {
        "range": "[0, ∞)",
        "direction": "lower",
        "tier": "Original",
        "complexity": "O(n)",
        "skip_large": False,
        "description": "Sum of squared errors (within-cluster sum of squares)",
    },
    "NCI": {
        "range": "[-1, 1]",
        "direction": "higher",
        "tier": "Original",
        "complexity": "O(n)",
        "skip_large": False,
        "description": "New Correlation Index: correlation between point-centroid and centroid-global distances",
    },
    # ========== Tier 1 CVIs (6) - Centroid-based, O(n·d) ==========
    "Ball-Hall": {
        "range": "[0, ∞)",
        "direction": "lower",
        "tier": "Tier 1",
        "complexity": "O(n·d)",
        "skip_large": False,
        "description": "Mean of per-cluster mean dispersion",
    },
    "Ratkowsky-Lance": {
        "range": "[0, 1]",
        "direction": "higher",
        "tier": "Tier 1",
        "complexity": "O(n·d)",
        "skip_large": False,
        "description": "Per-feature BGSS/TSS ratio",
    },
    "Ray-Turi": {
        "range": "[0, ∞)",
        "direction": "lower",
        "tier": "Tier 1",
        "complexity": "O(n·d)",
        "skip_large": False,
        "description": "Mean squared distance / min centroid distance²",
    },
    "RMSSTD": {
        "range": "[0, ∞)",
        "direction": "lower",
        "tier": "Tier 1",
        "complexity": "O(n·d)",
        "skip_large": False,
        "description": "Root mean square standard deviation",
    },
    "R-squared": {
        "range": "[0, 1]",
        "direction": "higher",
        "tier": "Tier 1",
        "complexity": "O(n·d)",
        "skip_large": False,
        "description": "Between-group SS / Total SS",
    },
    "Wemmert-Gancarski": {
        "range": "[0, 1]",
        "direction": "higher",
        "tier": "Tier 1",
        "complexity": "O(n·k·d)",
        "skip_large": False,
        "description": "Centroid membership quality index",
    },
    # ========== Tier 2 CVIs (14) - Scatter-matrix or pairwise-distance ==========
    "CS index": {
        "range": "[0, ∞)",
        "direction": "lower",
        "tier": "Tier 2",
        "complexity": "O(n²)",
        "skip_large": False,
        "description": "Compact-Separated: max intra-nearest / min centroid dist",
    },
    "COP": {
        "range": "[0, ∞)",
        "direction": "lower",
        "tier": "Tier 2",
        "complexity": "O(n²)",
        "skip_large": False,
        "description": "Mean of centroid dist / max intra dist",
    },
    "S_Dbw": {
        "range": "[0, ∞)",
        "direction": "lower",
        "tier": "Tier 2",
        "complexity": "O(n²)",
        "skip_large": False,
        "description": "Scatter + inter-cluster density",
    },
    "Det Ratio": {
        "range": "[0, ∞)",
        "direction": "higher",
        "tier": "Tier 2",
        "complexity": "O(n·d²)",
        "skip_large": False,
        "description": "det(Total) / det(Within)",
    },
    "Gamma": {
        "range": "[-1, 1]",
        "direction": "higher",
        "tier": "Tier 2",
        "complexity": "O(n²)",
        "skip_large": True,
        "description": "Concordant-discordant pair ratio",
    },
    "Generalized Dunn": {
        "range": "[0, ∞)",
        "direction": "higher",
        "tier": "Tier 2",
        "complexity": "O(n²)",
        "skip_large": False,
        "description": "Centroid-based inter / max diameter",
    },
    "G-plus": {
        "range": "[0, 1]",
        "direction": "lower",
        "tier": "Tier 2",
        "complexity": "O(n²)",
        "skip_large": True,
        "description": "Normalized discordant pairs",
    },
    "I-index (PBM)": {
        "range": "[0, ∞)",
        "direction": "higher",
        "tier": "Tier 2",
        "complexity": "O(n·d)",
        "skip_large": False,
        "description": "Composite: (1/K × E_T/E_W × D_B)²",
    },
    "Log_Det_Ratio": {
        "range": "(-∞, ∞)",
        "direction": "higher",
        "tier": "Tier 2",
        "complexity": "O(n·d²)",
        "skip_large": False,
        "description": "N × log(det(T)/det(W))",
    },
    "McClain-Rao": {
        "range": "[0, ∞)",
        "direction": "lower",
        "tier": "Tier 2",
        "complexity": "O(n²)",
        "skip_large": True,
        "description": "Mean within / mean between distance",
    },
    "Point-Biserial": {
        "range": "[-1, 1]",
        "direction": "higher",
        "tier": "Tier 2",
        "complexity": "O(n²)",
        "skip_large": True,
        "description": "Distance-cluster correlation",
    },
    "SD validity": {
        "range": "[0, ∞)",
        "direction": "lower",
        "tier": "Tier 2",
        "complexity": "O(n·d)",
        "skip_large": False,
        "description": "Scatter + centroid distance measure",
    },
    "Tau": {
        "range": "[-1, 1]",
        "direction": "higher",
        "tier": "Tier 2",
        "complexity": "O(n²)",
        "skip_large": True,
        "description": "Normalized concordance index",
    },
    "Trace_WiB": {
        "range": "[0, ∞)",
        "direction": "higher",
        "tier": "Tier 2",
        "complexity": "O(n·d²)",
        "skip_large": False,
        "description": "trace(W⁻¹B): separation relative to compactness",
    },
    "Ksq_DetW": {
        "range": "[0, ∞)",
        "direction": "lower",
        "tier": "Tier 2",
        "complexity": "O(n·d²)",
        "skip_large": False,
        "description": "K² × det(W)",
    },
    # ========== Tier 3 CVIs (4) - Per-cluster covariance or specialized ==========
    "Banfield-Raftery": {
        "range": "(-∞, ∞)",
        "direction": "lower",
        "tier": "Tier 3",
        "complexity": "O(n·d²)",
        "skip_large": False,
        "description": "Sum of n_j × log(trace(W_j)/n_j)",
    },
    "Negentropy": {
        "range": "(-∞, ∞)",
        "direction": "lower",
        "tier": "Tier 3",
        "complexity": "O(n·d²)",
        "skip_large": False,
        "description": "Cluster entropy relative to Gaussian",
    },
    "NIVA": {
        "range": "[0, ∞)",
        "direction": "lower",
        "tier": "Tier 3",
        "complexity": "O(n²)",
        "skip_large": True,
        "description": "Nearest intra/inter distance ratio",
    },
    "Score Function": {
        "range": "[0, 1]",
        "direction": "higher",
        "tier": "Tier 3",
        "complexity": "O(n·d)",
        "skip_large": False,
        "description": "1 - 1/exp(exp(bdc-wcd))",
    },
    "Scott-Symons": {
        "range": "(-∞, ∞)",
        "direction": "lower",
        "tier": "Tier 3",
        "complexity": "O(n·d²)",
        "skip_large": False,
        "description": "Sum of n_j × log(det(W_j/n_j))",
    },
    # ========== External (4) ==========
    "Adjusted Rand Index": {
        "range": "[-1, 1]",
        "direction": "higher",
        "tier": "External",
        "complexity": "O(n)",
        "skip_large": False,
        "description": "Chance-corrected Rand Index",
    },
    "Van Dongen": {
        "range": "[0, 1]",
        "direction": "lower",
        "tier": "External",
        "complexity": "O(n)",
        "skip_large": False,
        "description": "1 - (row_max_sum + col_max_sum) / 2n",
    },
    "Variation of Information": {
        "range": "[0, log(n)]",
        "direction": "lower",
        "tier": "External",
        "complexity": "O(n)",
        "skip_large": False,
        "description": "H(U|V) + H(V|U)",
    },
    "Omega": {
        "range": "[0, 1]",
        "direction": "lower",
        "tier": "External",
        "complexity": "O(n)",
        "skip_large": False,
        "description": "Composite: mean([1-ARI], VD, VI_normalized)",
    },
}


# Metrics to auto-skip above 50k rows (FR-004a)
LARGE_DATASET_SKIP = ["Gamma", "Tau", "Point-Biserial", "G-plus", "McClain-Rao", "NIVA"]

# Threshold for auto-skipping expensive metrics
LARGE_DATASET_THRESHOLD = 50_000

# Redundant metrics (excluded by default)
REDUNDANT_METRICS = ["Trace_W", "Baker-Hubert Gamma", "Sym-index", "Rand Index", "Log_SS_Ratio"]

# Internal metrics (do not require ground truth labels)
INTERNAL_METRICS = [name for name, info in METRIC_REFERENCE.items() if info["tier"] != "External"]

# External metrics (require ground truth labels)
EXTERNAL_METRICS = [name for name, info in METRIC_REFERENCE.items() if info["tier"] == "External"]


def get_metric_info(name: str) -> dict | None:
    """Get reference information for a metric by name."""
    return METRIC_REFERENCE.get(name)


def get_direction_symbol(direction: str) -> str:
    """Convert direction to display symbol."""
    return "↑ Higher" if direction == "higher" else "↓ Lower"


def format_metric_value(value: float, precision: int = 4) -> str:
    """Format metric value for display."""
    if value is None or (isinstance(value, float) and (value != value)):  # NaN check
        return "N/A"
    return f"{value:.{precision}f}"
