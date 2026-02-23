"""Core evaluation functionality."""

from metricate.core.evaluator import (
    calculate_all_metrics,
    evaluate,
    filter_noise_points,
)
from metricate.core.exceptions import (
    ColumnNotFoundError,
    ComputationError,
    DimensionMismatchError,
    FileNotFoundError,
    InsufficientClustersError,
    InvalidCSVError,
    InvalidMetricError,
    MetricateError,
)
from metricate.core.loader import (
    ClusteringData,
    detect_columns,
    load_comparison_pair,
    load_csv,
)
from metricate.core.metrics import (
    EXTERNAL_METRIC_FUNCTIONS,
    METRIC_FUNCTIONS,
    compute_cluster_stats,
    compute_concordance_pairs,
    compute_scatter_matrices,
    precompute_all,
)
from metricate.core.reference import (
    LARGE_DATASET_SKIP,
    LARGE_DATASET_THRESHOLD,
    METRIC_REFERENCE,
    REDUNDANT_METRICS,
    get_metric_info,
)

__all__ = [
    # Reference
    "METRIC_REFERENCE",
    "LARGE_DATASET_THRESHOLD",
    "LARGE_DATASET_SKIP",
    "REDUNDANT_METRICS",
    "get_metric_info",
    # Exceptions
    "MetricateError",
    "FileNotFoundError",
    "InvalidCSVError",
    "ColumnNotFoundError",
    "InsufficientClustersError",
    "DimensionMismatchError",
    "InvalidMetricError",
    "ComputationError",
    # Loader
    "ClusteringData",
    "load_csv",
    "detect_columns",
    "load_comparison_pair",
    # Metrics
    "METRIC_FUNCTIONS",
    "EXTERNAL_METRIC_FUNCTIONS",
    "precompute_all",
    "compute_cluster_stats",
    "compute_scatter_matrices",
    "compute_concordance_pairs",
    # Evaluator
    "evaluate",
    "calculate_all_metrics",
    "filter_noise_points",
]
