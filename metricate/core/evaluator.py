"""
Core evaluation engine for calculating clustering metrics.

This module provides the main entry point for evaluating clustering quality
using the 34 implemented metrics.
"""

from pathlib import Path

import numpy as np

from metricate.core.exceptions import (
    ComputationError,
    InsufficientClustersError,
)
from metricate.core.loader import load_csv
from metricate.core.metrics import (
    METRIC_FUNCTIONS,
    precompute_all,
)
from metricate.core.reference import (
    LARGE_DATASET_SKIP,
    LARGE_DATASET_THRESHOLD,
)
from metricate.output.report import EvaluationResult, MetricValue


def filter_noise_points(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Remove noise points (cluster_id=-1) from the data.

    Args:
        embeddings: Embedding matrix (n_samples, n_features)
        labels: Cluster labels array

    Returns:
        Tuple of (filtered_embeddings, filtered_labels, noise_count)
    """
    noise_mask = labels == -1
    noise_count = noise_mask.sum()

    if noise_count > 0:
        valid_mask = ~noise_mask
        return embeddings[valid_mask], labels[valid_mask], int(noise_count)

    return embeddings, labels, 0


def calculate_all_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    exclude: list[str] | None = None,
    force_all: bool = False,
) -> EvaluationResult:
    """
    Calculate all internal clustering metrics for the given data.

    This function handles precomputation of shared values and calls each
    metric function with the appropriate parameters.

    Args:
        X: Embedding matrix (n_samples, n_features)
        labels: Cluster labels array
        exclude: List of metric names to skip
        force_all: If True, compute O(n²) metrics even on large datasets

    Returns:
        EvaluationResult containing all computed metrics
    """
    exclude = set(exclude or [])
    result = EvaluationResult()

    n_samples = len(labels)
    n_clusters = len(np.unique(labels))

    # Add metadata
    result.metadata = {
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "n_clusters": n_clusters,
    }

    # Check if we should auto-skip expensive metrics
    auto_skip = set()
    if n_samples > LARGE_DATASET_THRESHOLD and not force_all:
        auto_skip = set(LARGE_DATASET_SKIP)
        result.add_warning(
            f"Dataset has {n_samples:,} rows (>{LARGE_DATASET_THRESHOLD:,}). "
            f"Auto-skipping O(n²) metrics: {', '.join(sorted(auto_skip))}. "
            "Use force_all=True to override."
        )

    # Warn about unrecognized exclusions
    all_known = set(METRIC_FUNCTIONS.keys())
    unknown = exclude - all_known
    if unknown:
        result.add_warning(
            f"Unrecognized metric names in exclude list: {', '.join(sorted(unknown))}"
        )

    # Precompute shared values
    try:
        precomputed = precompute_all(X, labels)
    except Exception as e:
        raise ComputationError(
            metric="precomputation", message=f"Failed to precompute shared values: {e}"
        )

    # Calculate each metric
    for metric_name, metric_func in METRIC_FUNCTIONS.items():
        # Check if excluded
        if metric_name in exclude:
            result.add_metric(
                MetricValue(
                    metric=metric_name,
                    value=None,
                    computed=False,
                    skip_reason="User excluded",
                )
            )
            continue

        # Check if auto-skipped
        if metric_name in auto_skip:
            result.add_metric(
                MetricValue(
                    metric=metric_name,
                    value=None,
                    computed=False,
                    skip_reason=f"Auto-skipped (n > {LARGE_DATASET_THRESHOLD:,})",
                )
            )
            continue

        # Calculate the metric
        try:
            # Some metrics don't need X/labels (e.g., gamma_index uses precomputed S_plus/S_minus)
            if metric_name in ("Gamma", "G-plus", "Tau"):
                value = metric_func(**precomputed)
            else:
                value = metric_func(X, labels, **precomputed)

            # Handle inf/nan
            if np.isnan(value) or np.isinf(value):
                result.add_metric(
                    MetricValue(
                        metric=metric_name,
                        value=None,
                        computed=False,
                        skip_reason=f"Computation returned {value}",
                    )
                )
            else:
                result.add_metric(
                    MetricValue(
                        metric=metric_name,
                        value=float(value),
                        computed=True,
                    )
                )
        except Exception as e:
            result.add_metric(
                MetricValue(
                    metric=metric_name,
                    value=None,
                    computed=False,
                    skip_reason=f"Error: {str(e)[:50]}",
                )
            )

    return result


def evaluate(
    csv_path: str | Path,
    *,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
    exclude: list[str] | None = None,
    force_all: bool = False,
) -> EvaluationResult:
    """
    Evaluate clustering quality for a CSV file.

    This is the main public API for single-file evaluation.

    Args:
        csv_path: Path to the CSV file containing clustering data
        label_col: Name of the cluster label column (auto-detected if not provided)
        embedding_cols: List of embedding column names (auto-detected if not provided)
        exclude: List of metric names to skip
        force_all: If True, compute O(n²) metrics even on large datasets (>50k rows)

    Returns:
        EvaluationResult containing all computed metrics with ranges and directions

    Raises:
        FileNotFoundError: If CSV file does not exist
        InvalidCSVError: If CSV is malformed
        ColumnNotFoundError: If required columns not found
        InsufficientClustersError: If fewer than 2 clusters

    Example:
        >>> result = metricate.evaluate("clustering.csv")
        >>> print(result.to_table())
        >>> df = result.to_dataframe()
    """
    # Load and validate data
    data = load_csv(csv_path, label_col=label_col, embedding_cols=embedding_cols)

    # Filter noise points
    X, labels, noise_count = filter_noise_points(data.embeddings, data.labels)

    # Check we still have enough clusters after noise removal
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        raise InsufficientClustersError(
            required=2,
            actual=n_clusters,
            message="After removing noise points, fewer than 2 clusters remain",
        )

    # Calculate metrics
    result = calculate_all_metrics(X, labels, exclude=exclude, force_all=force_all)

    # Add file metadata
    result.metadata["source_file"] = str(csv_path)
    result.metadata["label_col"] = data.label_col
    result.metadata["embedding_cols"] = data.embedding_cols

    # Add noise warning if applicable
    if noise_count > 0:
        result.add_warning(f"Excluded {noise_count:,} noise points (cluster_id=-1) from evaluation")
        result.metadata["noise_points_excluded"] = noise_count

    return result
