"""
Percentile normalization for clustering metrics.

This module provides functions to normalize metric values to [0, 1] range
using percentile ranks, accounting for metric direction (higher/lower is better).
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from metricate.core.reference import METRIC_REFERENCE


def get_metric_direction(metric_name: str) -> str:
    """Get the direction for a metric (higher or lower is better).

    Args:
        metric_name: Name of the metric.

    Returns:
        "higher" or "lower" indicating which direction is better.
    """
    if metric_name in METRIC_REFERENCE:
        return METRIC_REFERENCE[metric_name].get("direction", "higher")
    return "higher"  # Default to higher is better


def percentile_normalize_column(
    values: np.ndarray, higher_is_better: bool = True
) -> np.ndarray:
    """Convert values to percentile ranks in [0, 1].

    Args:
        values: Array of metric values (may contain NaN).
        higher_is_better: If True, higher values get higher ranks.
            If False, lower values get higher ranks.

    Returns:
        Array of normalized values in [0, 1], with NaN preserved.
    """
    # Handle empty or all-NaN case
    valid_mask = ~np.isnan(values)
    n_valid = valid_mask.sum()

    if n_valid == 0:
        return np.full_like(values, np.nan)

    if n_valid == 1:
        # Single value: assign 0.5
        result = np.full_like(values, np.nan)
        result[valid_mask] = 0.5
        return result

    # Compute ranks for valid values only
    valid_values = values[valid_mask]
    ranks = rankdata(valid_values, method="average")

    # Normalize to [0, 1]
    normalized = (ranks - 1) / (n_valid - 1)

    # Invert if lower is better (so that "better" always maps to higher norm value)
    if not higher_is_better:
        normalized = 1.0 - normalized

    # Place back into result array
    result = np.full_like(values, np.nan, dtype=float)
    result[valid_mask] = normalized

    return result


def normalize_metrics(
    df: pd.DataFrame,
    metric_cols: list[str],
    suffix: str = "_norm",
) -> pd.DataFrame:
    """Add normalized columns for all metrics.

    Normalizes each metric column using percentile ranks, accounting for
    the metric's direction (higher/lower is better). The normalized values
    are always in [0, 1] where 1 = best.

    Args:
        df: DataFrame containing metric columns.
        metric_cols: List of metric column names to normalize.
        suffix: Suffix to add for normalized columns (default: "_norm").

    Returns:
        DataFrame with additional normalized columns.
    """
    df = df.copy()

    for col in metric_cols:
        if col not in df.columns:
            continue

        values = df[col].values.astype(float)
        direction = get_metric_direction(col)
        higher_is_better = direction == "higher"

        normalized = percentile_normalize_column(values, higher_is_better)
        df[f"{col}{suffix}"] = normalized

    return df


def get_internal_metric_names() -> list[str]:
    """Get list of internal (non-external) metric names.

    External metrics (ARI, Van Dongen, VI, Omega) require ground truth
    and are excluded from training data generation.

    Returns:
        List of internal metric names.
    """
    external = {"ARI", "Van Dongen", "VI", "Omega"}
    return [m for m in METRIC_REFERENCE.keys() if m not in external]
