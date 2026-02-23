"""
Metricate: A clustering evaluation toolkit.

Evaluate clustering quality with 34 metrics, compare clusterings,
and generate degraded datasets for testing metric robustness.

Example:
    >>> import metricate
    >>> result = metricate.evaluate("clustering.csv")
    >>> print(result.to_table())

    >>> # Compare two clusterings
    >>> comparison = metricate.compare("v1.csv", "v2.csv")
    >>> print(f"Winner: {comparison.winner}")

    >>> # Generate degraded datasets
    >>> result = metricate.degrade("data.csv", "./output/")
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

__version__ = "0.1.0"


def evaluate(
    csv_path: str | Path,
    *,
    exclude: list[str] | None = None,
    force_all: bool = False,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
):
    """
    Evaluate a single clustering and return all metric scores.

    Computes up to 34 clustering quality metrics on the provided dataset.
    Automatically detects label and embedding columns if not specified.

    Args:
        csv_path: Path to CSV file containing clustering data.
        exclude: List of metric names to skip (e.g., ["Gamma", "Tau"]).
        force_all: If True, compute O(n²) metrics even for large datasets.
        label_col: Name of cluster label column (auto-detected if None).
        embedding_cols: List of embedding column names (auto-detected if None).

    Returns:
        EvaluationResult: Object containing metric scores with methods:
            - to_table(): Formatted string table
            - to_dataframe(): pandas DataFrame
            - to_json(): JSON string
            - to_csv(): CSV string

    Raises:
        FileNotFoundError: If csv_path does not exist.
        InvalidCSVError: If file is not valid CSV.
        ColumnNotFoundError: If required columns not found.
        InsufficientClustersError: If fewer than 2 clusters.

    Example:
        >>> result = metricate.evaluate("clustering.csv")
        >>> print(result.to_table())
        >>> df = result.to_dataframe()
    """
    from metricate.core.evaluator import evaluate as _evaluate

    return _evaluate(
        csv_path,
        exclude=exclude,
        force_all=force_all,
        label_col=label_col,
        embedding_cols=embedding_cols,
    )


def compare(
    csv_path_a: str | Path,
    csv_path_b: str | Path,
    *,
    exclude: list[str] | None = None,
    force_all: bool = False,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
    name_a: str = "A",
    name_b: str = "B",
):
    """
    Compare two clusterings and determine the winner.

    Evaluates both clusterings and determines which is better based on
    the majority of metrics. Each metric "votes" for the better clustering
    according to its direction preference (higher/lower is better).

    Args:
        csv_path_a: Path to first CSV file.
        csv_path_b: Path to second CSV file.
        exclude: List of metric names to skip.
        force_all: If True, compute O(n²) metrics even for large datasets.
        label_col: Name of cluster label column (auto-detected if None).
        embedding_cols: List of embedding column names (auto-detected if None).
        name_a: Display name for first clustering (default: "A").
        name_b: Display name for second clustering (default: "B").

    Returns:
        ComparisonResult: Object containing comparison with attributes:
            - winner: Name of the winning clustering
            - wins: Dict with win counts {"A": n, "B": m, "Tie": t}
            - metric_winners: Dict mapping metric name to winner
            - to_table(): Formatted comparison table
            - to_dataframe(): pandas DataFrame

    Example:
        >>> result = metricate.compare("v1.csv", "v2.csv")
        >>> print(f"Winner: {result.winner}")
        >>> print(result.to_table())
    """
    from metricate.comparison.compare import compare as _compare

    return _compare(
        csv_path_a,
        csv_path_b,
        exclude=exclude,
        force_all=force_all,
        label_col=label_col,
        embedding_cols=embedding_cols,
        name_a=name_a,
        name_b=name_b,
    )


def degrade(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    levels: list[str] | None = None,
    types: list[str] | None = None,
    visualize: bool = True,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
    random_seed: int = 42,
):
    """
    Generate degraded versions of a clustering dataset.

    Creates multiple degraded versions of the input clustering at various
    intensity levels using different degradation strategies. Useful for
    testing how metrics respond to systematic quality degradation.

    Args:
        csv_path: Path to input CSV with clustering data.
        output_dir: Directory to write degraded datasets.
        levels: Intensity levels (default: ["5pct", "10pct", "25pct", "50pct"]).
        types: Degradation types to apply (default: all 19 types).
        visualize: If True, generate HTML visualizations.
        label_col: Name of cluster label column (auto-detected if None).
        embedding_cols: List of embedding column names (auto-detected if None).
        random_seed: Random seed for reproducibility.

    Returns:
        DegradationResult: Object containing:
            - output_dir: Path to output directory
            - degradations: List of DegradationEntry objects
            - manifest_path: Path to manifest.json
            - csv_files: List of generated CSV paths
            - summary(): Text summary of generation

    Example:
        >>> result = metricate.degrade("clustering.csv", "./output/")
        >>> print(result.summary())
        >>> print(f"Generated {len(result.csv_files)} files")
    """
    from metricate.degradation.toolkit import degrade as _degrade

    return _degrade(
        csv_path,
        output_dir,
        levels=levels,
        types=types,
        visualize=visualize,
        label_col=label_col,
        embedding_cols=embedding_cols,
        random_seed=random_seed,
    )


def list_metrics(include_reference: bool = False) -> list[str] | pd.DataFrame:
    """
    List all available clustering metrics.

    Args:
        include_reference: If True, return DataFrame with full metadata
            (range, direction, tier, complexity). If False, return list of names.

    Returns:
        List of metric names, or DataFrame with columns:
            - metric: Metric name
            - range: Value range (e.g., "[-1, 1]")
            - direction: "higher" or "lower" is better
            - tier: Category (Original, Tier1, Tier2, Tier3, External)
            - complexity: Time complexity (O(n) or O(n²))

    Example:
        >>> metrics = metricate.list_metrics()
        >>> print(f"{len(metrics)} metrics available")
        >>> df = metricate.list_metrics(include_reference=True)
    """
    from metricate.core.reference import METRIC_REFERENCE

    if include_reference:
        data = []
        for name, info in METRIC_REFERENCE.items():
            data.append(
                {
                    "metric": name,
                    "range": info["range"],
                    "direction": info["direction"],
                    "tier": info["tier"],
                    "complexity": info["complexity"],
                }
            )
        return pd.DataFrame(data)
    return list(METRIC_REFERENCE.keys())


def list_degradations() -> dict[str, list[str]]:
    """
    List all available degradation types organized by category.

    Returns:
        Dict mapping category names to lists of degradation types:
            - Label Manipulation: label_swap_random, label_swap_neighboring, ...
            - Cluster Structure: merge_random, split_largest, ...
            - Point Manipulation: noise_injection, random_removal, ...
            - Cluster Removal: remove_smallest_clusters, ...
            - Embedding Manipulation: embedding_perturbation, ...

    Example:
        >>> types = metricate.list_degradations()
        >>> for category, degradations in types.items():
        ...     print(f"{category}: {len(degradations)} types")
    """
    from metricate.degradation.toolkit import DEGRADATION_TYPES

    return dict(DEGRADATION_TYPES)


__all__ = [
    "evaluate",
    "compare",
    "degrade",
    "list_metrics",
    "list_degradations",
    "__version__",
]
