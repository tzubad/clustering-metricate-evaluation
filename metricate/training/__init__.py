"""
Training dataset generation module for metricate.

This module provides functions to generate labeled training datasets
from clustering CSVs by systematically degrading good clusterings
and calculating metrics for both original and degraded versions.

Example:
    >>> import metricate
    >>> result = metricate.generate_training_data("clustering.csv", "./output/")
    >>> df = result.to_dataframe()
    >>> result.to_csv("training_dataset.csv")
"""

from metricate.training.generator import (
    generate_training_data,
    generate_training_data_batch,
)
from metricate.training.result import TrainingDataResult

__all__ = [
    "generate_training_data",
    "generate_training_data_batch",
    "TrainingDataResult",
]
