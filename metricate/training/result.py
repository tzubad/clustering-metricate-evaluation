"""
Result container for training dataset generation.

This module provides the TrainingDataResult dataclass that holds
the generated training records and provides export methods.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class TrainingDataResult:
    """Container for training dataset generation results.

    Attributes:
        records: List of record dictionaries, one per clustering version.
        metadata: Generation metadata (timestamp, parameters, file counts).
        warnings: Non-fatal warnings encountered during generation.
        errors: Files that failed to process.
    """

    records: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Set default metadata if not provided."""
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()

    @property
    def n_originals(self) -> int:
        """Count of original clusterings processed."""
        return sum(1 for r in self.records if r.get("quality") == 1)

    @property
    def n_degraded(self) -> int:
        """Count of degraded versions generated."""
        return sum(1 for r in self.records if r.get("quality") == 0)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to pandas DataFrame with proper dtypes.

        Returns:
            DataFrame with all records, properly typed columns.
        """
        if not self.records:
            return pd.DataFrame()

        df = pd.DataFrame(self.records)

        # Set proper dtypes for known columns
        int_cols = ["quality", "n_clusters", "n_samples", "metrics_computed"]
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].astype("Int64")  # Nullable integer

        float_cols = ["quality_score"]
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        str_cols = [
            "clustering_name",
            "topic",
            "degradation_type",
            "degradation_level",
            "metrics_failed",
        ]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", "")
                df[col] = df[col].replace("None", "")

        return df

    def to_csv(self, path: str | Path) -> None:
        """Export to CSV file.

        Args:
            path: Output file path.
        """
        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def to_parquet(self, path: str | Path) -> None:
        """Export to Parquet file (requires pyarrow).

        Args:
            path: Output file path.

        Raises:
            ImportError: If pyarrow is not installed.
        """
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet export. "
                "Install with: pip install pyarrow"
            )

        df = self.to_dataframe()
        df.to_parquet(path, index=False)

    def summary(self) -> str:
        """Human-readable summary of generation results.

        Returns:
            Multi-line summary string.
        """
        lines = [
            "Training Dataset Generation Summary",
            "=" * 40,
            f"Total records: {len(self.records)}",
            f"  - Original clusterings: {self.n_originals}",
            f"  - Degraded versions: {self.n_degraded}",
        ]

        if self.metadata:
            lines.append("")
            lines.append("Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")

        if self.warnings:
            lines.append("")
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings[:5]:  # Show first 5
                lines.append(f"  - {w}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more")

        if self.errors:
            lines.append("")
            lines.append(f"Errors ({len(self.errors)}):")
            for e in self.errors[:5]:
                lines.append(f"  - {e}")
            if len(self.errors) > 5:
                lines.append(f"  ... and {len(self.errors) - 5} more")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TrainingDataResult("
            f"records={len(self.records)}, "
            f"originals={self.n_originals}, "
            f"degraded={self.n_degraded}, "
            f"warnings={len(self.warnings)}, "
            f"errors={len(self.errors)})"
        )
