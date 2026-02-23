"""
Output formatters for metric evaluation results.

This module provides various output format options for evaluation results
including tables, DataFrames, JSON, and CSV.
"""

import json

import pandas as pd

from metricate.output.report import ComparisonResult, EvaluationResult


def to_dataframe(result: EvaluationResult, include_skipped: bool = True) -> pd.DataFrame:
    """
    Convert evaluation result to a pandas DataFrame.

    Args:
        result: EvaluationResult to convert
        include_skipped: Whether to include skipped metrics

    Returns:
        DataFrame with columns: Metric, Value, Range, Direction, Tier, Status
    """
    data = []
    for m in result.metrics:
        if not include_skipped and not m.computed:
            continue

        row = {
            "Metric": m.metric,
            "Value": m.value,
            "Range": m.range,
            "Direction": m.direction,
            "Tier": m.tier,
        }
        if not m.computed:
            row["Status"] = f"Skipped: {m.skip_reason}"
        else:
            row["Status"] = "Computed"
        data.append(row)

    return pd.DataFrame(data)


def to_table(
    result: EvaluationResult,
    format: str = "simple",
    include_skipped: bool = False,
    max_value_width: int = 12,
) -> str:
    """
    Generate a formatted text table from evaluation results.

    Args:
        result: EvaluationResult to format
        format: Table format ('simple', 'grid', 'markdown')
        include_skipped: Whether to include skipped metrics
        max_value_width: Maximum width for value column

    Returns:
        Formatted table string
    """
    # Build rows
    rows = []
    for m in result.metrics:
        if not include_skipped and not m.computed:
            continue

        if m.value is not None:
            # Format value with appropriate precision
            if abs(m.value) >= 1000 or (abs(m.value) < 0.01 and m.value != 0):
                value_str = f"{m.value:.4e}"
            else:
                value_str = f"{m.value:.6f}"
        else:
            value_str = "—"

        rows.append(
            {
                "Metric": m.metric,
                "Value": value_str,
                "Range": m.range,
                "Direction": m.direction,
            }
        )

    if not rows:
        return "No metrics computed"

    if format == "markdown":
        return _markdown_table(rows)
    elif format == "grid":
        return _grid_table(rows)
    else:  # simple
        return _simple_table(rows)


def to_json(
    result: EvaluationResult,
    indent: int = 2,
    include_metadata: bool = True,
) -> str:
    """
    Convert evaluation result to JSON string.

    Args:
        result: EvaluationResult to convert
        indent: JSON indentation level
        include_metadata: Whether to include metadata and warnings

    Returns:
        JSON string representation
    """
    data = {
        "metrics": [m.to_dict() for m in result.metrics],
    }

    if include_metadata:
        data["metadata"] = result.metadata
        if result.warnings:
            data["warnings"] = result.warnings

    return json.dumps(data, indent=indent, default=str)


def to_csv(
    result: EvaluationResult,
    include_skipped: bool = True,
) -> str:
    """
    Convert evaluation result to CSV string.

    Args:
        result: EvaluationResult to convert
        include_skipped: Whether to include skipped metrics

    Returns:
        CSV string representation
    """
    df = to_dataframe(result, include_skipped=include_skipped)
    return df.to_csv(index=False)


def format_comparison(
    comparison: ComparisonResult,
    format: str = "simple",
    show_winner: bool = True,
) -> str:
    """
    Format a comparison result as a table.

    Args:
        comparison: ComparisonResult to format
        format: Table format ('simple', 'grid', 'markdown')
        show_winner: Whether to include winner column

    Returns:
        Formatted comparison table
    """
    df = comparison.to_dataframe()

    if df.empty:
        return "No comparison data"

    if show_winner and comparison.baseline_name:
        # Add winner column based on metric direction
        from metricate.core.reference import METRIC_REFERENCE

        winners = []
        for _, row in df.iterrows():
            metric_name = row["Metric"]
            ref = METRIC_REFERENCE.get(metric_name, {})
            direction = ref.get("direction", "higher")

            # Get values for comparison
            eval_names = list(comparison.evaluations.keys())
            vals = [row.get(n) for n in eval_names]

            # Skip if any value is None
            if any(v is None for v in vals):
                winners.append("—")
                continue

            # Determine winner
            if direction == "higher":
                best_idx = max(range(len(vals)), key=lambda i: vals[i])
            else:
                best_idx = min(range(len(vals)), key=lambda i: vals[i])

            winners.append(eval_names[best_idx])

        df["Winner"] = winners

    # Convert to requested format
    rows = df.to_dict("records")

    if format == "markdown":
        return _markdown_table(rows)
    elif format == "grid":
        return _grid_table(rows)
    else:
        return _simple_table(rows)


# =============================================================================
# Internal table formatters
# =============================================================================


def _simple_table(rows: list[dict]) -> str:
    """Generate a simple text table."""
    if not rows:
        return "No data"

    # Get column order
    cols = list(rows[0].keys())

    # Calculate column widths
    widths = {}
    for col in cols:
        max_width = max(len(str(col)), max(len(str(row.get(col, ""))) for row in rows))
        widths[col] = min(max_width, 30)  # Cap at 30 chars

    # Build header
    header = " | ".join(str(col).ljust(widths[col]) for col in cols)
    separator = "-+-".join("-" * widths[col] for col in cols)

    # Build rows
    lines = [header, separator]
    for row in rows:
        row_str = " | ".join(
            str(row.get(col, ""))[: widths[col]].ljust(widths[col]) for col in cols
        )
        lines.append(row_str)

    return "\n".join(lines)


def _grid_table(rows: list[dict]) -> str:
    """Generate a grid-style table with box drawing characters."""
    if not rows:
        return "No data"

    cols = list(rows[0].keys())

    # Calculate widths with padding
    widths = {}
    for col in cols:
        max_width = max(len(str(col)), max(len(str(row.get(col, ""))) for row in rows))
        widths[col] = min(max_width + 2, 32)  # Padding + cap

    # Box drawing
    top = "┌" + "┬".join("─" * widths[col] for col in cols) + "┐"
    mid = "├" + "┼".join("─" * widths[col] for col in cols) + "┤"
    bot = "└" + "┴".join("─" * widths[col] for col in cols) + "┘"

    # Header
    header = (
        "│"
        + "│".join(f" {str(col)[: widths[col] - 2].center(widths[col] - 2)} " for col in cols)
        + "│"
    )

    # Rows
    lines = [top, header, mid]
    for row in rows:
        row_str = (
            "│"
            + "│".join(
                f" {str(row.get(col, ''))[: widths[col] - 2].ljust(widths[col] - 2)} "
                for col in cols
            )
            + "│"
        )
        lines.append(row_str)
    lines.append(bot)

    return "\n".join(lines)


def _markdown_table(rows: list[dict]) -> str:
    """Generate a markdown table."""
    if not rows:
        return "No data"

    cols = list(rows[0].keys())

    # Header
    header = "| " + " | ".join(cols) + " |"
    separator = "|" + "|".join("---" for _ in cols) + "|"

    # Rows
    lines = [header, separator]
    for row in rows:
        row_str = "| " + " | ".join(str(row.get(col, "")) for col in cols) + " |"
        lines.append(row_str)

    return "\n".join(lines)
