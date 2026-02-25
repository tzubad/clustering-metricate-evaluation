"""
Visualization generators for degradation results.

Creates HTML visualizations using Plotly to show the effects of
each degradation on the clustering.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

if TYPE_CHECKING:
    from metricate.degradation.toolkit import DegradationEntry


def generate_visualizations(
    degradations: list["DegradationEntry"],
    output_dir: Path,
    original_csv_path: str | None = None,
) -> list[str]:
    """
    Generate HTML visualization for each degradation type.

    Creates a 2D scatter plot with a dropdown to select severity level,
    showing the clustering after degradation colored by cluster label.
    Includes the baseline (original) data in the dropdown.

    Args:
        degradations: List of DegradationEntry objects with file paths
        output_dir: Directory to write HTML files
        original_csv_path: Path to the original CSV for baseline comparison

    Returns:
        List of generated HTML file paths
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for visualizations. Install with: pip install plotly")

    visualizations = []
    output_path = Path(output_dir)

    # Load baseline data if provided
    baseline_data = None
    if original_csv_path:
        try:
            df_orig = pd.read_csv(original_csv_path)
            embedding_col = _find_embedding_col(df_orig)
            label_col = _find_label_col(df_orig)
            if embedding_col and label_col:
                if pd.api.types.is_string_dtype(df_orig[embedding_col]) or df_orig[embedding_col].dtype == object:
                    embeddings = _parse_embeddings(df_orig[embedding_col])
                else:
                    embeddings = df_orig[embedding_col].values

                if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                    if embeddings.shape[1] > 2:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2, random_state=42)
                        baseline_coords = pca.fit_transform(embeddings)
                    else:
                        baseline_coords = embeddings

                    baseline_data = {
                        "coords": baseline_coords,
                        "labels": df_orig[label_col].astype(str),
                        "n_rows": len(df_orig),
                    }
        except Exception:
            pass

    # Group degradations by type
    by_type: dict[str, list] = {}
    for entry in degradations:
        if entry.type not in by_type:
            by_type[entry.type] = []
        by_type[entry.type].append(entry)

    # Sort levels for consistent ordering
    level_order = ["5pct", "10pct", "25pct", "50pct", "75pct", "100pct"]

    for deg_type, entries in by_type.items():
        try:
            # Sort entries by level
            sorted_entries = sorted(
                entries,
                key=lambda e: level_order.index(e.level) if e.level in level_order else 999
            )

            # Process all levels and collect data
            level_data = []
            all_labels = set()

            # Add baseline labels first
            if baseline_data:
                all_labels.update(baseline_data["labels"].unique())

            for entry in sorted_entries:
                df = pd.read_csv(entry.filepath)

                embedding_col = _find_embedding_col(df)
                label_col = _find_label_col(df)

                if embedding_col is None or label_col is None:
                    continue

                # Parse embeddings
                if pd.api.types.is_string_dtype(df[embedding_col]) or df[embedding_col].dtype == object:
                    embeddings = _parse_embeddings(df[embedding_col])
                else:
                    embeddings = df[embedding_col].values

                # Reduce to 2D if needed
                if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                    if embeddings.shape[1] > 2:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2, random_state=42)
                        coords_2d = pca.fit_transform(embeddings)
                    else:
                        coords_2d = embeddings
                else:
                    continue

                labels = df[label_col].astype(str)
                all_labels.update(labels.unique())

                level_data.append({
                    "entry": entry,
                    "coords": coords_2d,
                    "labels": labels,
                })

            if not level_data:
                continue

            # Create consistent color mapping across all levels
            unique_labels = sorted(all_labels, key=lambda x: (x.lstrip('-').isdigit(), int(x) if x.lstrip('-').isdigit() else 0, x))
            colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set3
            color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

            # Create figure with all traces
            fig = go.Figure()

            # Track traces per level for visibility toggling
            traces_per_level = []
            trace_idx = 0

            # Add baseline first if available
            if baseline_data:
                level_trace_start = trace_idx
                for label in unique_labels:
                    mask = baseline_data["labels"] == label
                    if not mask.any():
                        continue

                    fig.add_trace(go.Scatter(
                        x=baseline_data["coords"][mask, 0],
                        y=baseline_data["coords"][mask, 1],
                        mode="markers",
                        name=f"Cluster {label}",
                        marker={"color": color_map[label], "size": 6},
                        legendgroup=label,
                        showlegend=True,
                        visible=True,
                        hovertemplate=f"Cluster: {label}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>",
                    ))
                    trace_idx += 1

                traces_per_level.append({
                    "level": "baseline",
                    "label": f"Baseline ({baseline_data['n_rows']:,} pts)",
                    "description": "Original clustering (no degradation)",
                    "n_rows": baseline_data["n_rows"],
                    "start": level_trace_start,
                    "end": trace_idx,
                })

            # Add degraded levels
            for i, data in enumerate(level_data):
                entry = data["entry"]
                coords = data["coords"]
                labels = data["labels"]
                is_first = (i == 0 and baseline_data is None)

                level_trace_start = trace_idx

                # Add one trace per cluster for this level
                for label in unique_labels:
                    mask = labels == label
                    if not mask.any():
                        continue

                    fig.add_trace(go.Scatter(
                        x=coords[mask, 0],
                        y=coords[mask, 1],
                        mode="markers",
                        name=f"Cluster {label}",
                        marker={"color": color_map[label], "size": 6},
                        legendgroup=label,
                        showlegend=is_first,
                        visible=False,  # All degraded levels start hidden (baseline is shown)
                        hovertemplate=f"Cluster: {label}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>",
                    ))
                    trace_idx += 1

                # Format the dropdown label based on degradation type
                dropdown_label = _format_dropdown_label(entry)

                traces_per_level.append({
                    "level": entry.level,
                    "label": dropdown_label,
                    "description": entry.change_description,
                    "n_rows": entry.n_rows,
                    "start": level_trace_start,
                    "end": trace_idx,
                })

            # Create dropdown buttons
            buttons = []
            total_traces = trace_idx
            type_title = _format_type_name(deg_type)

            for level_info in traces_per_level:
                # Create visibility array: True for this level's traces, False for others
                visibility = [False] * total_traces
                for idx in range(level_info["start"], level_info["end"]):
                    visibility[idx] = True

                buttons.append({
                    "label": level_info["label"],
                    "method": "update",
                    "args": [
                        {"visible": visibility},
                        {"annotations": [
                            {
                                "text": "Severity:",
                                "x": 0.0,
                                "xref": "paper",
                                "y": 1.18,
                                "yref": "paper",
                                "showarrow": False,
                                "font": {"size": 12, "color": "#666"},
                            },
                            {
                                "text": level_info["description"],
                                "x": 0.5,
                                "xref": "paper",
                                "y": -0.08,
                                "yref": "paper",
                                "showarrow": False,
                                "font": {"size": 11, "color": "#888"},
                            }
                        ]}
                    ]
                })

            # Initial description
            initial_desc = traces_per_level[0]["description"] if traces_per_level else ""

            # Update layout with dropdown
            fig.update_layout(
                title={
                    "text": type_title,
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 18},
                },
                template="plotly_white",
                width=900,
                height=700,
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                legend={
                    "yanchor": "top",
                    "y": 0.99,
                    "xanchor": "right",
                    "x": 0.99,
                    "bgcolor": "rgba(255,255,255,0.8)",
                },
                updatemenus=[
                    {
                        "active": 0,
                        "buttons": buttons,
                        "direction": "down",
                        "showactive": True,
                        "x": 0.0,
                        "xanchor": "left",
                        "y": 1.15,
                        "yanchor": "top",
                        "bgcolor": "white",
                        "bordercolor": "#ccc",
                        "font": {"size": 12},
                    }
                ],
                annotations=[
                    {
                        "text": "Severity:",
                        "x": 0.0,
                        "xref": "paper",
                        "y": 1.18,
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 12, "color": "#666"},
                    },
                    {
                        "text": initial_desc,
                        "x": 0.5,
                        "xref": "paper",
                        "y": -0.08,
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 11, "color": "#888"},
                    }
                ],
                margin={"b": 80},  # Extra bottom margin for description
            )

            # Save
            viz_filename = f"{deg_type}.html"
            viz_path = output_path / viz_filename
            fig.write_html(str(viz_path), include_plotlyjs="cdn")
            visualizations.append(str(viz_path))

        except Exception:
            # Skip types that can't be visualized
            continue

    return visualizations


def _format_dropdown_label(entry: "DegradationEntry") -> str:
    """Format dropdown label based on degradation type."""
    deg_type = entry.type
    level = entry.level
    n_rows = entry.n_rows

    # For percentage-based degradations, show percentage
    if deg_type in ["label_swap_random", "label_swap_neighboring", "label_swap_distant",
                    "random_removal", "core_removal", "boundary_reassignment",
                    "embedding_perturbation", "centroid_displacement", "noise_injection"]:
        pct = level.replace("pct", "%")
        return f"{pct} ({n_rows:,} pts)"

    # For cluster operations (merge, split, remove), show the description from level
    elif deg_type.startswith("merge_") or deg_type.startswith("split_") or deg_type.startswith("remove_"):
        # Extract the number from description or use level-based estimate
        level_num = {"5pct": 1, "10pct": 1, "25pct": 2, "50pct": 5}.get(level, 1)
        if deg_type.startswith("merge_"):
            return f"{level_num} merge{'s' if level_num > 1 else ''} ({n_rows:,} pts)"
        elif deg_type.startswith("split_"):
            return f"{level_num} split{'s' if level_num > 1 else ''} ({n_rows:,} pts)"
        else:  # remove_
            return f"{level_num} removed ({n_rows:,} pts)"

    # Default fallback
    return f"{level} ({n_rows:,} pts)"


def generate_index(
    degradations: list["DegradationEntry"],
    output_dir: Path,
) -> str:
    """
    Generate an index.html file linking to all visualizations.

    Creates a dashboard-style page with:
    - Summary statistics
    - Links to each degradation type
    - Organized by degradation category

    Args:
        degradations: List of DegradationEntry objects
        output_dir: Directory containing the HTML visualizations

    Returns:
        Path to the generated index.html
    """
    output_path = Path(output_dir)

    # Group degradations by type
    by_type = {}
    for d in degradations:
        if d.type not in by_type:
            by_type[d.type] = []
        by_type[d.type].append(d)

    # Category mapping
    CATEGORIES = {
        "Label Manipulation": ["label_swap_random", "label_swap_neighboring", "label_swap_distant"],
        "Cluster Structure": [
            "merge_random",
            "merge_nearest",
            "merge_farthest",
            "split_random",
            "split_largest",
            "split_loosest",
        ],
        "Point Manipulation": [
            "noise_injection",
            "random_removal",
            "core_removal",
            "boundary_reassignment",
        ],
        "Cluster Removal": [
            "remove_smallest_clusters",
            "remove_largest_clusters",
            "remove_tightest_clusters",
        ],
        "Embedding Manipulation": ["embedding_perturbation", "centroid_displacement"],
    }

    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        "    <title>Degradation Visualizations Index</title>",
        "    <style>",
        '        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; ',
        "               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }",
        "        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }",
        "        h2 { color: #555; margin-top: 30px; }",
        "        .summary { background: white; padding: 20px; border-radius: 8px; margin-bottom: 30px;",
        "                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }",
        "        .stat { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 4px; }",
        "        .stat-value { font-size: 24px; font-weight: bold; color: #007bff; }",
        "        .stat-label { font-size: 12px; color: #666; text-transform: uppercase; }",
        "        .category { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;",
        "                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "        .category h3 { margin-top: 0; color: #333; }",
        "        .degradation-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; }",
        "        .degradation-card { background: #f8f9fa; padding: 15px; border-radius: 4px;",
        "                            border-left: 3px solid #007bff; }",
        "        .degradation-card h4 { margin: 0 0 10px 0; font-size: 14px; }",
        "        .degradation-card a { color: #007bff; text-decoration: none; font-size: 13px; }",
        "        .degradation-card a:hover { text-decoration: underline; }",
        "        .level-links { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }",
        "        .level-link { background: #007bff; color: white; padding: 4px 10px; border-radius: 4px;",
        "                      text-decoration: none; font-size: 12px; }",
        "        .level-link:hover { background: #0056b3; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>🔬 Degradation Visualizations</h1>",
        '    <div class="summary">',
        "        <h2>Summary</h2>",
        '        <div class="summary-grid">',
        f'            <div class="stat"><div class="stat-value">{len(degradations)}</div><div class="stat-label">Total Degradations</div></div>',
        f'            <div class="stat"><div class="stat-value">{len(by_type)}</div><div class="stat-label">Degradation Types</div></div>',
        f'            <div class="stat"><div class="stat-value">{len({d.level for d in degradations})}</div><div class="stat-label">Intensity Levels</div></div>',
        "        </div>",
        "    </div>",
    ]

    # Add each category
    for category, types in CATEGORIES.items():
        category_degradations = [t for t in types if t in by_type]
        if not category_degradations:
            continue

        html_parts.extend(
            [
                '    <div class="category">',
                f"        <h3>{category}</h3>",
                '        <div class="degradation-grid">',
            ]
        )

        for deg_type in category_degradations:
            entries = by_type[deg_type]
            viz_file = f"{deg_type}.html"

            html_parts.extend(
                [
                    '            <div class="degradation-card">',
                    f"                <h4>{_format_type_name(deg_type)}</h4>",
                    f'                <a href="{viz_file}">View Visualization →</a>',
                    '                <div class="level-links">',
                ]
            )

            for entry in sorted(entries, key=lambda e: e.level):
                csv_file = entry.filename
                html_parts.append(
                    f'                    <a href="{csv_file}" class="level-link">{entry.level}</a>'
                )

            html_parts.extend(
                [
                    "                </div>",
                    "            </div>",
                ]
            )

        html_parts.extend(
            [
                "        </div>",
                "    </div>",
            ]
        )

    html_parts.extend(
        [
            '    <footer style="text-align: center; color: #666; margin-top: 40px; font-size: 12px;">',
            "        Generated by <strong>Metricate</strong> Degradation Toolkit",
            "    </footer>",
            "</body>",
            "</html>",
        ]
    )

    # Write index
    index_path = output_path / "index.html"
    index_path.write_text("\n".join(html_parts))

    return str(index_path)


def _find_embedding_col(df: pd.DataFrame) -> str | None:
    """Find the embedding column in a dataframe."""
    candidates = ["reduced_embedding", "embedding", "embeddings"]
    for col in candidates:
        if col in df.columns:
            return col
    # Look for columns that might contain embeddings
    for col in df.columns:
        if "embed" in col.lower():
            return col
    return None


def _find_label_col(df: pd.DataFrame) -> str | None:
    """Find the cluster label column in a dataframe."""
    candidates = ["cluster_id", "cluster", "label", "labels", "cluster_label"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _parse_embeddings(series: pd.Series) -> np.ndarray:
    """Parse string-encoded embeddings to numpy array."""
    import ast

    embeddings = []
    for val in series:
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                embeddings.append(parsed)
            except:
                embeddings.append([0, 0])
        elif isinstance(val, (list, np.ndarray)):
            embeddings.append(val)
        else:
            embeddings.append([0, 0])

    return np.array(embeddings)


def _format_type_name(deg_type: str) -> str:
    """Format degradation type name for display."""
    return deg_type.replace("_", " ").title()
