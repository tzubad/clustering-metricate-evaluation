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
) -> list[str]:
    """
    Generate HTML visualization for each degradation.

    Creates a 2D scatter plot showing the clustering after degradation,
    colored by cluster label.

    Args:
        degradations: List of DegradationEntry objects with file paths
        output_dir: Directory to write HTML files

    Returns:
        List of generated HTML file paths
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for visualizations. Install with: pip install plotly")

    visualizations = []
    output_path = Path(output_dir)

    for entry in degradations:
        try:
            # Load the degraded data
            df = pd.read_csv(entry.filepath)

            # Find embedding columns
            embedding_col = _find_embedding_col(df)
            label_col = _find_label_col(df)

            if embedding_col is None or label_col is None:
                continue

            # Parse embeddings if needed
            if df[embedding_col].dtype == object:
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
                # Skip if we can't process
                continue

            # Create visualization
            labels = df[label_col].astype(str)

            fig = px.scatter(
                x=coords_2d[:, 0],
                y=coords_2d[:, 1],
                color=labels,
                title=f"{entry.type} @ {entry.level}",
                labels={"x": "Dimension 1", "y": "Dimension 2", "color": "Cluster"},
            )

            fig.update_layout(
                template="plotly_white",
                title_x=0.5,
                width=800,
                height=600,
                legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
            )

            # Add annotation with stats
            fig.add_annotation(
                text=f"Points: {entry.n_rows} | {entry.change_description}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.1,
                showarrow=False,
                font={"size": 10, "color": "gray"},
            )

            # Save
            viz_filename = f"{entry.type}.html"
            viz_path = output_path / viz_filename
            fig.write_html(str(viz_path), include_plotlyjs="cdn")
            visualizations.append(str(viz_path))

        except Exception:
            # Skip files that can't be visualized
            continue

    return visualizations


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
