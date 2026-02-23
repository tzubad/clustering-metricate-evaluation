"""
Interactive visualization of 18 degradation types using Plotly.
Generates an HTML file with subplots for each degradation type.
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
BASE_PATH = Path('degraded_datasets')
OUTPUT_FILE = 'degradation_visualization.html'

def parse_reduced_embedding(emb_str):
    """Parse the reduced_embedding string to extract x, y coordinates."""
    try:
        coords = ast.literal_eval(emb_str)
        return coords[0], coords[1]  # Use first 2 dimensions
    except:
        return np.nan, np.nan

def load_sample_data(filepath, sample_size=2000):
    """Load and sample data from a degraded dataset."""
    df = pd.read_csv(filepath, usecols=['cluster_id', 'reduced_embedding'])
    
    # Sample for visualization performance
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Parse embeddings
    coords = df['reduced_embedding'].apply(parse_reduced_embedding)
    df['x'] = coords.apply(lambda c: c[0])
    df['y'] = coords.apply(lambda c: c[1])
    
    return df[['cluster_id', 'x', 'y']].dropna()

def get_type_display_name(type_name):
    """Convert type name to display-friendly format."""
    name_map = {
        'boundary_reassignment': 'Boundary Reassignment',
        'centroid_displacement': 'Centroid Displacement',
        'core_removal': 'Core Point Removal',
        'embedding_perturbation': 'Embedding Perturbation',
        'label_swap_distant': 'Label Swap (Distant)',
        'label_swap_neighboring': 'Label Swap (Neighboring)',
        'label_swap_random': 'Label Swap (Random)',
        'merge_farthest': 'Merge Clusters (Farthest)',
        'merge_nearest': 'Merge Clusters (Nearest)',
        'merge_random': 'Merge Clusters (Random)',
        'noise_injection': 'Noise Injection',
        'random_removal': 'Random Point Removal',
        'remove_largest_clusters': 'Remove Largest Clusters',
        'remove_smallest_clusters': 'Remove Smallest Clusters',
        'remove_tightest_clusters': 'Remove Tightest Clusters',
        'split_largest': 'Split Largest Clusters',
        'split_loosest': 'Split Loosest Clusters',
        'split_random': 'Split Random Clusters'
    }
    return name_map.get(type_name, type_name.replace('_', ' ').title())

def create_visualization():
    """Create the main interactive visualization."""
    
    # Load manifest
    manifest = pd.read_csv(BASE_PATH / 'manifest.csv')
    
    # Get unique degradation types
    deg_types = manifest['type'].unique()
    print(f"Found {len(deg_types)} degradation types")
    
    # Arrange in 6x3 grid
    n_cols = 3
    n_rows = 6
    
    # Create subplot titles
    subplot_titles = [get_type_display_name(t) for t in sorted(deg_types)]
    
    # Create figure with subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.06
    )
    
    # Color palette for clusters
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173'
    ]
    
    # Process each degradation type
    for idx, deg_type in enumerate(sorted(deg_types)):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        print(f"Processing {deg_type}...")
        
        # Get files for this type (use middle level for representative view)
        type_files = manifest[manifest['type'] == deg_type].sort_values('level')
        mid_idx = len(type_files) // 2
        sample_file = type_files.iloc[mid_idx]['filename']
        level = type_files.iloc[mid_idx]['level']
        
        # Load data
        df = load_sample_data(BASE_PATH / sample_file)
        
        # Get unique clusters and assign colors
        unique_clusters = df['cluster_id'].unique()
        cluster_colors = {c: colors[i % len(colors)] for i, c in enumerate(sorted(unique_clusters))}
        
        # Add scatter for each cluster
        for cluster_id in sorted(unique_clusters):
            cluster_data = df[df['cluster_id'] == cluster_id]
            color = cluster_colors[cluster_id]
            
            # Format level for display
            if level >= 1:
                level_str = f"{int(level)}"
            else:
                level_str = f"{level:.0%}"
            
            fig.add_trace(
                go.Scattergl(
                    x=cluster_data['x'],
                    y=cluster_data['y'],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=color,
                        opacity=0.6
                    ),
                    name=f'Cluster {cluster_id}',
                    hovertemplate=f'Cluster: {cluster_id}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Update subplot title to include level
        fig.layout.annotations[idx].text = f"{get_type_display_name(deg_type)}<br><sup>(level: {level_str})</sup>"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Clustering Degradation Types Visualization</b><br><sup>Each plot shows a representative degradation level. Points colored by cluster assignment.</sup>',
            x=0.5,
            font=dict(size=20)
        ),
        height=2000,
        width=1600,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa'
    )
    
    # Update all axes
    fig.update_xaxes(showticklabels=False, showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    fig.update_yaxes(showticklabels=False, showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    
    # Save to HTML
    fig.write_html(
        OUTPUT_FILE,
        include_plotlyjs=True,
        full_html=True,
        config={
            'displayModeBar': True,
            'scrollZoom': True,
            'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
            'toImageButtonOptions': {'format': 'png', 'scale': 2}
        }
    )
    
    print(f"\nVisualization saved to {OUTPUT_FILE}")
    return fig

if __name__ == '__main__':
    create_visualization()
