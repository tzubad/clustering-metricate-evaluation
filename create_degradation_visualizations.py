"""
Create 18 interactive HTML visualizations - one per degradation type.
Each HTML has a dropdown to toggle between degradation levels + original.
Uses UMAP fitted on original data for consistent projections.
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
import plotly.graph_objects as go
from umap import UMAP
import json

# Configuration
BASE_PATH = Path('degraded_datasets')
ORIGINAL_FILE = Path('narrative_dataset_model_1247315_with_reduced.csv')
OUTPUT_DIR = Path('degradation_visualizations')
SAMPLE_SIZE = 5000  # Sample for performance

def parse_embedding(emb_str):
    """Parse embedding string to numpy array."""
    try:
        return np.array(ast.literal_eval(emb_str))
    except:
        return None

def load_original_data():
    """Load the original dataset."""
    print("Loading original dataset...")
    df = pd.read_csv(ORIGINAL_FILE, usecols=['post_id', 'cluster_id', 'reduced_embedding'])
    return df

def load_degraded_data(filepath):
    """Load a degraded dataset."""
    df = pd.read_csv(filepath, usecols=['post_id', 'cluster_id', 'reduced_embedding'])
    return df

def fit_umap_on_original(df, n_components=2, random_state=42):
    """Fit UMAP on original data embeddings."""
    print("Parsing embeddings from original data...")
    embeddings = df['reduced_embedding'].apply(parse_embedding)
    valid_mask = embeddings.apply(lambda x: x is not None)
    embeddings = np.stack(embeddings[valid_mask].values)
    
    print(f"Fitting UMAP on {len(embeddings)} points...")
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=random_state
    )
    umap_model.fit(embeddings)
    
    return umap_model, valid_mask

def transform_data(df, umap_model, sample_ids=None):
    """Transform data using fitted UMAP model."""
    if sample_ids is not None:
        df = df[df['post_id'].isin(sample_ids)].copy()
    
    embeddings = df['reduced_embedding'].apply(parse_embedding)
    valid_mask = embeddings.apply(lambda x: x is not None)
    df = df[valid_mask].copy()
    embeddings = np.stack(embeddings[valid_mask].values)
    
    coords = umap_model.transform(embeddings)
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]
    
    return df[['post_id', 'cluster_id', 'x', 'y']]

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

def format_level(level, deg_type):
    """Format level for display."""
    if deg_type == 'noise_injection':
        return f"{int(level)} points"
    elif level >= 1:
        if deg_type.startswith('merge_'):
            return f"{int(level)} pairs"
        else:
            return f"{int(level)} clusters"
    else:
        return f"{level:.0%}"

def create_html_for_type(deg_type, manifest, original_df, umap_model, sample_ids):
    """Create interactive HTML for one degradation type."""
    
    display_name = get_type_display_name(deg_type)
    print(f"\nProcessing {display_name}...")
    
    # Get all files for this type, sorted by level
    type_files = manifest[manifest['type'] == deg_type].sort_values('level')
    
    # Color palette
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
        '#5254a3', '#6b6ecf', '#9c9ede', '#b5cf6b', '#cedb9c'
    ]
    
    # Process original data first
    print("  - Processing original...")
    orig_transformed = transform_data(original_df.copy(), umap_model, sample_ids)
    
    # Get all unique clusters across all versions
    all_clusters = set(orig_transformed['cluster_id'].unique())
    
    # Store all datasets
    datasets = {'Original': orig_transformed}
    
    # Process each degradation level
    for _, row in type_files.iterrows():
        level = row['level']
        filename = row['filename']
        level_name = format_level(level, deg_type)
        
        print(f"  - Processing {level_name}...")
        
        deg_df = load_degraded_data(BASE_PATH / filename)
        
        # For types that modify embeddings, we need special handling
        if deg_type in ['embedding_perturbation', 'centroid_displacement']:
            # These modify embeddings, so we need to re-transform
            deg_transformed = transform_data(deg_df, umap_model, sample_ids)
        elif deg_type == 'noise_injection':
            # Noise injection adds new points
            # Transform the sampled original points + new noise points
            orig_ids = set(sample_ids) if sample_ids else set(original_df['post_id'])
            noise_ids = set(deg_df['post_id']) - orig_ids
            # Sample from noise points if too many
            noise_sample = list(noise_ids)[:500]
            combined_ids = orig_ids | set(noise_sample)
            deg_transformed = transform_data(deg_df, umap_model, combined_ids)
        else:
            # For label changes or removals, just use sample_ids that exist
            deg_transformed = transform_data(deg_df, umap_model, sample_ids)
        
        datasets[level_name] = deg_transformed
        all_clusters.update(deg_transformed['cluster_id'].unique())
    
    # Assign consistent colors to clusters
    sorted_clusters = sorted(all_clusters, key=lambda x: (x == -1, x))  # Put -1 (noise) last
    cluster_colors = {c: colors[i % len(colors)] for i, c in enumerate(sorted_clusters)}
    cluster_colors[-1] = '#888888'  # Gray for noise cluster
    
    # Create figure with all traces (initially hidden except Original)
    fig = go.Figure()
    
    # Track button visibility
    dataset_names = list(datasets.keys())
    
    for ds_idx, (ds_name, df) in enumerate(datasets.items()):
        visible = (ds_idx == 0)  # Only first (Original) visible initially
        
        for cluster_id in sorted(df['cluster_id'].unique(), key=lambda x: (x == -1, x)):
            cluster_data = df[df['cluster_id'] == cluster_id]
            color = cluster_colors.get(cluster_id, '#888888')
            
            cluster_label = 'Noise' if cluster_id == -1 else f'Cluster {cluster_id}'
            
            fig.add_trace(
                go.Scattergl(
                    x=cluster_data['x'],
                    y=cluster_data['y'],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=color,
                        opacity=0.7
                    ),
                    name=cluster_label,
                    legendgroup=str(cluster_id),
                    showlegend=(ds_idx == 0),  # Only show legend for first dataset
                    visible=visible,
                    hovertemplate=f'{cluster_label}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra>{ds_name}</extra>',
                    customdata=[ds_name] * len(cluster_data)
                )
            )
    
    # Count traces per dataset
    traces_per_dataset = []
    for ds_name, df in datasets.items():
        n_clusters = df['cluster_id'].nunique()
        traces_per_dataset.append(n_clusters)
    
    # Create dropdown buttons
    buttons = []
    trace_idx = 0
    
    for ds_idx, ds_name in enumerate(dataset_names):
        n_traces = traces_per_dataset[ds_idx]
        
        # Create visibility array
        visibility = [False] * sum(traces_per_dataset)
        for i in range(n_traces):
            visibility[trace_idx + i] = True
        
        buttons.append(dict(
            label=ds_name,
            method='update',
            args=[{'visible': visibility}]
        ))
        
        trace_idx += n_traces
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>{display_name}</b><br><sup>Toggle degradation levels using the dropdown below</sup>',
            x=0.5,
            font=dict(size=18)
        ),
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction='down',
                showactive=True,
                x=0.0,
                xanchor='left',
                y=1.15,
                yanchor='top',
                bgcolor='white',
                bordercolor='#ccc',
                font=dict(size=12)
            )
        ],
        annotations=[
            dict(
                text='<b>Degradation Level:</b>',
                x=0.0,
                xref='paper',
                y=1.2,
                yref='paper',
                align='left',
                showarrow=False,
                font=dict(size=12)
            )
        ],
        height=700,
        width=1000,
        showlegend=True,
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa',
        xaxis=dict(title='UMAP 1', showgrid=True, gridcolor='#e0e0e0'),
        yaxis=dict(title='UMAP 2', showgrid=True, gridcolor='#e0e0e0')
    )
    
    # Save HTML
    output_file = OUTPUT_DIR / f'{deg_type}.html'
    fig.write_html(
        output_file,
        include_plotlyjs=True,
        full_html=True,
        config={
            'displayModeBar': True,
            'scrollZoom': True,
            'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
            'toImageButtonOptions': {'format': 'png', 'scale': 2}
        }
    )
    
    print(f"  Saved: {output_file}")
    return output_file

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load manifest
    manifest = pd.read_csv(BASE_PATH / 'manifest.csv')
    deg_types = sorted(manifest['type'].unique())
    print(f"Found {len(deg_types)} degradation types")
    
    # Load and prepare original data
    original_df = load_original_data()
    
    # Sample for performance
    if len(original_df) > SAMPLE_SIZE:
        sample_df = original_df.sample(n=SAMPLE_SIZE, random_state=42)
        sample_ids = set(sample_df['post_id'])
    else:
        sample_df = original_df
        sample_ids = set(original_df['post_id'])
    
    # Fit UMAP on original
    umap_model, valid_mask = fit_umap_on_original(sample_df)
    
    # Create HTML for each type
    created_files = []
    for deg_type in deg_types:
        output_file = create_html_for_type(
            deg_type, manifest, original_df, umap_model, sample_ids
        )
        created_files.append(output_file)
    
    # Create index page
    create_index_page(deg_types, created_files)
    
    print(f"\n{'='*60}")
    print(f"Created {len(created_files)} visualization files in {OUTPUT_DIR}/")
    print(f"Open {OUTPUT_DIR}/index.html to browse all visualizations")

def create_index_page(deg_types, files):
    """Create an index page linking to all visualizations."""
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Degradation Visualizations Index</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 40px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }
        .card a {
            text-decoration: none;
            color: #1a73e8;
            font-weight: 600;
            font-size: 16px;
        }
        .card p {
            color: #666;
            margin-top: 8px;
            font-size: 14px;
        }
        .category {
            display: inline-block;
            padding: 4px 8px;
            background: #e8f0fe;
            color: #1a73e8;
            border-radius: 4px;
            font-size: 12px;
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <h1>🔬 Clustering Degradation Visualizations</h1>
    <p class="subtitle">Interactive visualizations of 18 degradation types. Each page allows toggling between degradation levels.</p>
    <div class="grid">
"""
    
    categories = {
        'Label Manipulation': ['label_swap_distant', 'label_swap_neighboring', 'label_swap_random', 'boundary_reassignment'],
        'Point Removal': ['random_removal', 'core_removal'],
        'Cluster Operations': ['merge_farthest', 'merge_nearest', 'merge_random', 
                              'split_largest', 'split_loosest', 'split_random',
                              'remove_largest_clusters', 'remove_smallest_clusters', 'remove_tightest_clusters'],
        'Embedding Changes': ['embedding_perturbation', 'centroid_displacement'],
        'Data Addition': ['noise_injection']
    }
    
    type_to_category = {}
    for cat, types in categories.items():
        for t in types:
            type_to_category[t] = cat
    
    for deg_type in deg_types:
        display_name = get_type_display_name(deg_type)
        category = type_to_category.get(deg_type, 'Other')
        
        html += f"""
        <div class="card">
            <span class="category">{category}</span>
            <a href="{deg_type}.html">{display_name}</a>
            <p>Compare original vs degraded clustering</p>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    with open(OUTPUT_DIR / 'index.html', 'w') as f:
        f.write(html)
    
    print(f"\nCreated index page: {OUTPUT_DIR}/index.html")

if __name__ == '__main__':
    main()
