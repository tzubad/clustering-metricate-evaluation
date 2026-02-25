import metricate

# Generate degradations with visualizations
result = metricate.degrade(
    "/Users/alonneduva/Desktop/MindINT/Research/research/degrading-clustering-dataset/datasets/narrative_dataset_17clusters_full.csv",
    output_dir="./outpu_test/",
    visualize=True  # Enable visualization generation
)

# Check generated visualization paths
print(f"Index page: {result.index_html_path}")
print(f"Visualizations: {result.visualizations}")