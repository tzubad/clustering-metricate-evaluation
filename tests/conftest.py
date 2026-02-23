"""Pytest fixtures for metricate tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Path to existing test data
TEST_DATA_DIR = Path(__file__).parent.parent / "degraded_datasets"
TEST_DATA_17_DIR = Path(__file__).parent.parent / "degraded_datasets_17clusters"


@pytest.fixture
def sample_clustering_csv(tmp_path):
    """Create a simple clustering CSV for testing."""
    np.random.seed(42)
    n_points = 100
    n_clusters = 3
    n_dims = 10

    # Generate random embeddings
    embeddings = np.random.randn(n_points, n_dims)

    # Assign to clusters
    labels = np.random.randint(0, n_clusters, n_points)

    # Create DataFrame
    df = pd.DataFrame(
        {"cluster_id": labels, **{f"embedding_{i}": embeddings[:, i] for i in range(n_dims)}}
    )

    # Save to temp file
    csv_path = tmp_path / "test_clustering.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def sample_clustering_with_noise(tmp_path):
    """Create a clustering CSV with noise points (label=-1)."""
    np.random.seed(42)
    n_points = 100
    n_dims = 10

    # Generate random embeddings
    embeddings = np.random.randn(n_points, n_dims)

    # Assign to clusters with some noise
    labels = np.random.randint(0, 3, n_points)
    labels[:10] = -1  # First 10 points are noise

    df = pd.DataFrame(
        {"cluster_id": labels, **{f"embedding_{i}": embeddings[:, i] for i in range(n_dims)}}
    )

    csv_path = tmp_path / "test_clustering_noise.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def two_clusterings(tmp_path):
    """Create two clustering CSVs for comparison testing."""
    np.random.seed(42)
    n_points = 100
    n_dims = 10

    # Same embeddings, different labels
    embeddings = np.random.randn(n_points, n_dims)

    # Clustering A: 3 clusters
    labels_a = np.random.randint(0, 3, n_points)

    # Clustering B: 5 clusters (intentionally different)
    labels_b = np.random.randint(0, 5, n_points)

    df_a = pd.DataFrame(
        {"cluster_id": labels_a, **{f"embedding_{i}": embeddings[:, i] for i in range(n_dims)}}
    )

    df_b = pd.DataFrame(
        {"cluster_id": labels_b, **{f"embedding_{i}": embeddings[:, i] for i in range(n_dims)}}
    )

    csv_a = tmp_path / "clustering_a.csv"
    csv_b = tmp_path / "clustering_b.csv"

    df_a.to_csv(csv_a, index=False)
    df_b.to_csv(csv_b, index=False)

    return csv_a, csv_b


@pytest.fixture
def existing_degraded_csv():
    """Return path to an existing degraded dataset if available."""
    sample_file = TEST_DATA_DIR / "random_removal_10pct.csv"
    if sample_file.exists():
        return sample_file
    return None


@pytest.fixture
def large_clustering_csv(tmp_path):
    """Create a large clustering CSV (for performance testing)."""
    np.random.seed(42)
    n_points = 10000  # 10k points
    n_clusters = 20
    n_dims = 50

    embeddings = np.random.randn(n_points, n_dims)
    labels = np.random.randint(0, n_clusters, n_points)

    df = pd.DataFrame(
        {"cluster_id": labels, **{f"embedding_{i}": embeddings[:, i] for i in range(n_dims)}}
    )

    csv_path = tmp_path / "large_clustering.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def invalid_csv_missing_cluster(tmp_path):
    """Create a CSV without cluster_id column."""
    df = pd.DataFrame(
        {
            "embedding_0": [0.1, 0.2, 0.3],
            "embedding_1": [0.4, 0.5, 0.6],
        }
    )
    csv_path = tmp_path / "no_cluster.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def single_cluster_csv(tmp_path):
    """Create a CSV with only one cluster (edge case)."""
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 0, 0],
            "embedding_0": [0.1, 0.2, 0.3, 0.4, 0.5],
            "embedding_1": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    csv_path = tmp_path / "single_cluster.csv"
    df.to_csv(csv_path, index=False)
    return csv_path
