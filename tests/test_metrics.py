"""
Unit tests for core clustering metrics.

Tests individual metric calculations using synthetic data to verify
correct computation and expected behavior.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs


class TestSilhouetteMetric:
    """Tests for Silhouette Score calculation."""

    def test_perfect_separation(self):
        """Well-separated clusters should have high silhouette."""
        from sklearn.metrics import silhouette_score

        # Create well-separated clusters
        X, labels = make_blobs(
            n_samples=300,
            n_features=10,
            centers=3,
            cluster_std=0.5,
            center_box=(-20, 20),
            random_state=42,
        )

        score = silhouette_score(X, labels)
        assert score > 0.7, f"Expected high silhouette for well-separated clusters, got {score}"

    def test_overlapping_clusters(self):
        """Overlapping clusters should have lower silhouette."""
        from sklearn.metrics import silhouette_score

        # Create overlapping clusters
        X, labels = make_blobs(
            n_samples=300,
            n_features=10,
            centers=3,
            cluster_std=3.0,
            center_box=(-5, 5),
            random_state=42,
        )

        score = silhouette_score(X, labels)
        assert score < 0.5, f"Expected lower silhouette for overlapping clusters, got {score}"

    def test_silhouette_range(self):
        """Silhouette should be in [-1, 1] range."""
        from sklearn.metrics import silhouette_score

        X, labels = make_blobs(n_samples=100, centers=3, random_state=42)
        score = silhouette_score(X, labels)

        assert -1 <= score <= 1, f"Silhouette {score} out of valid range [-1, 1]"


class TestDaviesBouldinMetric:
    """Tests for Davies-Bouldin Index calculation."""

    def test_lower_is_better(self):
        """Tighter clusters should have lower Davies-Bouldin."""
        from sklearn.metrics import davies_bouldin_score

        # Tight clusters
        X_tight, labels_tight = make_blobs(
            n_samples=300, centers=3, cluster_std=0.5, random_state=42
        )

        # Loose clusters
        X_loose, labels_loose = make_blobs(
            n_samples=300, centers=3, cluster_std=2.0, random_state=42
        )

        db_tight = davies_bouldin_score(X_tight, labels_tight)
        db_loose = davies_bouldin_score(X_loose, labels_loose)

        assert db_tight < db_loose, f"Expected lower DB for tight clusters: {db_tight} < {db_loose}"

    def test_positive_value(self):
        """Davies-Bouldin should always be non-negative."""
        from sklearn.metrics import davies_bouldin_score

        X, labels = make_blobs(n_samples=100, centers=3, random_state=42)
        score = davies_bouldin_score(X, labels)

        assert score >= 0, f"Davies-Bouldin {score} should be non-negative"


class TestCalinskiHarabaszMetric:
    """Tests for Calinski-Harabasz Index calculation."""

    def test_higher_is_better(self):
        """Well-separated clusters should have higher CH index."""
        from sklearn.metrics import calinski_harabasz_score

        # Well-separated
        X_sep, labels_sep = make_blobs(
            n_samples=300, centers=3, cluster_std=0.5, center_box=(-20, 20), random_state=42
        )

        # Overlapping
        X_overlap, labels_overlap = make_blobs(
            n_samples=300, centers=3, cluster_std=3.0, center_box=(-5, 5), random_state=42
        )

        ch_sep = calinski_harabasz_score(X_sep, labels_sep)
        ch_overlap = calinski_harabasz_score(X_overlap, labels_overlap)

        assert ch_sep > ch_overlap, (
            f"Expected higher CH for separated clusters: {ch_sep} > {ch_overlap}"
        )

    def test_positive_value(self):
        """Calinski-Harabasz should always be positive."""
        from sklearn.metrics import calinski_harabasz_score

        X, labels = make_blobs(n_samples=100, centers=3, random_state=42)
        score = calinski_harabasz_score(X, labels)

        assert score > 0, f"Calinski-Harabasz {score} should be positive"


class TestDunnIndex:
    """Tests for Dunn Index calculation."""

    def test_dunn_calculation(self):
        """Test basic Dunn index calculation."""
        from scipy.spatial.distance import pdist, squareform

        from metricate.core.metrics import dunn_index

        # Create well-separated clusters
        X, labels = make_blobs(
            n_samples=150, centers=3, cluster_std=0.5, center_box=(-10, 10), random_state=42
        )

        dist_matrix = squareform(pdist(X))
        score = dunn_index(X, labels, dist_matrix=dist_matrix)

        assert score > 0, f"Dunn index {score} should be positive"

    def test_dunn_higher_for_separated(self):
        """Well-separated clusters should have higher Dunn index."""
        from scipy.spatial.distance import pdist, squareform

        from metricate.core.metrics import dunn_index

        # Well-separated
        X_sep, labels_sep = make_blobs(
            n_samples=150, centers=3, cluster_std=0.3, center_box=(-15, 15), random_state=42
        )

        # Overlapping
        X_overlap, labels_overlap = make_blobs(
            n_samples=150, centers=3, cluster_std=2.0, center_box=(-5, 5), random_state=42
        )

        dist_sep = squareform(pdist(X_sep))
        dist_overlap = squareform(pdist(X_overlap))

        dunn_sep = dunn_index(X_sep, labels_sep, dist_matrix=dist_sep)
        dunn_overlap = dunn_index(X_overlap, labels_overlap, dist_matrix=dist_overlap)

        assert dunn_sep > dunn_overlap, (
            f"Expected higher Dunn for separated: {dunn_sep} > {dunn_overlap}"
        )


class TestSSEMetric:
    """Tests for Sum of Squared Errors (SSE) calculation."""

    def test_sse_calculation(self):
        """Test basic SSE calculation."""
        from metricate.core.metrics import sse

        X, labels = make_blobs(n_samples=100, centers=3, random_state=42)
        score = sse(X, labels)

        assert score >= 0, f"SSE {score} should be non-negative"

    def test_sse_lower_for_tight(self):
        """Tighter clusters should have lower SSE."""
        from metricate.core.metrics import sse

        # Tight clusters
        X_tight, labels_tight = make_blobs(
            n_samples=100, centers=3, cluster_std=0.5, random_state=42
        )

        # Loose clusters
        X_loose, labels_loose = make_blobs(
            n_samples=100, centers=3, cluster_std=3.0, random_state=42
        )

        sse_tight = sse(X_tight, labels_tight)
        sse_loose = sse(X_loose, labels_loose)

        assert sse_tight < sse_loose, (
            f"Expected lower SSE for tight clusters: {sse_tight} < {sse_loose}"
        )


class TestBallHallIndex:
    """Tests for Ball-Hall Index calculation."""

    def test_ball_hall_calculation(self):
        """Test Ball-Hall index calculation."""
        from metricate.core.metrics import ball_hall, precompute_all

        X, labels = make_blobs(n_samples=100, centers=3, random_state=42)
        precomputed = precompute_all(X, labels)
        score = ball_hall(X, labels, **precomputed)

        assert score >= 0, f"Ball-Hall {score} should be non-negative"


class TestMetricDirections:
    """Tests to verify metric direction preferences are correct."""

    @pytest.fixture
    def good_clustering(self):
        """Create a well-clustered dataset."""
        X, labels = make_blobs(
            n_samples=300, centers=5, cluster_std=0.5, center_box=(-20, 20), random_state=42
        )
        return X, labels

    @pytest.fixture
    def bad_clustering(self):
        """Create a poorly-clustered dataset (random labels)."""
        X, _ = make_blobs(n_samples=300, centers=5, random_state=42)
        labels = np.random.randint(0, 5, size=300)
        return X, labels

    def test_silhouette_direction(self, good_clustering, bad_clustering):
        """Higher silhouette = better clustering."""
        from sklearn.metrics import silhouette_score

        good_score = silhouette_score(*good_clustering)
        bad_score = silhouette_score(*bad_clustering)

        assert good_score > bad_score, "Silhouette should be higher for good clustering"

    def test_davies_bouldin_direction(self, good_clustering, bad_clustering):
        """Lower Davies-Bouldin = better clustering."""
        from sklearn.metrics import davies_bouldin_score

        good_score = davies_bouldin_score(*good_clustering)
        bad_score = davies_bouldin_score(*bad_clustering)

        assert good_score < bad_score, "Davies-Bouldin should be lower for good clustering"

    def test_calinski_harabasz_direction(self, good_clustering, bad_clustering):
        """Higher Calinski-Harabasz = better clustering."""
        from sklearn.metrics import calinski_harabasz_score

        good_score = calinski_harabasz_score(*good_clustering)
        bad_score = calinski_harabasz_score(*bad_clustering)

        assert good_score > bad_score, "Calinski-Harabasz should be higher for good clustering"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_two_clusters_minimum(self):
        """Metrics should work with minimum 2 clusters."""
        from sklearn.metrics import davies_bouldin_score, silhouette_score

        X, labels = make_blobs(n_samples=50, centers=2, random_state=42)

        # Should not raise
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)

        assert sil is not None
        assert db is not None

    def test_single_point_clusters(self):
        """Handle clusters with single points gracefully."""
        from metricate.core.metrics import sse

        X = np.array(
            [
                [0, 0],
                [0, 1],
                [0, 2],  # Cluster 0
                [10, 0],  # Cluster 1 (single point)
                [20, 0],
                [20, 1],  # Cluster 2
            ]
        )
        labels = np.array([0, 0, 0, 1, 2, 2])

        # Should not raise
        sse_val = sse(X, labels)
        assert sse_val >= 0


class TestMetricConsistency:
    """Tests for metric calculation consistency."""

    def test_deterministic_results(self):
        """Same input should produce same output."""
        from sklearn.metrics import davies_bouldin_score, silhouette_score

        X, labels = make_blobs(n_samples=100, centers=3, random_state=42)

        scores1 = {
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        }

        scores2 = {
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        }

        for metric in scores1:
            assert np.isclose(scores1[metric], scores2[metric]), (
                f"{metric} produced different results: {scores1[metric]} vs {scores2[metric]}"
            )
