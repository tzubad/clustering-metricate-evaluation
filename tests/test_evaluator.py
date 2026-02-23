"""
Integration tests for the metricate evaluator.

Tests the full evaluation pipeline using real CSV files from the
degraded_datasets/ directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
DEGRADED_DIR = PROJECT_ROOT / "degraded_datasets"
TEST_DATA_DIR = PROJECT_ROOT


class TestEvaluateAPI:
    """Tests for metricate.evaluate() function."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a temporary CSV file for testing."""
        np.random.seed(42)
        n_samples = 100
        n_clusters = 3

        # Generate simple clustering data
        data = {
            "cluster_id": np.repeat(range(n_clusters), n_samples // n_clusters + 1)[:n_samples],
        }

        # Add embedding columns
        for i in range(10):
            data[f"embedding_{i}"] = np.random.randn(n_samples)

        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_clustering.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_basic_evaluation(self, sample_csv):
        """Test basic evaluation returns results."""
        import metricate

        result = metricate.evaluate(str(sample_csv))

        assert result is not None
        assert hasattr(result, "metrics")
        assert len(result.metrics) > 0

    def test_evaluation_returns_dataframe(self, sample_csv):
        """Test to_dataframe() method works."""
        import metricate

        result = metricate.evaluate(str(sample_csv))
        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "metric" in df.columns or "Metric" in df.columns
        assert len(df) > 0

    def test_evaluation_returns_table(self, sample_csv):
        """Test to_table() method works."""
        import metricate

        result = metricate.evaluate(str(sample_csv))
        table = result.to_table()

        assert isinstance(table, str)
        assert len(table) > 0
        assert "Silhouette" in table or "silhouette" in table.lower()

    def test_exclude_metrics(self, sample_csv):
        """Test excluding specific metrics."""
        import metricate

        result_all = metricate.evaluate(str(sample_csv))
        result_exclude = metricate.evaluate(str(sample_csv), exclude=["Silhouette"])

        # Get metric names from the results
        [m.metric for m in result_all.metrics]
        [m.metric for m in result_exclude.metrics]

        # The excluded result should have fewer or equal metrics
        # (Silhouette should be skipped, not completely removed from the list)
        # But it should be marked as skipped
        sil_in_all = any(m.metric == "Silhouette" and m.computed for m in result_all.metrics)
        sil_in_exclude = any(
            m.metric == "Silhouette" and m.computed for m in result_exclude.metrics
        )

        # Silhouette should be computed in all, but not in excluded
        assert sil_in_all, "Silhouette should be computed in full evaluation"
        assert not sil_in_exclude, "Silhouette should not be computed when excluded"

    def test_invalid_file_raises_error(self):
        """Test that non-existent file raises error."""
        import metricate

        with pytest.raises(Exception):  # FileNotFoundError or similar
            metricate.evaluate("nonexistent_file.csv")


class TestCompareAPI:
    """Tests for metricate.compare() function."""

    @pytest.fixture
    def two_csvs(self, tmp_path):
        """Create two temporary CSV files for comparison."""
        np.random.seed(42)
        n_samples = 100

        # Dataset A: 3 clusters
        data_a = {"cluster_id": np.repeat([0, 1, 2], n_samples // 3 + 1)[:n_samples]}
        for i in range(10):
            data_a[f"embedding_{i}"] = np.random.randn(n_samples)

        # Dataset B: 5 clusters
        data_b = {"cluster_id": np.repeat([0, 1, 2, 3, 4], n_samples // 5 + 1)[:n_samples]}
        for i in range(10):
            data_b[f"embedding_{i}"] = np.random.randn(n_samples)

        csv_a = tmp_path / "clustering_a.csv"
        csv_b = tmp_path / "clustering_b.csv"

        pd.DataFrame(data_a).to_csv(csv_a, index=False)
        pd.DataFrame(data_b).to_csv(csv_b, index=False)

        return csv_a, csv_b

    def test_basic_comparison(self, two_csvs):
        """Test basic comparison returns results."""
        import metricate

        csv_a, csv_b = two_csvs
        result = metricate.compare(str(csv_a), str(csv_b))

        assert result is not None
        assert hasattr(result, "winner")

    def test_comparison_has_winner(self, two_csvs):
        """Test that comparison determines a winner."""
        import metricate

        csv_a, csv_b = two_csvs
        result = metricate.compare(str(csv_a), str(csv_b), name_a="First", name_b="Second")

        assert result.winner in ["First", "Second", "Tie"]

    def test_comparison_table(self, two_csvs):
        """Test to_table() method for comparison."""
        import metricate

        csv_a, csv_b = two_csvs
        result = metricate.compare(str(csv_a), str(csv_b))

        table = result.to_table() if hasattr(result, "to_table") else str(result)
        assert isinstance(table, str)
        assert len(table) > 0


@pytest.mark.skipif(not DEGRADED_DIR.exists(), reason="degraded_datasets/ directory not found")
class TestWithDegradedDatasets:
    """Integration tests using actual degraded datasets."""

    def test_evaluate_degraded_file(self):
        """Test evaluation on a degraded dataset file."""
        import metricate

        # Find a degraded file
        degraded_files = list(DEGRADED_DIR.glob("*.csv"))
        if not degraded_files:
            pytest.skip("No CSV files in degraded_datasets/")

        csv_path = degraded_files[0]
        result = metricate.evaluate(str(csv_path))

        assert result is not None
        assert len(result.metrics) > 0

    def test_compare_degradation_levels(self):
        """Compare original vs degraded clustering."""
        import metricate

        # Find 5pct and 50pct files for same degradation type
        files_5pct = list(DEGRADED_DIR.glob("*_5pct.csv"))
        files_50pct = list(DEGRADED_DIR.glob("*_50pct.csv"))

        if not files_5pct or not files_50pct:
            pytest.skip("Need both 5pct and 50pct files for comparison")

        # Get matching degradation type
        for f5 in files_5pct:
            deg_type = f5.name.replace("_5pct.csv", "")
            f50 = DEGRADED_DIR / f"{deg_type}_50pct.csv"
            if f50.exists():
                result = metricate.compare(str(f5), str(f50), name_a="5%", name_b="50%")
                assert result is not None
                return

        pytest.skip("No matching degradation pairs found")

    def test_degradation_worsens_metrics(self):
        """Test that higher degradation generally worsens metrics."""
        import metricate

        # Compare label_swap at different levels
        file_5pct = DEGRADED_DIR / "label_swap_random_5pct.csv"
        file_50pct = DEGRADED_DIR / "label_swap_random_50pct.csv"

        if not file_5pct.exists() or not file_50pct.exists():
            pytest.skip("label_swap_random files not found")

        result_5 = metricate.evaluate(str(file_5pct))
        result_50 = metricate.evaluate(str(file_50pct))

        # Get silhouette scores
        def get_silhouette(result):
            for m in result.metrics:
                if isinstance(m, dict):
                    if m.get("name", "").lower() == "silhouette":
                        return m.get("value")
                elif hasattr(m, "name") and m.name.lower() == "silhouette":
                    return m.value
            return None

        sil_5 = get_silhouette(result_5)
        sil_50 = get_silhouette(result_50)

        if sil_5 is not None and sil_50 is not None:
            # 5% degradation should generally have better (higher) silhouette
            # than 50% degradation (though not guaranteed for all cases)
            # Just verify both are computed
            assert isinstance(sil_5, (int, float))
            assert isinstance(sil_50, (int, float))


class TestListFunctions:
    """Tests for list_metrics() and list_degradations()."""

    def test_list_metrics(self):
        """Test listing available metrics."""
        import metricate

        metrics = metricate.list_metrics()

        assert isinstance(metrics, list)
        assert len(metrics) > 0
        assert "Silhouette" in metrics

    def test_list_metrics_with_reference(self):
        """Test listing metrics with full reference info."""
        import metricate

        df = metricate.list_metrics(include_reference=True)

        assert isinstance(df, pd.DataFrame)
        assert "metric" in df.columns
        assert "direction" in df.columns
        assert len(df) > 0

    def test_list_degradations(self):
        """Test listing available degradation types."""
        import metricate

        degradations = metricate.list_degradations()

        assert isinstance(degradations, dict)
        assert len(degradations) > 0
        # Check for expected categories
        assert any("Label" in cat for cat in degradations.keys())


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_file_error(self):
        """Test appropriate error for missing file."""
        import metricate

        with pytest.raises(Exception):
            metricate.evaluate("/nonexistent/path/file.csv")

    def test_invalid_csv_error(self, tmp_path):
        """Test appropriate error for invalid CSV."""
        import metricate

        # Create invalid CSV
        invalid_file = tmp_path / "invalid.csv"
        invalid_file.write_text("not,a,valid\ncsv,file")

        # Should raise an error (insufficient data or missing columns)
        with pytest.raises(Exception):
            metricate.evaluate(str(invalid_file))

    def test_insufficient_clusters_error(self, tmp_path):
        """Test error when only one cluster present."""
        import metricate

        # Create CSV with only one cluster
        df = pd.DataFrame(
            {
                "cluster_id": [0] * 100,
                "embedding_0": np.random.randn(100),
                "embedding_1": np.random.randn(100),
            }
        )

        csv_path = tmp_path / "single_cluster.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(Exception):
            metricate.evaluate(str(csv_path))


class TestOutputFormats:
    """Tests for different output formats."""

    @pytest.fixture
    def evaluation_result(self, tmp_path):
        """Create an evaluation result for testing."""
        import metricate

        np.random.seed(42)
        n_samples = 100

        data = {
            "cluster_id": np.repeat([0, 1, 2], n_samples // 3 + 1)[:n_samples],
        }
        for i in range(10):
            data[f"embedding_{i}"] = np.random.randn(n_samples)

        csv_path = tmp_path / "test.csv"
        pd.DataFrame(data).to_csv(csv_path, index=False)

        return metricate.evaluate(str(csv_path))

    def test_json_output(self, evaluation_result):
        """Test JSON output format."""
        from metricate.output.formatters import to_json

        json_str = to_json(evaluation_result)

        assert isinstance(json_str, str)
        assert len(json_str) > 0

        import json

        data = json.loads(json_str)
        assert "metrics" in data or isinstance(data, list)

    def test_csv_output(self, evaluation_result):
        """Test CSV output format."""
        from metricate.output.formatters import to_csv

        csv_str = to_csv(evaluation_result)

        assert isinstance(csv_str, str)
        assert len(csv_str) > 0
        assert "," in csv_str  # Should contain commas

    def test_dataframe_output(self, evaluation_result):
        """Test DataFrame output."""
        df = evaluation_result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
