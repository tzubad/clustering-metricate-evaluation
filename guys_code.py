#!/usr/bin/env python3
"""
Script to evaluate KMeans clustering with different k values.

This script:
1. Takes as input a CSV with reduced embeddings (from add_reduced_embeddings.py)
2. Runs KMeans clustering with different k values
3. Compares each run against original clusters using:
   - ARI (Adjusted Rand Index)
   - NMI (Normalized Mutual Information)
   - V-measure
4. Calculates intrinsic metrics for each k:
   - Silhouette Score
   - Davies-Bouldin Index
   - Calinski-Harabasz Score
   - Inertia
5. Outputs results as a CSV table

Usage:
    uv run python narratives_v2/evaluation/evaluate_kmeans_clustering.py \
        --input dataset_with_reduced.csv \
        --output kmeans_evaluation.csv \
        --k-min 10 \
        --k-max 100 \
        --k-step 5
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


def load_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load reduced embeddings and original cluster labels from CSV.

    Args:
        csv_path: Path to CSV file

    Returns:
        Tuple of (embeddings_array, original_labels)
    """
    logger.info(f"Loading data from {csv_path}")

    embeddings = []
    labels = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Load reduced embedding
            reduced_emb = json.loads(row["reduced_embedding"])
            embeddings.append(reduced_emb)

            # Load original cluster label
            labels.append(int(row["cluster_id"]))

            if (i + 1) % 10000 == 0:
                logger.info(f"Loaded {i + 1} rows...")

    embeddings_array = np.array(embeddings, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.int32)

    logger.info(f"Loaded {len(embeddings)} samples")
    logger.info(f"Embedding shape: {embeddings_array.shape}")
    logger.info(f"Original clusters: {len(set(labels))} unique (including noise)")
    logger.info(f"Noise points (cluster -1): {np.sum(labels_array == -1)}")

    return embeddings_array, labels_array


def evaluate_kmeans(
    embeddings: np.ndarray,
    original_labels: np.ndarray,
    k_values: list[int],
    random_state: int = 42,
) -> list[dict]:
    """
    Run KMeans with different k values and evaluate.

    Args:
        embeddings: Embedding vectors (n_samples, n_features)
        original_labels: Original cluster assignments
        k_values: List of k values to try
        random_state: Random seed

    Returns:
        List of evaluation result dictionaries
    """
    results = []

    # Filter out noise points for comparison metrics that require them
    # (but keep them for intrinsic metrics)
    non_noise_mask = original_labels != -1
    embeddings[non_noise_mask]
    non_noise_labels = original_labels[non_noise_mask]

    logger.info(f"Will evaluate {len(k_values)} different k values")
    logger.info(f"Non-noise samples for comparison metrics: {len(non_noise_labels)}")

    for i, k in enumerate(k_values):
        logger.info(f"\nEvaluating k={k} ({i + 1}/{len(k_values)})...")

        # Run KMeans on all data
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings)

        # Get labels for non-noise points only (for comparison metrics)
        kmeans_labels_non_noise = kmeans_labels[non_noise_mask]

        # Calculate comparison metrics (against original, excluding noise)
        ari = adjusted_rand_score(non_noise_labels, kmeans_labels_non_noise)
        nmi = normalized_mutual_info_score(
            non_noise_labels, kmeans_labels_non_noise, average_method="arithmetic"
        )
        v_measure = v_measure_score(non_noise_labels, kmeans_labels_non_noise)

        # Calculate intrinsic metrics (on all data including noise points)
        silhouette = silhouette_score(embeddings, kmeans_labels)
        davies_bouldin = davies_bouldin_score(embeddings, kmeans_labels)
        calinski_harabasz = calinski_harabasz_score(embeddings, kmeans_labels)
        inertia = kmeans.inertia_

        result = {
            "k": k,
            "ari": ari,
            "nmi": nmi,
            "v_measure": v_measure,
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "calinski_harabasz": calinski_harabasz,
            "inertia": inertia,
        }

        results.append(result)

        logger.info(f"  ARI: {ari:.4f}")
        logger.info(f"  NMI: {nmi:.4f}")
        logger.info(f"  V-measure: {v_measure:.4f}")
        logger.info(f"  Silhouette: {silhouette:.4f}")
        logger.info(f"  Davies-Bouldin: {davies_bouldin:.4f}")
        logger.info(f"  Calinski-Harabasz: {calinski_harabasz:.2f}")
        logger.info(f"  Inertia: {inertia:.2f}")

    return results


def save_results(results: list[dict], output_path: Path):
    """
    Save evaluation results to CSV.

    Args:
        results: List of evaluation dictionaries
        output_path: Path to output CSV file
    """
    logger.info(f"Saving results to {output_path}")

    fieldnames = [
        "k",
        "ari",
        "nmi",
        "v_measure",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "inertia",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result)

    logger.success(f"Results saved to {output_path}")


def print_summary(results: list[dict]):
    """
    Print summary of best results.

    Args:
        results: List of evaluation dictionaries
    """
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY OF BEST RESULTS")
    logger.info("=" * 80)

    # Find best k for each metric
    best_ari = max(results, key=lambda x: x["ari"])
    best_nmi = max(results, key=lambda x: x["nmi"])
    best_v_measure = max(results, key=lambda x: x["v_measure"])
    best_silhouette = max(results, key=lambda x: x["silhouette"])
    best_davies_bouldin = min(results, key=lambda x: x["davies_bouldin"])
    best_calinski = max(results, key=lambda x: x["calinski_harabasz"])

    logger.info("\nComparison Metrics (vs original clustering):")
    logger.info(f"  Best ARI: k={best_ari['k']}, score={best_ari['ari']:.4f}")
    logger.info(f"  Best NMI: k={best_nmi['k']}, score={best_nmi['nmi']:.4f}")
    logger.info(
        f"  Best V-measure: k={best_v_measure['k']}, score={best_v_measure['v_measure']:.4f}"
    )

    logger.info("\nIntrinsic Quality Metrics:")
    logger.info(
        f"  Best Silhouette: k={best_silhouette['k']}, score={best_silhouette['silhouette']:.4f}"
    )
    logger.info(
        f"  Best Davies-Bouldin (lower is better): k={best_davies_bouldin['k']}, score={best_davies_bouldin['davies_bouldin']:.4f}"
    )
    logger.info(
        f"  Best Calinski-Harabasz: k={best_calinski['k']}, score={best_calinski['calinski_harabasz']:.2f}"
    )

    # Find k that appears most often as best
    best_ks = [
        best_ari["k"],
        best_nmi["k"],
        best_v_measure["k"],
        best_silhouette["k"],
        best_davies_bouldin["k"],
        best_calinski["k"],
    ]
    from collections import Counter

    k_counts = Counter(best_ks)
    most_common_k = k_counts.most_common(1)[0][0]

    logger.info(
        f"\nMost frequently best k: {most_common_k} (best in {k_counts[most_common_k]} metrics)"
    )
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KMeans clustering with different k values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate k from 10 to 100 in steps of 5
  uv run python narratives_v2/evaluation/evaluate_kmeans_clustering.py \\
    --input narrative_dataset_model_1247315_with_reduced.csv \\
    --output kmeans_evaluation_1247315.csv \\
    --k-min 10 \\
    --k-max 100 \\
    --k-step 5

  # Evaluate specific k values
  uv run python narratives_v2/evaluation/evaluate_kmeans_clustering.py \\
    --input narrative_dataset_model_1230497_with_reduced.csv \\
    --output kmeans_evaluation_1230497.csv \\
    --k-min 10 \\
    --k-max 50 \\
    --k-step 2

Metrics Explained:
  Comparison metrics (compare KMeans vs original HDBSCAN clustering):
    - ARI (Adjusted Rand Index): -1 to 1, higher is better (1 = perfect match)
    - NMI (Normalized Mutual Info): 0 to 1, higher is better (1 = perfect match)
    - V-measure: 0 to 1, higher is better (harmonic mean of homogeneity and completeness)

  Intrinsic quality metrics (measure KMeans clustering quality):
    - Silhouette Score: -1 to 1, higher is better (1 = well-separated clusters)
    - Davies-Bouldin Index: 0 to ∞, lower is better (0 = perfect separation)
    - Calinski-Harabasz Score: 0 to ∞, higher is better (higher = denser, well-separated)
    - Inertia: 0 to ∞, lower is better (sum of squared distances to centroids)
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file with reduced embeddings",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file for results",
    )

    parser.add_argument(
        "--k-min",
        type=int,
        default=10,
        help="Minimum k value to test (default: 10)",
    )

    parser.add_argument(
        "--k-max",
        type=int,
        default=100,
        help="Maximum k value to test (default: 100)",
    )

    parser.add_argument(
        "--k-step",
        type=int,
        default=5,
        help="Step size for k values (default: 5)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Validate paths
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate k values
    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    logger.info(f"Will test k values: {k_values}")

    try:
        # Load data
        embeddings, original_labels = load_data(input_path)

        # Run evaluation
        results = evaluate_kmeans(embeddings, original_labels, k_values, args.random_state)

        # Save results
        save_results(results, output_path)

        # Print summary
        print_summary(results)

        logger.success(f"\nEvaluation complete! Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
