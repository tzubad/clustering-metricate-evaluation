# Feature Specification: Metricate - Clustering Evaluation Product

**Feature Branch**: `001-metricate-product`  
**Created**: February 23, 2026  
**Status**: Draft  
**Input**: User description: "Metricate product - clustering evaluation tool with metrics, comparison, and degradation visualization"

## Clarifications

### Session 2026-02-23

- Q: Should redundant/obsolete metrics be included in the default calculation? → A: Exclude 5 redundant metrics (Trace_W, Baker-Hubert Gamma, Sym-index, Rand Index, Log_SS_Ratio) - calculate 34 by default
- Q: What user interface should Metricate provide? → A: Python module with CLI and simple web UI (web UI lowest priority)
- Q: What output format should metric results use? → A: Formatted table as default, with JSON and CSV export options
- Q: How to handle O(n²) metrics for large datasets? → A: Auto-skip above 50k rows with warning, provide --force-all flag; document in README
- Q: How to validate comparison mode inputs? → A: Allow different row counts but display prominent warning about comparison validity

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Single Clustering Evaluation (Priority: P1)

A data scientist has produced a clustering result and wants to understand its quality. They provide a CSV file containing posts with cluster labels and embeddings. Metricate calculates all applicable clustering quality metrics and returns a comprehensive report showing each metric's score along with a reminder of the metric's valid range and interpretation direction (higher/lower is better).

**Why this priority**: This is the core functionality - without single clustering evaluation, the product has no value. It provides immediate, actionable insights about clustering quality.

**Independent Test**: Upload a CSV with clustering results and verify that all metrics are calculated and displayed with proper range context.

**Acceptance Scenarios**:

1. **Given** a user has a CSV file with columns: post content, cluster_id, and reduced_embedding (or dimension columns), **When** they submit the file to Metricate, **Then** the system returns scores for all 34 applicable metrics with their ranges and optimal directions.

2. **Given** a user uploads a valid clustering CSV, **When** metrics are calculated, **Then** each metric displays: metric name, calculated value, valid range (e.g., [-1, 1]), and direction indicator (↑ higher is better OR ↓ lower is better).

3. **Given** a user uploads a CSV with missing or malformed embeddings, **When** processing occurs, **Then** the system displays a clear error message indicating which rows/columns have issues.

---

### User Story 2 - Metric Exclusion (Priority: P2)

A researcher knows that certain metrics are not relevant to their analysis (e.g., external metrics when no ground truth exists, or computationally expensive metrics for large datasets). They want to exclude specific metrics from calculation to save time and reduce noise in results.

**Why this priority**: Builds on P1 functionality, allowing customization. Essential for practical use when users know which metrics are irrelevant to their analysis.

**Independent Test**: Submit a clustering with a list of metrics to exclude and verify those metrics are not calculated or displayed.

**Acceptance Scenarios**:

1. **Given** a user submits a clustering CSV with a list of metrics to exclude (e.g., ["Dunn Index", "S_Dbw"]), **When** Metricate processes the request, **Then** only the remaining metrics are calculated and returned.

2. **Given** a user excludes external metrics (ARI, Van Dongen, VI, Omega), **When** no ground truth labels are available, **Then** the system skips these metrics without errors.

3. **Given** a user provides an invalid metric name in the exclusion list, **When** processing occurs, **Then** the system warns about unrecognized metrics but continues processing valid ones.

---

### User Story 3 - Two Clustering Comparison (Priority: P2)

A data scientist has two competing clustering solutions (e.g., different k values, different algorithms) and wants to objectively determine which is better. They submit two clustering CSVs, and Metricate calculates metrics for both, then indicates which clustering "wins" based on which one has better scores on more metrics.

**Why this priority**: Same priority as P2 (exclusion) as both extend core functionality. Comparison is a common use case when tuning clustering parameters.

**Independent Test**: Submit two clustering CSVs and verify the system correctly identifies the winner based on metric wins.

**Acceptance Scenarios**:

1. **Given** a user submits two clustering CSV files, **When** Metricate processes both, **Then** the system calculates all metrics for each clustering and displays a side-by-side comparison (with warning if row counts differ).

2. **Given** two clusterings are evaluated, **When** displaying results, **Then** each metric row shows: Clustering A value, Clustering B value, and a winner indicator (A or B) based on optimal direction.

3. **Given** Clustering A wins 20 metrics and Clustering B wins 14 metrics, **When** summarizing results, **Then** the system declares Clustering A as the overall winner with a clear win count summary (e.g., "Clustering A wins: 20/34 metrics").

4. **Given** two clusterings tie on a specific metric (equal values), **When** determining winners, **Then** that metric is marked as a tie and excluded from the win count.

5. **Given** two clustering files with different row counts (e.g., 10,000 vs 8,500), **When** comparison is requested, **Then** the system proceeds with comparison but displays a prominent warning: "⚠️ WARNING: Clustering A has 10,000 points, Clustering B has 8,500 points (15% difference). Metric comparisons may be misleading when underlying datasets differ. Interpret results with caution."

---

### User Story 4 - Degradation Generation Module (Priority: P3)

A researcher wants to test how robust their clustering metrics are to various types of degradation. They provide a baseline clustering CSV, and the degradation module generates multiple degraded versions (label swaps, noise injection, cluster merges, etc.) along with interactive HTML visualizations showing the degradation effects.

**Why this priority**: Advanced functionality that extends the product's research capabilities. Depends on P1 core functionality being stable.

**Independent Test**: Submit a baseline clustering and verify degraded datasets are generated with corresponding HTML visualizations.

**Acceptance Scenarios**:

1. **Given** a user provides a clustering CSV, **When** they request degradation generation, **Then** the system produces degraded datasets for all 19 degradation types at multiple severity levels (5%, 10%, 25%, 50%).

2. **Given** degraded datasets are generated, **When** the process completes, **Then** interactive HTML visualizations are created showing embedding space with color-coded degradation effects.

3. **Given** a degradation suite is generated, **When** the user views results, **Then** a manifest CSV lists all generated files with degradation type, level, and file path.

4. **Given** the degradation module runs, **When** outputting HTMLs, **Then** an index.html is created linking to all individual degradation visualizations for easy navigation.

---

### User Story 5 - Web UI for Browser-Based Access (Priority: P4)

A non-technical stakeholder or researcher without Python setup wants to quickly evaluate a clustering without writing code. They access a simple web interface, upload their CSV file, and receive a formatted report in the browser.

**Why this priority**: Lowest priority as it adds deployment complexity. Core users (data scientists) are served by module + CLI. Web UI expands accessibility but is not essential for MVP.

**Independent Test**: Start the web server, upload a CSV via browser, and verify metrics are displayed correctly.

**Acceptance Scenarios**:

1. **Given** a user accesses the Metricate web interface, **When** they upload a valid clustering CSV, **Then** they see all 34 metrics displayed with ranges and directions in a formatted table.

2. **Given** the web UI is running, **When** a user uploads two files for comparison, **Then** they see a side-by-side comparison with the winner highlighted.

---

### Edge Cases

- What happens when a clustering has only 1 cluster? System should return an error or limited metrics that apply.
- What happens when embeddings have NaN or infinite values? System should identify and report problematic rows.
- How does the system handle very large files (>100k rows)? Auto-skip O(n²) metrics above 50k rows with warning; user can override with --force-all flag.
- What happens when comparing two clusterings with different numbers of data points? System allows comparison but displays prominent warning about potentially misleading results.
- How does the system handle clusters with fewer than 2 points? Some metrics require minimum cluster sizes.

## Requirements *(mandatory)*

### Functional Requirements

#### Core Evaluation (P1)

- **FR-001**: System MUST accept a CSV file with columns for post content, cluster labels (cluster_id), and embeddings (either reduced_embedding as string or dim_0 through dim_N columns).
- **FR-002**: System MUST calculate all 34 applicable clustering metrics from the following categories (excluding 5 redundant metrics: Trace_W, Baker-Hubert Gamma, Sym-index, Rand Index, Log_SS_Ratio):
  - **Internal Original (6)**: Silhouette, Dunn Index, Davies-Bouldin, Calinski-Harabasz, SSE, NCI
  - **Tier 1 CVIs (5)**: Ball-Hall, Ratkowsky-Lance, Ray-Turi, RMSSTD, R-squared, Wemmert-Gancarski
  - **Tier 2 CVIs (14)**: CS index, COP, S_Dbw, Det Ratio, Gamma, Generalized Dunn, G-plus, I-index (PBM), Log_Det_Ratio, McClain-Rao, Point-Biserial, SD validity, Tau, Trace_WiB, Ksq_DetW
  - **Tier 3 CVIs (5)**: Banfield-Raftery, Negentropy, NIVA, Score Function, Scott-Symons
  - **External (4)**: Adjusted Rand Index, Van Dongen, Variation of Information, Omega (only when ground truth provided)
- **FR-003**: System MUST display metric ranges and optimal directions for each calculated metric according to this reference:

| Metric | Range | Direction |
|--------|-------|-----------|
| Silhouette | [-1, 1] | ↑ Higher |
| Davies-Bouldin | [0, ∞) | ↓ Lower |
| Calinski-Harabasz | [0, ∞) | ↑ Higher |
| Dunn Index | [0, ∞) | ↑ Higher |
| SSE | [0, ∞) | ↓ Lower |
| NCI | [-1, 1] | ↑ Higher |
| Adjusted Rand Index | [-1, 1] | ↑ Higher |
| Van Dongen | [0, 1] | ↓ Lower |
| Variation of Information | [0, log(n)] | ↓ Lower |
| Omega | [0, 1] | ↓ Lower |
| Ball-Hall | [0, ∞) | ↓ Lower |
| Ratkowsky-Lance | [0, 1] | ↑ Higher |
| Ray-Turi | [0, ∞) | ↓ Lower |
| RMSSTD | [0, ∞) | ↓ Lower |
| R-squared | [0, 1] | ↑ Higher |
| Wemmert-Gancarski | [0, 1] | ↑ Higher |
| CS index | [0, ∞) | ↓ Lower |
| COP | [0, ∞) | ↓ Lower |
| S_Dbw | [0, ∞) | ↓ Lower |
| Det Ratio | [0, ∞) | ↑ Higher |
| Gamma | [-1, 1] | ↑ Higher |
| Generalized Dunn | [0, ∞) | ↑ Higher |
| G-plus | [0, 1] | ↓ Lower |
| I-index (PBM) | [0, ∞) | ↑ Higher |
| Log_Det_Ratio | (-∞, ∞) | ↑ Higher |
| McClain-Rao | [0, ∞) | ↓ Lower |
| Point-Biserial | [-1, 1] | ↑ Higher |
| SD validity | [0, ∞) | ↓ Lower |
| Tau | [-1, 1] | ↑ Higher |
| Trace_WiB | [0, ∞) | ↑ Higher |
| Ksq_DetW | [0, ∞) | ↓ Lower |
| Banfield-Raftery | (-∞, ∞) | ↓ Lower |
| Negentropy | (-∞, ∞) | ↓ Lower |
| NIVA | [0, ∞) | ↓ Lower |
| Score Function | [0, 1] | ↑ Higher |
| Scott-Symons | (-∞, ∞) | ↓ Lower |


- **FR-004**: System MUST validate input CSV and provide clear error messages for: missing required columns, invalid embedding formats, empty files, single-cluster datasets.

#### Performance Handling (P1)

- **FR-004a**: System MUST auto-skip O(n²) complexity metrics (Gamma, Tau, Point-Biserial, G-plus, McClain-Rao, NIVA) when dataset exceeds 50,000 rows.
- **FR-004b**: System MUST display a warning when metrics are auto-skipped due to dataset size, listing which metrics were skipped.
- **FR-004c**: System MUST provide a `--force-all` flag (CLI) or `force_all=True` parameter (API) to override auto-skip behavior.
- **FR-004d**: README documentation MUST explain the 50k threshold, which metrics are affected, and how to override.

#### Metric Exclusion (P2)

- **FR-005**: System MUST accept an optional list of metric names to exclude from calculation.
- **FR-006**: System MUST skip excluded metrics entirely (not calculate them) to save computation time.
- **FR-007**: System MUST warn users about unrecognized metric names in the exclusion list but continue processing valid metrics.

#### Comparison Mode (P2)

- **FR-008**: System MUST accept two clustering CSV files for comparison.
- **FR-009**: System MUST allow comparison of clusterings with different row counts, but MUST display a prominent warning (⚠️ WARNING) when row counts differ, explaining that metric comparisons may be misleading when comparing different underlying datasets.
- **FR-009a**: Warning message MUST include: row count for each file, percentage difference, and explicit statement that results should be interpreted with caution.
- **FR-010**: System MUST calculate metrics for both clusterings and produce a side-by-side comparison.
- **FR-011**: System MUST determine a winner for each metric based on the optimal direction (↑ or ↓).
- **FR-012**: System MUST declare an overall winner based on which clustering wins more metrics.
- **FR-013**: System MUST handle ties gracefully, excluding tied metrics from the win count.

#### Degradation Module (P3)

- **FR-014**: System MUST generate degraded datasets for 19 degradation types:
  - Label modifications: label_swap_random, label_swap_neighboring, label_swap_distant, boundary_reassignment
  - Cluster structure: merge_nearest, merge_farthest, merge_random, split_largest, split_loosest, split_random
  - Point removal: random_removal, core_removal, remove_smallest_clusters, remove_largest_clusters, remove_tightest_clusters
  - Embedding perturbation: noise_injection, embedding_perturbation, centroid_displacement
- **FR-015**: System MUST generate each degradation at multiple severity levels (5%, 10%, 25%, 50%).
- **FR-016**: System MUST create interactive HTML visualizations for each degradation showing embedding space with color-coded effects.
- **FR-017**: System MUST generate a manifest CSV listing all created files with metadata.
- **FR-018**: System MUST create an index.html linking to all visualization files.
- **FR-019**: System MUST reuse existing degradation toolkit code from the workspace (ClusteringDegrader class).

#### Interface - Python Module (P1)

- **FR-020**: System MUST be importable as a Python module (e.g., `from metricate import evaluate, compare`).
- **FR-021**: System MUST provide a programmatic API that returns results as Python dictionaries or DataFrames.
- **FR-021a**: Default output for programmatic API MUST be a pandas DataFrame with columns: metric_name, value, range, direction.

#### Interface - CLI (P2)

- **FR-022**: System MUST provide a command-line interface with commands: `metricate evaluate <file>`, `metricate compare <file1> <file2>`, `metricate degrade <file>`.
- **FR-023**: CLI MUST support `--exclude` flag to specify metrics to skip.
- **FR-024**: CLI MUST support `--output` flag to specify output format (json, csv, table). Default is formatted table.

#### Interface - Web UI (P4)

- **FR-025**: System MUST provide a simple web interface for browser-based file upload and evaluation.
- **FR-026**: Web UI MUST display results in a formatted, readable table with metric ranges and directions.
- **FR-027**: Web UI MUST support comparison mode with side-by-side display.

### Key Entities

- **Clustering**: A dataset with posts assigned to clusters, represented by a CSV with post content, cluster_id, and embeddings. Key attributes: number of clusters, number of points, embedding dimensionality.
- **Metric Result**: A single metric calculation outcome containing: metric name, calculated value, valid range, optimal direction, and interpretation.
- **Comparison Report**: A comparison of two clusterings containing: per-metric scores for both, winners per metric, overall winner, and win counts.
- **Degradation Suite**: A collection of degraded datasets with: degradation type, severity level, output file path, and HTML visualization path.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can upload a clustering CSV and receive metric scores within 30 seconds for datasets under 10,000 points.
- **SC-002**: All 34 metrics display with correct ranges and direction indicators as specified in the reference table.
- **SC-003**: Comparison mode correctly identifies the better clustering in 100% of test cases where one clustering is objectively superior.
- **SC-004**: Degradation module generates all 19 degradation types × 4 severity levels = 76 degraded datasets per run.
- **SC-005**: All generated HTML visualizations load correctly and display interactive embedding plots.
- **SC-006**: Users can exclude metrics and see reduced computation time proportional to excluded metric complexity.
- **SC-007**: Error messages clearly indicate the specific issue when invalid input is provided.

## Assumptions

- Users have clustering results in CSV format with embeddings already computed (10-dimensional reduced embeddings from PCA/UMAP).
- The primary use case is evaluating text clustering results from NLP pipelines.
- External metrics (ARI, Van Dongen, VI, Omega) will only be calculated when comparing against a ground truth or baseline clustering.
- The degradation module will use the existing `ClusteringDegrader` class from `degradation_toolkit.py`.
- The `calculate_all_metrics` function from the notebook will be extracted and reused.
- Interactive visualizations will use Plotly for HTML generation.
