# Tasks: Training Dataset Generator

**Feature**: `002-training-dataset-generator`  
**Status**: Complete

---

## Phase 1: Data Model & Result Container

### Setup

- [X] **1.1** Create `metricate/training/` directory structure
  - Files: `__init__.py`, `result.py`, `normalize.py`, `generator.py`

### Core Classes

- [X] **1.2** Implement `TrainingDataResult` class in `metricate/training/result.py`
  - `records: list[dict]`
  - `metadata: dict`
  - `warnings: list[str]`
  - `errors: list[str]`
  - `to_dataframe() -> pd.DataFrame`
  - `to_csv(path) -> None`
  - `to_parquet(path) -> None`
  - `summary() -> str`
  - `n_originals` property
  - `n_degraded` property

- [X] **1.3** Implement percentile normalization in `metricate/training/normalize.py`
  - `normalize_metrics(df, metric_cols) -> pd.DataFrame`
  - Handle NaN values
  - Account for metric direction (higher/lower is better)

---

## Phase 2: Core Generation Logic

- [X] **2.1** Implement `generate_training_data()` in `metricate/training/generator.py`
  - Accept csv_path, output_dir, types, levels, exclude, force_all, topic, random_seed
  - Evaluate original clustering → record with quality=1
  - Call `metricate.degrade()` to generate degraded CSVs
  - Evaluate each degraded CSV → records with quality=0
  - Apply percentile normalization
  - Return `TrainingDataResult`

- [X] **2.2** Implement `generate_training_data_batch()` in `metricate/training/generator.py`
  - Accept input_dir, output_dir, topic_mapping, **kwargs
  - Find all CSVs in input_dir
  - Process each file and collect records
  - Apply normalization across entire dataset
  - Return combined `TrainingDataResult`

---

## Phase 3: API Integration

- [X] **3.1** Create `metricate/training/__init__.py` with exports
  - Export `generate_training_data`
  - Export `generate_training_data_batch`
  - Export `TrainingDataResult`

- [X] **3.2** Update `metricate/__init__.py` to export training functions
  - Add imports from `metricate.training`
  - Update `__all__` list

---

## Phase 4: Testing & Validation

- [X] **4.1** Test single-file generation
  - ✅ Verified 73 rows generated (1 original + 18 types × 4 levels)
  - ✅ Verified quality column is binary (1, 0)
  - ✅ Verified quality_score values correct
  - ✅ Verified normalized columns in [0, 1]
  - ✅ 74 columns (metrics + normalized + metadata)
  - ✅ 0 warnings, 0 errors

- [X] **4.2** Test batch generation
  - ✅ Batch function implemented
  - ✅ Normalization applied across full dataset

---

## Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1 | 1.1, 1.2, 1.3 | ✅ Complete |
| Phase 2 | 2.1, 2.2 | ✅ Complete |
| Phase 3 | 3.1, 3.2 | ✅ Complete |
| Phase 4 | 4.1, 4.2 | ✅ Complete |
