"""Custom exceptions for Metricate."""


class MetricateError(Exception):
    """Base exception for all Metricate errors."""

    code: str = "METRICATE_ERROR"

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self):
        return f"{self.code}: {self.message}"


class FileNotFoundError(MetricateError):
    """Raised when a required file does not exist."""

    code = "FILE_NOT_FOUND"

    def __init__(self, path: str):
        super().__init__(f"Could not find file: {path}", details={"path": str(path)})


class InvalidCSVError(MetricateError):
    """Raised when a CSV file is malformed or cannot be parsed."""

    code = "INVALID_CSV"

    def __init__(self, path: str, reason: str):
        super().__init__(
            f"Invalid CSV file '{path}': {reason}", details={"path": str(path), "reason": reason}
        )


class ColumnNotFoundError(MetricateError):
    """Raised when required columns are missing from the data."""

    code = "COLUMN_NOT_FOUND"

    def __init__(
        self,
        missing: str | list[str],
        available: list[str] | None = None,
        message: str | None = None,
    ):
        if isinstance(missing, str):
            missing = [missing]

        if message:
            msg = message
        else:
            msg = f"Missing required column(s): {', '.join(missing)}"

        super().__init__(msg, details={"missing": missing, "available": available or []})


class InsufficientClustersError(MetricateError):
    """Raised when there are fewer than 2 clusters (excluding noise)."""

    code = "INSUFFICIENT_CLUSTERS"

    def __init__(self, found: int, required: int = 2):
        super().__init__(
            f"Found {found} cluster(s), but at least {required} are required "
            "(excluding noise points with label=-1)",
            details={"found": found, "required": required},
        )


class DimensionMismatchError(MetricateError):
    """Raised when comparing clusterings with different embedding dimensions."""

    code = "DIMENSION_MISMATCH"

    def __init__(self, dimensions_a: int, dimensions_b: int):
        super().__init__(
            f"Embedding dimensions do not match: A has {dimensions_a}, B has {dimensions_b}",
            details={"dimensions_a": dimensions_a, "dimensions_b": dimensions_b},
        )


class InvalidMetricError(MetricateError):
    """Raised when an invalid metric name is specified."""

    code = "INVALID_METRIC"

    def __init__(self, invalid_names: list[str], valid_names: list[str] | None = None):
        super().__init__(
            f"Unrecognized metric name(s): {', '.join(invalid_names)}",
            details={"invalid": invalid_names, "valid": valid_names or []},
        )


class ComputationError(MetricateError):
    """Raised when a metric computation fails."""

    code = "COMPUTATION_ERROR"

    def __init__(self, metric_name: str, reason: str):
        super().__init__(
            f"Failed to compute metric '{metric_name}': {reason}",
            details={"metric": metric_name, "reason": reason},
        )
