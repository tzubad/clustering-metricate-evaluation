"""Web UI functionality for Metricate.

Provides a Flask-based web interface for:
- Evaluating clustering CSV files
- Comparing two clusterings
- Viewing results in formatted tables
"""

from metricate.web.app import create_app, run_server

__all__ = ["create_app", "run_server"]
