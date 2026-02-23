"""
Flask web application for Metricate.

Provides a browser-based interface for:
- Uploading and evaluating clustering CSV files
- Comparing two clusterings
- Viewing results in formatted tables
"""

import os
import tempfile

from flask import Flask, jsonify, render_template, request

import metricate


def create_app(debug: bool = False) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        debug: Enable debug mode

    Returns:
        Configured Flask application
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["DEBUG"] = debug
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload

    @app.route("/")
    def index():
        """Render the main page with upload form."""
        metrics_list = metricate.list_metrics()
        total_metrics = len(metrics_list)
        return render_template("index.html", total_metrics=total_metrics)

    @app.route("/api/evaluate", methods=["POST"])
    def api_evaluate():
        """
        Evaluate a clustering CSV file.

        Accepts:
            - file: CSV file upload
            - exclude: Comma-separated metric names to exclude (optional)
            - force_all: "true" to force O(n²) metrics (optional)

        Returns:
            JSON with evaluation results or error message
        """
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith(".csv"):
            return jsonify({"error": "File must be a CSV"}), 400

        # Get optional parameters
        exclude_str = request.form.get("exclude", "")
        exclude = [x.strip() for x in exclude_str.split(",") if x.strip()] or None
        force_all = request.form.get("force_all", "false").lower() == "true"

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        try:
            # Run evaluation
            result = metricate.evaluate(tmp_path, exclude=exclude, force_all=force_all)

            # Convert to response format
            response = {
                "success": True,
                "filename": file.filename,
                "summary": {
                    "n_rows": result.n_rows,
                    "n_clusters": result.n_clusters,
                    "n_dimensions": result.n_dimensions,
                    "metrics_computed": len([m for m in result.metrics if m.value is not None]),
                    "metrics_skipped": len([m for m in result.metrics if m.value is None]),
                },
                "metrics": [
                    {
                        "name": m.metric,
                        "value": m.value,
                        "range": m.range,
                        "direction": m.direction,
                        "tier": m.tier,
                        "skipped": m.value is None,
                        "skip_reason": m.skip_reason,
                    }
                    for m in result.metrics
                ],
                "warnings": result.warnings,
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @app.route("/api/compare", methods=["POST"])
    def api_compare():
        """
        Compare two clustering CSV files.

        Accepts:
            - file1: First CSV file upload
            - file2: Second CSV file upload
            - name1: Display name for first file (optional)
            - name2: Display name for second file (optional)
            - exclude: Comma-separated metric names to exclude (optional)

        Returns:
            JSON with comparison results or error message
        """
        if "file1" not in request.files or "file2" not in request.files:
            return jsonify({"error": "Two files required for comparison"}), 400

        file1 = request.files["file1"]
        file2 = request.files["file2"]

        if file1.filename == "" or file2.filename == "":
            return jsonify({"error": "Both files must be selected"}), 400

        if not file1.filename.endswith(".csv") or not file2.filename.endswith(".csv"):
            return jsonify({"error": "Both files must be CSVs"}), 400

        # Get optional parameters
        name1 = request.form.get("name1", file1.filename)
        name2 = request.form.get("name2", file2.filename)
        exclude_str = request.form.get("exclude", "")
        exclude = [x.strip() for x in exclude_str.split(",") if x.strip()] or None

        # Save uploaded files temporarily
        tmp_paths = []
        try:
            for f in [file1, file2]:
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
                    f.save(tmp)
                    tmp_paths.append(tmp.name)

            # Run comparison
            result = metricate.compare(
                tmp_paths[0],
                tmp_paths[1],
                names=[name1, name2],
                exclude=exclude,
            )

            # Convert to response format
            response = {
                "success": True,
                "winner": result.winner,
                "summary": {
                    "wins": result.wins,
                    "ties": result.ties,
                    "total_metrics": result.wins[name1] + result.wins[name2] + result.ties,
                },
                "evaluations": {
                    name: {
                        "n_rows": eval_result.n_rows,
                        "n_clusters": eval_result.n_clusters,
                    }
                    for name, eval_result in result.evaluations.items()
                },
                "per_metric": [
                    {
                        "metric": row["Metric"],
                        "values": {name1: row.get(name1), name2: row.get(name2)},
                        "winner": row.get("Winner"),
                    }
                    for row in result.comparison_rows
                ],
                "warnings": result.warnings,
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

        finally:
            # Cleanup temp files
            for path in tmp_paths:
                if os.path.exists(path):
                    os.unlink(path)

    @app.route("/api/metrics", methods=["GET"])
    def api_metrics():
        """Return list of available metrics."""
        return jsonify(metricate.list_metrics())

    @app.route("/api/degradations", methods=["GET"])
    def api_degradations():
        """Return list of available degradation types."""
        return jsonify(metricate.list_degradations())

    return app


def run_server(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    """
    Run the Flask development server.

    Args:
        host: Host to bind to (default: 127.0.0.1)
        port: Port to listen on (default: 5000)
        debug: Enable debug mode
    """
    app = create_app(debug=debug)
    print(f"\n🚀 Metricate Web UI starting at http://{host}:{port}")
    print("   Press Ctrl+C to stop\n")
    app.run(host=host, port=port, debug=debug)
