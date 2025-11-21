"""
Smoke test for 05_retrain_wine_multiclass.ipynb

This test runs the notebook headless (using nbconvert) and verifies that:
1. The notebook executes without errors
2. The expected model artifacts are created (best_wine_model.pkl and best_wine_scaler.pkl)
"""

import os
import subprocess
import pytest
from pathlib import Path


class TestWineNotebookSmoke:
    """Smoke tests for the Wine multiclass retraining notebook."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_and_teardown(self):
        """Run the notebook before tests and clean up after."""
        models_dir = Path("models")

        # Ensure models directory exists
        models_dir.mkdir(exist_ok=True)

        # Clean up any previous artifacts before test
        model_file = models_dir / "best_wine_model.pkl"
        scaler_file = models_dir / "best_wine_scaler.pkl"
        if model_file.exists():
            model_file.unlink()
        if scaler_file.exists():
            scaler_file.unlink()

        # Run the notebook headless with nbconvert from the project root
        # The notebook uses relative paths (e.g., 'models/best_wine_model.pkl')
        # so we must run from the project root directory, not from notebooks/
        cmd = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "notebooks/05_retrain_wine_multiclass.ipynb",
            "--output",
            "05_retrain_wine_multiclass_executed.ipynb",
            "--ExecutePreprocessor.timeout=300",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=os.getcwd(),  # Run from current working directory (project root)
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                pytest.fail(
                    f"Notebook execution failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                )
        except subprocess.TimeoutExpired:
            pytest.fail("Notebook execution timed out after 600 seconds")
        except FileNotFoundError:
            pytest.skip(
                "jupyter nbconvert not found; install jupyterlab to run this test"
            )

        yield

        # Clean up the temporary executed notebook if desired (optional)
        executed_notebook = Path("05_retrain_wine_multiclass_executed.ipynb")
        if executed_notebook.exists():
            executed_notebook.unlink()

    def test_best_wine_model_pkl_created(self):
        """Assert that best_wine_model.pkl was created during notebook execution."""
        model_file = Path("models/best_wine_model.pkl")
        assert (
            model_file.exists()
        ), f"Expected {model_file} to be created by the notebook"
        assert model_file.stat().st_size > 0, f"{model_file} exists but is empty"

    def test_best_wine_scaler_pkl_created(self):
        """Assert that best_wine_scaler.pkl was created during notebook execution."""
        scaler_file = Path("models/best_wine_scaler.pkl")
        assert (
            scaler_file.exists()
        ), f"Expected {scaler_file} to be created by the notebook"
        assert scaler_file.stat().st_size > 0, f"{scaler_file} exists but is empty"

    def test_both_artifacts_exist(self):
        """Assert that both model and scaler artifacts exist and are non-empty."""
        model_file = Path("models/best_wine_model.pkl")
        scaler_file = Path("models/best_wine_scaler.pkl")
        assert (
            model_file.exists() and scaler_file.exists()
        ), "Both model and scaler files should exist"
        assert (
            model_file.stat().st_size > 0 and scaler_file.stat().st_size > 0
        ), "Both files should be non-empty"
