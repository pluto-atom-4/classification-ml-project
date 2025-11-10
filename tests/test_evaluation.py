"""
Tests for model evaluation module.

Run with: pytest tests/test_evaluation.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # Use non-GUI backend for testing
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.evaluation import (  # noqa: E402
    ClassificationEvaluator,
    evaluate_model,
    compare_models,
    calculate_class_metrics,
    find_optimal_threshold,
    print_evaluation_summary,
)


@pytest.fixture
def binary_classification_data():
    """Generate binary classification data."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def multiclass_classification_data():
    """Generate multiclass classification data."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42,
    )
    return X, y


@pytest.fixture
def simple_binary_predictions():
    """Create simple binary predictions for testing."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.2, 0.85, 0.6])
    return y_true, y_pred, y_pred_proba


@pytest.fixture
def simple_multiclass_predictions():
    """Create simple multiclass predictions for testing."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1, 0])
    y_pred_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.9, 0.05, 0.05],
            [0.2, 0.3, 0.5],
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.3, 0.4, 0.3],
            [0.7, 0.2, 0.1],
        ]
    )
    return y_true, y_pred, y_pred_proba


@pytest.fixture
def trained_models(binary_classification_data):
    """Train simple models for testing."""
    X, y = binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)

    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    return {
        "models": {"RandomForest": rf_model, "LogisticRegression": lr_model},
        "X_test": X_test,
        "y_test": y_test,
    }


class TestClassificationEvaluator:
    """Tests for ClassificationEvaluator class."""

    def test_init(self):
        """Test evaluator initialization."""
        evaluator = ClassificationEvaluator(model_name="TestModel")
        assert evaluator.model_name == "TestModel"
        assert evaluator.metrics_history == []

    def test_evaluate_binary_without_proba(self, simple_binary_predictions):
        """Test evaluation without probabilities."""
        y_true, y_pred, _ = simple_binary_predictions
        evaluator = ClassificationEvaluator()

        metrics = evaluator.evaluate(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "matthews_corrcoef" in metrics
        assert "cohen_kappa" in metrics

        # Check metric values are in valid range
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    def test_evaluate_binary_with_proba(self, simple_binary_predictions):
        """Test evaluation with probabilities."""
        y_true, y_pred, y_pred_proba = simple_binary_predictions
        evaluator = ClassificationEvaluator()

        metrics = evaluator.evaluate(y_true, y_pred, y_pred_proba)

        assert "roc_auc" in metrics
        assert "average_precision" in metrics
        assert "log_loss" in metrics

        # Check probability-based metrics
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["average_precision"] <= 1
        assert metrics["log_loss"] >= 0

    def test_evaluate_multiclass(self, simple_multiclass_predictions):
        """Test evaluation for multiclass classification."""
        y_true, y_pred, y_pred_proba = simple_multiclass_predictions
        evaluator = ClassificationEvaluator()

        metrics = evaluator.evaluate(y_true, y_pred, y_pred_proba, average="weighted")

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "log_loss" in metrics

    def test_metrics_history(self, simple_binary_predictions):
        """Test that metrics history is stored."""
        y_true, y_pred, y_pred_proba = simple_binary_predictions
        evaluator = ClassificationEvaluator()

        evaluator.evaluate(y_true, y_pred, y_pred_proba)
        evaluator.evaluate(y_true, y_pred, y_pred_proba)

        assert len(evaluator.metrics_history) == 2

    def test_get_classification_report(self, simple_binary_predictions):
        """Test classification report generation."""
        y_true, y_pred, _ = simple_binary_predictions
        evaluator = ClassificationEvaluator()

        report = evaluator.get_classification_report(y_true, y_pred)

        assert isinstance(report, str)
        assert "precision" in report
        assert "recall" in report
        assert "f1-score" in report

    def test_get_classification_report_with_names(self, simple_binary_predictions):
        """Test classification report with class names."""
        y_true, y_pred, _ = simple_binary_predictions
        evaluator = ClassificationEvaluator()

        report = evaluator.get_classification_report(
            y_true, y_pred, target_names=["Negative", "Positive"]
        )

        assert "Negative" in report
        assert "Positive" in report

    def test_compute_confusion_matrix(self, simple_binary_predictions):
        """Test confusion matrix computation."""
        y_true, y_pred, _ = simple_binary_predictions
        evaluator = ClassificationEvaluator()

        cm = evaluator.compute_confusion_matrix(y_true, y_pred)

        assert isinstance(cm, np.ndarray)
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)

    def test_compute_confusion_matrix_normalized(self, simple_binary_predictions):
        """Test normalized confusion matrix."""
        y_true, y_pred, _ = simple_binary_predictions
        evaluator = ClassificationEvaluator()

        cm = evaluator.compute_confusion_matrix(y_true, y_pred, normalize="true")

        # Each row should sum to 1 (normalized by true labels)
        assert np.allclose(cm.sum(axis=1), 1.0)

    def test_plot_confusion_matrix(self, simple_binary_predictions):
        """Test confusion matrix plotting."""
        y_true, y_pred, _ = simple_binary_predictions
        evaluator = ClassificationEvaluator(model_name="TestModel")

        fig = evaluator.plot_confusion_matrix(y_true, y_pred)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_confusion_matrix_with_names(self, simple_binary_predictions):
        """Test confusion matrix plotting with class names."""
        y_true, y_pred, _ = simple_binary_predictions
        evaluator = ClassificationEvaluator()

        fig = evaluator.plot_confusion_matrix(
            y_true, y_pred, class_names=["Negative", "Positive"]
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_confusion_matrix_normalized(self, simple_binary_predictions):
        """Test normalized confusion matrix plotting."""
        y_true, y_pred, _ = simple_binary_predictions
        evaluator = ClassificationEvaluator()

        fig = evaluator.plot_confusion_matrix(y_true, y_pred, normalize=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_roc_curve(self, simple_binary_predictions):
        """Test ROC curve plotting."""
        y_true, _, y_pred_proba = simple_binary_predictions
        evaluator = ClassificationEvaluator()

        fig = evaluator.plot_roc_curve(y_true, y_pred_proba)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_roc_curve_2d_proba(self, simple_binary_predictions):
        """Test ROC curve with 2D probability array."""
        y_true, _, y_pred_proba = simple_binary_predictions
        evaluator = ClassificationEvaluator()

        # Convert to 2D array
        y_pred_proba_2d = np.column_stack([1 - y_pred_proba, y_pred_proba])

        fig = evaluator.plot_roc_curve(y_true, y_pred_proba_2d)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_precision_recall_curve(self, simple_binary_predictions):
        """Test precision-recall curve plotting."""
        y_true, _, y_pred_proba = simple_binary_predictions
        evaluator = ClassificationEvaluator()

        fig = evaluator.plot_precision_recall_curve(y_true, y_pred_proba)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_metrics_comparison(self):
        """Test metrics comparison plotting."""
        evaluator = ClassificationEvaluator()

        metrics_dict = {
            "Model1": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
            },
            "Model2": {
                "accuracy": 0.90,
                "precision": 0.88,
                "recall": 0.92,
                "f1_score": 0.90,
            },
            "Model3": {
                "accuracy": 0.78,
                "precision": 0.75,
                "recall": 0.81,
                "f1_score": 0.78,
            },
        }

        fig = evaluator.plot_metrics_comparison(metrics_dict)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_evaluate_model_basic(self, trained_models):
        """Test basic model evaluation."""
        model = trained_models["models"]["RandomForest"]
        X_test = trained_models["X_test"]
        y_test = trained_models["y_test"]

        metrics = evaluate_model(model, X_test, y_test)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics

    def test_evaluate_model_with_name(self, trained_models):
        """Test model evaluation with custom name."""
        model = trained_models["models"]["RandomForest"]
        X_test = trained_models["X_test"]
        y_test = trained_models["y_test"]

        metrics = evaluate_model(model, X_test, y_test, model_name="Custom RF")

        assert isinstance(metrics, dict)

    def test_evaluate_model_different_averages(self, multiclass_classification_data):
        """Test model evaluation with different averaging methods."""
        X, y = multiclass_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        metrics_weighted = evaluate_model(model, X_test, y_test, average="weighted")
        metrics_macro = evaluate_model(model, X_test, y_test, average="macro")
        metrics_micro = evaluate_model(model, X_test, y_test, average="micro")

        # All should have the same keys
        assert set(metrics_weighted.keys()) == set(metrics_macro.keys())
        assert set(metrics_weighted.keys()) == set(metrics_micro.keys())


class TestCompareModels:
    """Tests for compare_models function."""

    def test_compare_models_basic(self, trained_models):
        """Test basic model comparison."""
        models = trained_models["models"]
        X_test = trained_models["X_test"]
        y_test = trained_models["y_test"]

        comparison_df = compare_models(models, X_test, y_test)

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == len(models)
        assert "accuracy" in comparison_df.columns
        assert "f1_score" in comparison_df.columns

    def test_compare_models_sorted(self, trained_models):
        """Test that comparison results are sorted."""
        models = trained_models["models"]
        X_test = trained_models["X_test"]
        y_test = trained_models["y_test"]

        comparison_df = compare_models(models, X_test, y_test)

        # Check that f1_scores are in descending order
        f1_scores = comparison_df["f1_score"].values
        assert all(f1_scores[i] >= f1_scores[i + 1] for i in range(len(f1_scores) - 1))

    def test_compare_models_single_model(self, trained_models):
        """Test comparison with single model."""
        model = {"OnlyModel": trained_models["models"]["RandomForest"]}
        X_test = trained_models["X_test"]
        y_test = trained_models["y_test"]

        comparison_df = compare_models(model, X_test, y_test)

        assert len(comparison_df) == 1
        assert "OnlyModel" in comparison_df.index


class TestCalculateClassMetrics:
    """Tests for calculate_class_metrics function."""

    def test_calculate_class_metrics_basic(self, simple_multiclass_predictions):
        """Test basic per-class metrics calculation."""
        y_true, y_pred, _ = simple_multiclass_predictions

        metrics_df = calculate_class_metrics(y_true, y_pred)

        assert isinstance(metrics_df, pd.DataFrame)
        assert "precision" in metrics_df.columns
        assert "recall" in metrics_df.columns
        assert "f1_score" in metrics_df.columns
        assert "support" in metrics_df.columns
        assert len(metrics_df) == len(np.unique(y_true))

    def test_calculate_class_metrics_with_labels(self, simple_multiclass_predictions):
        """Test per-class metrics with class labels."""
        y_true, y_pred, _ = simple_multiclass_predictions

        metrics_df = calculate_class_metrics(
            y_true, y_pred, class_labels=["Class A", "Class B", "Class C"]
        )

        assert "class_name" in metrics_df.columns
        assert "Class A" in metrics_df["class_name"].values
        assert "Class B" in metrics_df["class_name"].values
        assert "Class C" in metrics_df["class_name"].values

    def test_calculate_class_metrics_binary(self, simple_binary_predictions):
        """Test per-class metrics for binary classification."""
        y_true, y_pred, _ = simple_binary_predictions

        metrics_df = calculate_class_metrics(y_true, y_pred)

        assert len(metrics_df) == 2
        assert all(metrics_df["precision"] >= 0) and all(metrics_df["precision"] <= 1)


class TestFindOptimalThreshold:
    """Tests for find_optimal_threshold function."""

    def test_find_optimal_threshold_f1(self, simple_binary_predictions):
        """Test finding optimal threshold for F1 score."""
        y_true, _, y_pred_proba = simple_binary_predictions

        threshold, score = find_optimal_threshold(y_true, y_pred_proba, metric="f1")

        assert 0 <= threshold <= 1
        assert 0 <= score <= 1

    def test_find_optimal_threshold_precision(self, simple_binary_predictions):
        """Test finding optimal threshold for precision."""
        y_true, _, y_pred_proba = simple_binary_predictions

        threshold, score = find_optimal_threshold(
            y_true, y_pred_proba, metric="precision"
        )

        assert 0 <= threshold <= 1
        assert 0 <= score <= 1

    def test_find_optimal_threshold_recall(self, simple_binary_predictions):
        """Test finding optimal threshold for recall."""
        y_true, _, y_pred_proba = simple_binary_predictions

        threshold, score = find_optimal_threshold(y_true, y_pred_proba, metric="recall")

        assert 0 <= threshold <= 1
        assert 0 <= score <= 1

    def test_find_optimal_threshold_accuracy(self, simple_binary_predictions):
        """Test finding optimal threshold for accuracy."""
        y_true, _, y_pred_proba = simple_binary_predictions

        threshold, score = find_optimal_threshold(
            y_true, y_pred_proba, metric="accuracy"
        )

        assert 0 <= threshold <= 1
        assert 0 <= score <= 1

    def test_find_optimal_threshold_invalid_metric(self, simple_binary_predictions):
        """Test that invalid metric raises error."""
        y_true, _, y_pred_proba = simple_binary_predictions

        with pytest.raises(ValueError, match="Unknown metric"):
            find_optimal_threshold(y_true, y_pred_proba, metric="invalid")

    def test_find_optimal_threshold_2d_proba(self, simple_binary_predictions):
        """Test threshold finding with 2D probability array."""
        y_true, _, y_pred_proba = simple_binary_predictions

        # Convert to 2D array
        y_pred_proba_2d = np.column_stack([1 - y_pred_proba, y_pred_proba])

        threshold, score = find_optimal_threshold(y_true, y_pred_proba_2d, metric="f1")

        assert 0 <= threshold <= 1
        assert 0 <= score <= 1


class TestPrintEvaluationSummary:
    """Tests for print_evaluation_summary function."""

    def test_print_evaluation_summary_basic(self, simple_binary_predictions, capsys):
        """Test basic evaluation summary printing."""
        y_true, y_pred, y_pred_proba = simple_binary_predictions

        print_evaluation_summary(y_true, y_pred, y_pred_proba)

        captured = capsys.readouterr()
        assert "Evaluation Summary" in captured.out
        assert "accuracy" in captured.out
        assert "Classification Report" in captured.out
        assert "Confusion Matrix" in captured.out

    def test_print_evaluation_summary_with_names(
        self, simple_binary_predictions, capsys
    ):
        """Test evaluation summary with class names."""
        y_true, y_pred, y_pred_proba = simple_binary_predictions

        print_evaluation_summary(
            y_true,
            y_pred,
            y_pred_proba,
            model_name="Test Model",
            class_names=["Negative", "Positive"],
        )

        captured = capsys.readouterr()
        assert "Test Model" in captured.out
        assert "Negative" in captured.out
        assert "Positive" in captured.out

    def test_print_evaluation_summary_without_proba(
        self, simple_binary_predictions, capsys
    ):
        """Test evaluation summary without probabilities."""
        y_true, y_pred, _ = simple_binary_predictions

        print_evaluation_summary(y_true, y_pred)

        captured = capsys.readouterr()
        assert "Evaluation Summary" in captured.out
        assert "accuracy" in captured.out


class TestIntegration:
    """Integration tests for complete evaluation workflows."""

    def test_full_evaluation_workflow(self, binary_classification_data):
        """Test complete evaluation workflow."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        evaluator = ClassificationEvaluator(model_name="Random Forest")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Get metrics
        metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba)

        # Generate visualizations
        fig_cm = evaluator.plot_confusion_matrix(y_test, y_pred)
        fig_roc = evaluator.plot_roc_curve(y_test, y_pred_proba[:, 1])
        fig_pr = evaluator.plot_precision_recall_curve(y_test, y_pred_proba[:, 1])

        # Assertions
        assert metrics["accuracy"] > 0.5
        assert isinstance(fig_cm, plt.Figure)
        assert isinstance(fig_roc, plt.Figure)
        assert isinstance(fig_pr, plt.Figure)

        # Cleanup
        plt.close("all")

    def test_multiclass_evaluation_workflow(self, multiclass_classification_data):
        """Test evaluation workflow for multiclass classification."""
        X, y = multiclass_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, average="weighted")

        # Get per-class metrics
        y_pred = model.predict(X_test)
        class_metrics = calculate_class_metrics(
            y_test, y_pred, class_labels=["Class 0", "Class 1", "Class 2"]
        )

        # Assertions
        assert metrics["accuracy"] > 0.3  # Better than random
        assert len(class_metrics) == 3
        assert "Class 0" in class_metrics["class_name"].values

    def test_model_comparison_workflow(self, binary_classification_data):
        """Test complete model comparison workflow."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train multiple models
        rf_model = RandomForestClassifier(n_estimators=30, random_state=42)
        lr_model = LogisticRegression(random_state=42, max_iter=1000)

        rf_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)

        models = {"Random Forest": rf_model, "Logistic Regression": lr_model}

        # Compare models
        comparison_df = compare_models(models, X_test, y_test)

        # Plot comparison
        evaluator = ClassificationEvaluator()
        metrics_dict = comparison_df.to_dict("index")
        fig = evaluator.plot_metrics_comparison(metrics_dict)

        # Assertions
        assert len(comparison_df) == 2
        assert "Random Forest" in comparison_df.index
        assert "Logistic Regression" in comparison_df.index
        assert isinstance(fig, plt.Figure)

        # Cleanup
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
