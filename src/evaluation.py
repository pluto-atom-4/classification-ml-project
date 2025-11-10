"""
Model evaluation module for classification machine learning projects.

This module provides functions for evaluating classification models using
various metrics and visualization techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,
)


class ClassificationEvaluator:
    """Evaluate classification model performance."""

    def __init__(self, model_name: str = "Model"):
        """
        Initialize ClassificationEvaluator.

        Args:
            model_name: Name of the model being evaluated
        """
        self.model_name = model_name
        self.metrics_history = []

    def evaluate(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        y_pred_proba: Optional[Union[np.ndarray, pd.Series]] = None,
        average: str = "weighted",
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            average: Averaging method for multiclass ('weighted', 'macro', 'micro')

        Returns:
            Dictionary containing evaluation metrics
        """
        # Convert to numpy arrays if needed
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average=average, zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        }

        # Add metrics that require predicted probabilities
        if y_pred_proba is not None:
            y_pred_proba = np.asarray(y_pred_proba)

            try:
                # For binary classification or when probabilities are provided
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    if len(y_pred_proba.shape) == 2:
                        y_pred_proba_binary = y_pred_proba[:, 1]
                    else:
                        y_pred_proba_binary = y_pred_proba

                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba_binary)
                    metrics["average_precision"] = average_precision_score(
                        y_true, y_pred_proba_binary
                    )
                else:
                    # Multiclass classification
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_pred_proba, average=average, multi_class="ovr"
                    )

                # Log loss
                metrics["log_loss"] = log_loss(y_true, y_pred_proba)
            except (ValueError, IndexError):
                # Skip probability-based metrics if they can't be computed
                pass

        # Store metrics history
        self.metrics_history.append(metrics.copy())

        return metrics

    def get_classification_report(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        target_names: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes

        Returns:
            Classification report as string
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        return classification_report(y_true, y_pred, target_names=target_names)

    def compute_confusion_matrix(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        normalize: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization mode ('true', 'pred', 'all', or None)

        Returns:
            Confusion matrix array
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        return confusion_matrix(y_true, y_pred, normalize=normalize)

    def plot_confusion_matrix(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        class_names: Optional[List[str]] = None,
        normalize: bool = False,
        figsize: Tuple[int, int] = (8, 6),
        cmap: str = "Blues",
    ) -> plt.Figure:
        """
        Plot confusion matrix as a heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size
            cmap: Colormap for the heatmap

        Returns:
            Matplotlib figure object
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Compute confusion matrix
        norm_mode = "true" if normalize else None
        cm = self.compute_confusion_matrix(y_true, y_pred, normalize=norm_mode)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap=cmap,
            square=True,
            cbar_kws={"label": "Count"},
            ax=ax,
        )

        # Set labels
        if class_names is not None:
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names, rotation=0)

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        title = f"Confusion Matrix - {self.model_name}"
        if normalize:
            title += " (Normalized)"
        ax.set_title(title)

        plt.tight_layout()
        return fig

    def plot_roc_curve(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred_proba: Union[np.ndarray, pd.Series],
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """
        Plot ROC curve for binary classification.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        # Handle 2D probability arrays (take positive class)
        if len(y_pred_proba.shape) == 2:
            y_pred_proba = y_pred_proba[:, 1]

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot ROC curve
        ax.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {auc_score:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")

        # Set labels and title
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {self.model_name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_precision_recall_curve(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred_proba: Union[np.ndarray, pd.Series],
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """
        Plot precision-recall curve for binary classification.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        # Handle 2D probability arrays
        if len(y_pred_proba.shape) == 2:
            y_pred_proba = y_pred_proba[:, 1]

        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot precision-recall curve
        ax.plot(
            recall, precision, linewidth=2, label=f"PR curve (AP = {avg_precision:.3f})"
        )

        # Set labels and title
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve - {self.model_name}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        return fig

    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot comparison of multiple models' metrics.

        Args:
            metrics_dict: Dictionary mapping model names to their metrics
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        # Convert to DataFrame for easy plotting
        df = pd.DataFrame(metrics_dict).T

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot grouped bar chart
        df.plot(kind="bar", ax=ax, rot=45)

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.legend(loc="best", ncol=2)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        return fig


def evaluate_model(
    model: Any,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    model_name: str = "Model",
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Evaluate a trained classification model.

    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        average: Averaging method for multiclass metrics

    Returns:
        Dictionary containing evaluation metrics
    """
    evaluator = ClassificationEvaluator(model_name=model_name)

    # Get predictions
    y_pred = model.predict(X_test)

    # Get probabilities if available
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)

    # Evaluate
    metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba, average=average)

    return metrics


def compare_models(
    models: Dict[str, Any],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    average: str = "weighted",
) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.

    Args:
        models: Dictionary mapping model names to trained models
        X_test: Test features
        y_test: Test labels
        average: Averaging method for multiclass metrics

    Returns:
        DataFrame with metrics for each model
    """
    results = {}

    for name, model in models.items():
        metrics = evaluate_model(
            model, X_test, y_test, model_name=name, average=average
        )
        results[name] = metrics

    # Convert to DataFrame and transpose
    df = pd.DataFrame(results).T

    # Sort by accuracy (or F1 score)
    sort_by = "f1_score" if "f1_score" in df.columns else "accuracy"
    df = df.sort_values(by=sort_by, ascending=False)

    return df


def calculate_class_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    class_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculate per-class metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_labels: Names of classes

    Returns:
        DataFrame with metrics for each class
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Get unique classes
    classes = np.unique(y_true)

    # Calculate metrics for each class
    results = []
    for cls in classes:
        # Create binary labels for this class
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        metrics = {
            "class": cls,
            "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
            "f1_score": f1_score(y_true_binary, y_pred_binary, zero_division=0),
            "support": np.sum(y_true == cls),
        }
        results.append(metrics)

    df = pd.DataFrame(results)

    # Add class labels if provided
    if class_labels is not None:
        df["class_name"] = [
            class_labels[int(c)] if int(c) < len(class_labels) else str(c)
            for c in df["class"]
        ]
        # Reorder columns
        cols = ["class", "class_name", "precision", "recall", "f1_score", "support"]
        df = df[cols]

    return df


def find_optimal_threshold(
    y_true: Union[np.ndarray, pd.Series],
    y_pred_proba: Union[np.ndarray, pd.Series],
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find optimal classification threshold for binary classification.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')

    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Handle 2D probability arrays
    if len(y_pred_proba.shape) == 2:
        y_pred_proba = y_pred_proba[:, 1]

    # Try different thresholds
    thresholds = np.arange(0.0, 1.01, 0.01)
    scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

    # Find best threshold
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx]
    best_score = scores[best_idx]

    return optimal_threshold, best_score


def print_evaluation_summary(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_pred_proba: Optional[Union[np.ndarray, pd.Series]] = None,
    model_name: str = "Model",
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Print a comprehensive evaluation summary.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        model_name: Name of the model
        class_names: Names of classes
    """
    evaluator = ClassificationEvaluator(model_name=model_name)

    print(f"\n{'='*60}")
    print(f"Evaluation Summary: {model_name}")
    print(f"{'='*60}\n")

    # Overall metrics
    metrics = evaluator.evaluate(y_true, y_pred, y_pred_proba)

    print("Overall Metrics:")
    print("-" * 40)
    for metric_name, value in metrics.items():
        print(f"  {metric_name:20s}: {value:.4f}")

    # Classification report
    print(f"\n{'='*60}")
    print("Classification Report:")
    print("-" * 40)
    print(evaluator.get_classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    print(f"\n{'='*60}")
    print("Confusion Matrix:")
    print("-" * 40)
    cm = evaluator.compute_confusion_matrix(y_true, y_pred)
    print(cm)
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Example usage with synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    print("Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models
    print("\nTraining models...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)

    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    # Evaluate single model
    print("\n" + "=" * 60)
    print("Single Model Evaluation")
    print("=" * 60)

    evaluator = ClassificationEvaluator(model_name="Random Forest")
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)

    metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba)

    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Compare multiple models
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)

    models = {"Random Forest": rf_model, "Logistic Regression": lr_model}

    comparison_df = compare_models(models, X_test, y_test)
    print("\n", comparison_df)

    # Print detailed summary
    print_evaluation_summary(
        y_test,
        y_pred,
        y_pred_proba,
        model_name="Random Forest",
        class_names=["Class 0", "Class 1", "Class 2"],
    )
