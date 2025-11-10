"""
Classification models module for machine learning projects.

This module provides wrapper classes and utilities for training and using
various classification models with a consistent interface.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.base import BaseEstimator


class BinaryClassifier:
    """Wrapper for binary classification models."""

    def __init__(
        self, model_type: str = "random_forest", random_state: int = 42, **kwargs
    ):
        """
        Initialize a binary classifier.

        Args:
            model_type: Type of classifier ('random_forest', 'logistic_regression',
                       'svm', 'gradient_boosting', 'decision_tree', 'knn', 'naive_bayes')
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters for the specific model
        """
        self.model_type = model_type
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = self._create_model()
        self.is_fitted = False

    def _create_model(self) -> BaseEstimator:
        """Create the underlying model based on model_type."""
        models = {
            "random_forest": RandomForestClassifier,
            "logistic_regression": LogisticRegression,
            "svm": SVC,
            "gradient_boosting": GradientBoostingClassifier,
            "decision_tree": DecisionTreeClassifier,
            "knn": KNeighborsClassifier,
            "naive_bayes": GaussianNB,
            "adaboost": AdaBoostClassifier,
            "extra_trees": ExtraTreesClassifier,
        }

        if self.model_type not in models:
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                f"Available types: {list(models.keys())}"
            )

        model_class = models[self.model_type]

        # Set default parameters
        default_params = self.kwargs.copy()

        # Add random_state for models that support it
        if self.model_type in [
            "random_forest",
            "logistic_regression",
            "gradient_boosting",
            "decision_tree",
            "adaboost",
            "extra_trees",
        ]:
            default_params.setdefault("random_state", self.random_state)

        # Add probability=True for SVM
        if self.model_type == "svm":
            default_params.setdefault("probability", True)
            default_params.setdefault("random_state", self.random_state)

        # Set max_iter for logistic regression
        if self.model_type == "logistic_regression":
            default_params.setdefault("max_iter", 1000)

        return model_class(**default_params)

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> "BinaryClassifier":
        """
        Fit the classifier.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features to predict

        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict

        Returns:
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()

    def set_params(self, **params) -> "BinaryClassifier":
        """Set model parameters."""
        self.model.set_params(**params)
        return self


class MulticlassClassifier:
    """Wrapper for multiclass classification models."""

    def __init__(
        self, model_type: str = "random_forest", random_state: int = 42, **kwargs
    ):
        """
        Initialize a multiclass classifier.

        Args:
            model_type: Type of classifier
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters for the specific model
        """
        # Reuse BinaryClassifier logic as most models support multiclass natively
        self.classifier = BinaryClassifier(model_type, random_state, **kwargs)
        self.model = self.classifier.model
        self.is_fitted = False

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> "MulticlassClassifier":
        """Fit the classifier."""
        self.classifier.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class labels."""
        return self.classifier.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        return self.classifier.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.classifier.get_params()

    def set_params(self, **params) -> "MulticlassClassifier":
        """Set model parameters."""
        self.classifier.set_params(**params)
        return self


class ModelTrainer:
    """Train and manage multiple classification models."""

    def __init__(self, random_state: int = 42):
        """
        Initialize ModelTrainer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.trained_models: Dict[str, Any] = {}

    def train_multiple_models(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Train multiple models with different configurations.

        Args:
            X_train: Training features
            y_train: Training labels
            model_configs: Dictionary mapping model names to configurations

        Returns:
            Dictionary of trained models
        """
        if model_configs is None:
            # Default configurations
            model_configs = {
                "Random Forest": {"model_type": "random_forest", "n_estimators": 100},
                "Logistic Regression": {"model_type": "logistic_regression"},
                "SVM": {"model_type": "svm"},
                "Gradient Boosting": {
                    "model_type": "gradient_boosting",
                    "n_estimators": 100,
                },
            }

        for name, config in model_configs.items():
            print(f"Training {name}...")
            classifier = BinaryClassifier(random_state=self.random_state, **config)
            classifier.fit(X_train, y_train)
            self.trained_models[name] = classifier.model

        return self.trained_models

    def get_model(self, name: str) -> Any:
        """
        Get a trained model by name.

        Args:
            name: Name of the model

        Returns:
            Trained model

        Raises:
            KeyError: If model name not found
        """
        if name not in self.trained_models:
            raise KeyError(
                f"Model '{name}' not found. Available models: {list(self.trained_models.keys())}"
            )
        return self.trained_models[name]


class HyperparameterTuner:
    """Tune hyperparameters for classification models."""

    def __init__(
        self,
        model: BaseEstimator,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = "accuracy",
        random_state: int = 42,
    ):
        """
        Initialize HyperparameterTuner.

        Args:
            model: Base model to tune
            param_grid: Parameter grid for tuning
            cv: Number of cross-validation folds
            scoring: Scoring metric
            random_state: Random seed
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def grid_search(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_jobs: int = -1,
    ) -> BaseEstimator:
        """
        Perform grid search for hyperparameter tuning.

        Args:
            X: Training features
            y: Training labels
            n_jobs: Number of parallel jobs

        Returns:
            Best model found
        """
        grid_search = GridSearchCV(
            self.model,
            self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=n_jobs,
            verbose=1,
        )

        grid_search.fit(X, y)

        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_

        return self.best_model

    def random_search(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_iter: int = 10,
        n_jobs: int = -1,
    ) -> BaseEstimator:
        """
        Perform randomized search for hyperparameter tuning.

        Args:
            X: Training features
            y: Training labels
            n_iter: Number of parameter settings sampled
            n_jobs: Number of parallel jobs

        Returns:
            Best model found
        """
        random_search = RandomizedSearchCV(
            self.model,
            self.param_grid,
            n_iter=n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=n_jobs,
            random_state=self.random_state,
            verbose=1,
        )

        random_search.fit(X, y)

        self.best_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_

        return self.best_model

    def get_best_params(self) -> Dict[str, Any]:
        """Get the best parameters found."""
        if self.best_params is None:
            raise ValueError("No tuning has been performed yet")
        return self.best_params

    def get_best_score(self) -> float:
        """Get the best score achieved."""
        if self.best_score is None:
            raise ValueError("No tuning has been performed yet")
        return self.best_score


def cross_validate_model(
    model: BaseEstimator,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    cv: int = 5,
    scoring: str = "accuracy",
) -> Dict[str, float]:
    """
    Perform cross-validation on a model.

    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        cv: Number of folds
        scoring: Scoring metric

    Returns:
        Dictionary with mean and std of scores
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    return {"mean_score": scores.mean(), "std_score": scores.std(), "scores": scores}


def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """
    Save a trained model to disk.

    Args:
        model: Trained model to save
        filepath: Path to save the model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {filepath}")


def load_model(filepath: Union[str, Path]) -> Any:
    """
    Load a trained model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded model

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    print(f"Model loaded from {filepath}")
    return model


def get_feature_importance(
    model: Any, feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get feature importance from a trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Names of features

    Returns:
        DataFrame with feature importance sorted by importance

    Raises:
        AttributeError: If model doesn't have feature_importances_
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not have feature_importances_ attribute")

    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})

    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df = importance_df.reset_index(drop=True)

    return importance_df


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("Generating sample data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Example 1: Binary Classifier
    print("\n" + "=" * 60)
    print("Example 1: Binary Classifier")
    print("=" * 60)

    classifier = BinaryClassifier(model_type="random_forest", n_estimators=100)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)

    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy:.4f}")

    # Example 2: Train Multiple Models
    print("\n" + "=" * 60)
    print("Example 2: Train Multiple Models")
    print("=" * 60)

    trainer = ModelTrainer(random_state=42)
    models = trainer.train_multiple_models(X_train, y_train)

    print(f"\nTrained {len(models)} models:")
    for name in models.keys():
        print(f"  - {name}")

    # Example 3: Hyperparameter Tuning
    print("\n" + "=" * 60)
    print("Example 3: Hyperparameter Tuning")
    print("=" * 60)

    base_model = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5],
    }

    tuner = HyperparameterTuner(base_model, param_grid, cv=3, scoring="accuracy")

    print("\nPerforming grid search...")
    best_model = tuner.grid_search(X_train, y_train, n_jobs=1)

    print(f"\nBest parameters: {tuner.get_best_params()}")
    print(f"Best CV score: {tuner.get_best_score():.4f}")

    # Example 4: Cross-validation
    print("\n" + "=" * 60)
    print("Example 4: Cross-Validation")
    print("=" * 60)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_results = cross_validate_model(model, X_train, y_train, cv=5)

    print("\nCross-validation results:")
    print(f"  Mean score: {cv_results['mean_score']:.4f}")
    print(f"  Std score: {cv_results['std_score']:.4f}")

    # Example 5: Feature Importance
    print("\n" + "=" * 60)
    print("Example 5: Feature Importance")
    print("=" * 60)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    importance_df = get_feature_importance(model)
    print("\nTop 5 most important features:")
    print(importance_df.head())

    # Example 6: Save and Load Model
    print("\n" + "=" * 60)
    print("Example 6: Save and Load Model")
    print("=" * 60)

    save_model(model, "models/example_model.pkl")
    loaded_model = load_model("models/example_model.pkl")

    # Verify loaded model works
    loaded_pred = loaded_model.predict(X_test[:5])
    print(f"\nPredictions from loaded model: {loaded_pred}")
