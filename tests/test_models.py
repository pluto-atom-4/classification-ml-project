"""
Tests for models module.

Run with: pytest tests/test_models.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tempfile  # noqa: E402
import shutil  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from src.models import (  # noqa: E402
    BinaryClassifier,
    MulticlassClassifier,
    ModelTrainer,
    HyperparameterTuner,
    cross_validate_model,
    save_model,
    load_model,
    get_feature_importance,
)


@pytest.fixture
def binary_data():
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def multiclass_data():
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def temp_dir():
    """Create temporary directory for model saving."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


class TestBinaryClassifier:
    """Tests for BinaryClassifier class."""

    def test_init_random_forest(self):
        """Test initialization with random forest."""
        classifier = BinaryClassifier(model_type="random_forest")
        assert classifier.model_type == "random_forest"
        assert classifier.random_state == 42
        assert not classifier.is_fitted

    def test_init_logistic_regression(self):
        """Test initialization with logistic regression."""
        classifier = BinaryClassifier(model_type="logistic_regression")
        assert classifier.model_type == "logistic_regression"
        assert hasattr(classifier.model, "fit")

    def test_init_svm(self):
        """Test initialization with SVM."""
        classifier = BinaryClassifier(model_type="svm")
        assert classifier.model_type == "svm"
        # Check that probability=True is set by default
        assert classifier.model.probability is True

    def test_init_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError, match="Unknown model type"):
            BinaryClassifier(model_type="invalid_model")

    def test_init_with_kwargs(self):
        """Test initialization with additional parameters."""
        classifier = BinaryClassifier(
            model_type="random_forest", n_estimators=50, max_depth=5
        )
        assert classifier.model.n_estimators == 50
        assert classifier.model.max_depth == 5

    def test_fit_predict(self, binary_data):
        """Test fitting and predicting."""
        X_train, X_test, y_train, y_test = binary_data
        classifier = BinaryClassifier(model_type="random_forest", n_estimators=10)

        # Fit the model
        result = classifier.fit(X_train, y_train)
        assert result is classifier  # Check method chaining
        assert classifier.is_fitted

        # Make predictions
        y_pred = classifier.predict(X_test)
        assert len(y_pred) == len(y_test)
        assert set(y_pred).issubset({0, 1})

    def test_predict_proba(self, binary_data):
        """Test probability predictions."""
        X_train, X_test, y_train, y_test = binary_data
        classifier = BinaryClassifier(model_type="random_forest", n_estimators=10)
        classifier.fit(X_train, y_train)

        y_pred_proba = classifier.predict_proba(X_test)
        assert y_pred_proba.shape == (len(y_test), 2)
        assert np.allclose(y_pred_proba.sum(axis=1), 1.0)

    def test_predict_before_fit_raises_error(self, binary_data):
        """Test that predicting before fitting raises error."""
        X_train, X_test, y_train, y_test = binary_data
        classifier = BinaryClassifier(model_type="random_forest")

        with pytest.raises(ValueError, match="must be fitted"):
            classifier.predict(X_test)

    def test_predict_proba_before_fit_raises_error(self, binary_data):
        """Test that predict_proba before fitting raises error."""
        X_train, X_test, y_train, y_test = binary_data
        classifier = BinaryClassifier(model_type="random_forest")

        with pytest.raises(ValueError, match="must be fitted"):
            classifier.predict_proba(X_test)

    def test_get_params(self):
        """Test getting model parameters."""
        classifier = BinaryClassifier(
            model_type="random_forest", n_estimators=50, max_depth=10
        )
        params = classifier.get_params()

        assert isinstance(params, dict)
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 10

    def test_set_params(self):
        """Test setting model parameters."""
        classifier = BinaryClassifier(model_type="random_forest", n_estimators=50)
        classifier.set_params(n_estimators=100, max_depth=5)

        assert classifier.model.n_estimators == 100
        assert classifier.model.max_depth == 5

    def test_all_model_types(self, binary_data):
        """Test all supported model types."""
        X_train, X_test, y_train, y_test = binary_data

        model_types = [
            "random_forest",
            "logistic_regression",
            "svm",
            "gradient_boosting",
            "decision_tree",
            "knn",
            "naive_bayes",
            "adaboost",
            "extra_trees",
        ]

        for model_type in model_types:
            classifier = BinaryClassifier(model_type=model_type)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            assert len(y_pred) == len(y_test)
            assert classifier.is_fitted


class TestMulticlassClassifier:
    """Tests for MulticlassClassifier class."""

    def test_init(self):
        """Test initialization."""
        classifier = MulticlassClassifier(model_type="random_forest")
        assert not classifier.is_fitted
        assert hasattr(classifier, "model")

    def test_fit_predict(self, multiclass_data):
        """Test fitting and predicting."""
        X_train, X_test, y_train, y_test = multiclass_data
        classifier = MulticlassClassifier(model_type="random_forest", n_estimators=10)

        classifier.fit(X_train, y_train)
        assert classifier.is_fitted

        y_pred = classifier.predict(X_test)
        assert len(y_pred) == len(y_test)
        assert set(y_pred).issubset({0, 1, 2})

    def test_predict_proba(self, multiclass_data):
        """Test probability predictions for multiclass."""
        X_train, X_test, y_train, y_test = multiclass_data
        classifier = MulticlassClassifier(model_type="random_forest", n_estimators=10)
        classifier.fit(X_train, y_train)

        y_pred_proba = classifier.predict_proba(X_test)
        assert y_pred_proba.shape == (len(y_test), 3)
        assert np.allclose(y_pred_proba.sum(axis=1), 1.0)

    def test_get_set_params(self):
        """Test getting and setting parameters."""
        classifier = MulticlassClassifier(model_type="random_forest", n_estimators=50)
        params = classifier.get_params()
        assert params["n_estimators"] == 50

        classifier.set_params(n_estimators=100)
        assert classifier.get_params()["n_estimators"] == 100


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    def test_init(self):
        """Test initialization."""
        trainer = ModelTrainer(random_state=42)
        assert trainer.random_state == 42
        assert trainer.trained_models == {}

    def test_train_multiple_models_default(self, binary_data):
        """Test training multiple models with default configurations."""
        X_train, X_test, y_train, y_test = binary_data
        trainer = ModelTrainer()

        models = trainer.train_multiple_models(X_train, y_train)

        assert isinstance(models, dict)
        assert len(models) > 0
        assert "Random Forest" in models
        assert "Logistic Regression" in models

    def test_train_multiple_models_custom(self, binary_data):
        """Test training with custom configurations."""
        X_train, X_test, y_train, y_test = binary_data
        trainer = ModelTrainer()

        custom_configs = {
            "RF_Small": {"model_type": "random_forest", "n_estimators": 10},
            "LR": {"model_type": "logistic_regression"},
        }

        models = trainer.train_multiple_models(X_train, y_train, custom_configs)

        assert len(models) == 2
        assert "RF_Small" in models
        assert "LR" in models

    def test_get_model(self, binary_data):
        """Test getting a trained model."""
        X_train, X_test, y_train, y_test = binary_data
        trainer = ModelTrainer()

        trainer.train_multiple_models(X_train, y_train)
        model = trainer.get_model("Random Forest")

        assert model is not None
        assert hasattr(model, "predict")

    def test_get_model_not_found(self):
        """Test that getting non-existent model raises error."""
        trainer = ModelTrainer()

        with pytest.raises(KeyError, match="not found"):
            trainer.get_model("NonExistent")


class TestHyperparameterTuner:
    """Tests for HyperparameterTuner class."""

    def test_init(self):
        """Test initialization."""
        model = RandomForestClassifier()
        param_grid = {"n_estimators": [10, 50], "max_depth": [3, 5]}

        tuner = HyperparameterTuner(model, param_grid, cv=3)

        assert tuner.model is model
        assert tuner.param_grid == param_grid
        assert tuner.cv == 3
        assert tuner.best_model is None

    def test_grid_search(self, binary_data):
        """Test grid search."""
        X_train, X_test, y_train, y_test = binary_data

        model = RandomForestClassifier(random_state=42)
        param_grid = {"n_estimators": [10, 20], "max_depth": [3, 5]}

        tuner = HyperparameterTuner(model, param_grid, cv=3)
        best_model = tuner.grid_search(X_train, y_train, n_jobs=1)

        assert best_model is not None
        assert tuner.best_params is not None
        assert tuner.best_score is not None
        assert "n_estimators" in tuner.best_params
        assert "max_depth" in tuner.best_params

    def test_random_search(self, binary_data):
        """Test randomized search."""
        X_train, X_test, y_train, y_test = binary_data

        model = RandomForestClassifier(random_state=42)
        param_grid = {"n_estimators": [10, 20, 30], "max_depth": [3, 5, 7]}

        tuner = HyperparameterTuner(model, param_grid, cv=3)
        best_model = tuner.random_search(X_train, y_train, n_iter=3, n_jobs=1)

        assert best_model is not None
        assert tuner.best_params is not None
        assert tuner.best_score is not None

    def test_get_best_params_before_tuning(self):
        """Test that getting best params before tuning raises error."""
        model = RandomForestClassifier()
        param_grid = {"n_estimators": [10, 50]}
        tuner = HyperparameterTuner(model, param_grid)

        with pytest.raises(ValueError, match="No tuning has been performed"):
            tuner.get_best_params()

    def test_get_best_score_before_tuning(self):
        """Test that getting best score before tuning raises error."""
        model = RandomForestClassifier()
        param_grid = {"n_estimators": [10, 50]}
        tuner = HyperparameterTuner(model, param_grid)

        with pytest.raises(ValueError, match="No tuning has been performed"):
            tuner.get_best_score()


class TestCrossValidateModel:
    """Tests for cross_validate_model function."""

    def test_cross_validate_basic(self, binary_data):
        """Test basic cross-validation."""
        X_train, X_test, y_train, y_test = binary_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        results = cross_validate_model(model, X_train, y_train, cv=3)

        assert "mean_score" in results
        assert "std_score" in results
        assert "scores" in results
        assert 0 <= results["mean_score"] <= 1
        assert len(results["scores"]) == 3

    def test_cross_validate_different_scoring(self, binary_data):
        """Test cross-validation with different scoring metric."""
        X_train, X_test, y_train, y_test = binary_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        results = cross_validate_model(model, X_train, y_train, cv=3, scoring="f1")

        assert "mean_score" in results
        assert 0 <= results["mean_score"] <= 1


class TestSaveLoadModel:
    """Tests for save_model and load_model functions."""

    def test_save_load_model(self, binary_data, temp_dir):
        """Test saving and loading a model."""
        X_train, X_test, y_train, y_test = binary_data

        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Get predictions before saving
        pred_before = model.predict(X_test)

        # Save model
        model_path = Path(temp_dir) / "test_model.pkl"
        save_model(model, model_path)

        assert model_path.exists()

        # Load model
        loaded_model = load_model(model_path)

        # Get predictions after loading
        pred_after = loaded_model.predict(X_test)

        # Predictions should be identical
        np.testing.assert_array_equal(pred_before, pred_after)

    def test_save_model_creates_directory(self, binary_data, temp_dir):
        """Test that save_model creates directories if needed."""
        X_train, X_test, y_train, y_test = binary_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Save to nested directory that doesn't exist
        model_path = Path(temp_dir) / "subdir" / "model.pkl"
        save_model(model, model_path)

        assert model_path.exists()
        assert model_path.parent.exists()

    def test_load_model_not_found(self, temp_dir):
        """Test that loading non-existent model raises error."""
        model_path = Path(temp_dir) / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError):
            load_model(model_path)


class TestGetFeatureImportance:
    """Tests for get_feature_importance function."""

    def test_feature_importance_basic(self, binary_data):
        """Test getting feature importance."""
        X_train, X_test, y_train, y_test = binary_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        importance_df = get_feature_importance(model)

        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == X_train.shape[1]

        # Check that it's sorted by importance
        importances = importance_df["importance"].values
        assert all(
            importances[i] >= importances[i + 1] for i in range(len(importances) - 1)
        )

    def test_feature_importance_with_names(self, binary_data):
        """Test feature importance with custom feature names."""
        X_train, X_test, y_train, y_test = binary_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        importance_df = get_feature_importance(model, feature_names)

        assert all(name in importance_df["feature"].values for name in feature_names)

    def test_feature_importance_no_attribute(self, binary_data):
        """Test that models without feature_importances_ raise error."""
        X_train, X_test, y_train, y_test = binary_data

        # Logistic regression doesn't have feature_importances_
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        with pytest.raises(AttributeError, match="does not have feature_importances_"):
            get_feature_importance(model)


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_binary_workflow(self, binary_data):
        """Test complete binary classification workflow."""
        X_train, X_test, y_train, y_test = binary_data

        # Train classifier
        classifier = BinaryClassifier(
            model_type="random_forest", n_estimators=50, random_state=42
        )
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)

        # Evaluate
        accuracy = (y_pred == y_test).mean()

        assert accuracy > 0.5  # Better than random
        assert len(y_pred_proba) == len(y_test)

    def test_complete_multiclass_workflow(self, multiclass_data):
        """Test complete multiclass classification workflow."""
        X_train, X_test, y_train, y_test = multiclass_data

        # Train classifier
        classifier = MulticlassClassifier(
            model_type="random_forest", n_estimators=50, random_state=42
        )
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = classifier.predict(X_test)

        # Evaluate
        accuracy = (y_pred == y_test).mean()

        assert accuracy > 0.3  # Better than random for 3 classes

    def test_trainer_and_tuner_workflow(self, binary_data):
        """Test workflow with ModelTrainer and HyperparameterTuner."""
        X_train, X_test, y_train, y_test = binary_data

        # Train multiple models
        trainer = ModelTrainer(random_state=42)
        custom_configs = {
            "RF": {"model_type": "random_forest", "n_estimators": 20},
            "LR": {"model_type": "logistic_regression"},
        }
        models = trainer.train_multiple_models(X_train, y_train, custom_configs)

        # Tune one of the models
        base_model = RandomForestClassifier(random_state=42)
        param_grid = {"n_estimators": [10, 20], "max_depth": [3, 5]}

        tuner = HyperparameterTuner(base_model, param_grid, cv=3)
        best_model = tuner.grid_search(X_train, y_train, n_jobs=1)

        # Verify we have trained models
        assert len(models) == 2
        assert best_model is not None
        assert tuner.best_score > 0

    def test_save_load_workflow(self, binary_data, temp_dir):
        """Test complete save and load workflow."""
        X_train, X_test, y_train, y_test = binary_data

        # Train model
        classifier = BinaryClassifier(
            model_type="random_forest", n_estimators=30, random_state=42
        )
        classifier.fit(X_train, y_train)

        # Get feature importance
        importance_df = get_feature_importance(classifier.model)

        # Save model
        model_path = Path(temp_dir) / "final_model.pkl"
        save_model(classifier.model, model_path)

        # Load and use
        loaded_model = load_model(model_path)
        y_pred = loaded_model.predict(X_test)

        assert len(y_pred) == len(y_test)
        assert len(importance_df) == X_train.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
