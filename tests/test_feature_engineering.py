"""
Tests for feature engineering module.

Run with: pytest tests/test_feature_engineering.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import (
    FeatureCreator,
    FeatureSelector,
    DimensionalityReducer,
    create_date_features,
    create_text_features,
)

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_0": np.random.randn(100),
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        }
    )


@pytest.fixture
def classification_data():
    """Generate classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=3,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    return df, y


class TestFeatureCreator:
    """Tests for FeatureCreator class."""

    def test_init(self):
        """Test initialization."""
        creator = FeatureCreator()
        assert creator.created_features == []

    def test_create_polynomial_features(self, sample_df):
        """Test polynomial feature creation."""
        creator = FeatureCreator()
        df_poly = creator.create_polynomial_features(
            sample_df, columns=["feature_0", "feature_1"], degree=2
        )

        # Should have more features than original
        assert df_poly.shape[1] > sample_df.shape[1]
        assert df_poly.shape[0] == sample_df.shape[0]
        assert len(creator.created_features) > 0

    def test_create_polynomial_features_degree_3(self, sample_df):
        """Test polynomial features with degree 3."""
        creator = FeatureCreator()
        df_poly = creator.create_polynomial_features(
            sample_df, columns=["feature_0"], degree=3
        )

        assert df_poly.shape[1] > sample_df.shape[1]

    def test_create_interaction_features_multiply(self, sample_df):
        """Test interaction features with multiplication."""
        creator = FeatureCreator()
        df_interaction = creator.create_interaction_features(
            sample_df, column_pairs=[("feature_0", "feature_1")], operation="multiply"
        )

        assert "feature_0_multiply_feature_1" in df_interaction.columns
        assert df_interaction.shape[0] == sample_df.shape[0]
        # Verify the operation
        expected = sample_df["feature_0"] * sample_df["feature_1"]
        pd.testing.assert_series_equal(
            df_interaction["feature_0_multiply_feature_1"], expected, check_names=False
        )

    def test_create_interaction_features_add(self, sample_df):
        """Test interaction features with addition."""
        creator = FeatureCreator()
        df_interaction = creator.create_interaction_features(
            sample_df, column_pairs=[("feature_0", "feature_1")], operation="add"
        )

        assert "feature_0_add_feature_1" in df_interaction.columns
        expected = sample_df["feature_0"] + sample_df["feature_1"]
        pd.testing.assert_series_equal(
            df_interaction["feature_0_add_feature_1"], expected, check_names=False
        )

    def test_create_interaction_features_divide(self, sample_df):
        """Test interaction features with division."""
        creator = FeatureCreator()
        df_interaction = creator.create_interaction_features(
            sample_df, column_pairs=[("feature_0", "feature_1")], operation="divide"
        )

        assert "feature_0_divide_feature_1" in df_interaction.columns
        # Should not have NaN or inf values
        assert not df_interaction["feature_0_divide_feature_1"].isna().any()
        assert not np.isinf(df_interaction["feature_0_divide_feature_1"]).any()

    def test_create_interaction_invalid_operation(self, sample_df):
        """Test that invalid operation raises error."""
        creator = FeatureCreator()

        with pytest.raises(ValueError, match="Unknown operation"):
            creator.create_interaction_features(
                sample_df,
                column_pairs=[("feature_0", "feature_1")],
                operation="invalid",
            )

    def test_create_binned_features_quantile(self, sample_df):
        """Test binned features with quantile strategy."""
        creator = FeatureCreator()
        df_binned = creator.create_binned_features(
            sample_df, columns=["feature_0"], n_bins=5, strategy="quantile"
        )

        assert "feature_0_binned" in df_binned.columns
        # Should have 5 unique bins (0-4)
        unique_bins = df_binned["feature_0_binned"].dropna().unique()
        assert len(unique_bins) <= 5

    def test_create_binned_features_uniform(self, sample_df):
        """Test binned features with uniform strategy."""
        creator = FeatureCreator()
        df_binned = creator.create_binned_features(
            sample_df, columns=["feature_0"], n_bins=4, strategy="uniform"
        )

        assert "feature_0_binned" in df_binned.columns

    def test_create_binned_invalid_strategy(self, sample_df):
        """Test that invalid binning strategy raises error."""
        creator = FeatureCreator()

        with pytest.raises(ValueError, match="Unknown strategy"):
            creator.create_binned_features(
                sample_df, columns=["feature_0"], n_bins=5, strategy="invalid"
            )

    def test_create_ratio_features(self, sample_df):
        """Test ratio feature creation."""
        creator = FeatureCreator()
        df_ratio = creator.create_ratio_features(
            sample_df, numerator_cols=["feature_0"], denominator_cols=["feature_1"]
        )

        assert "feature_0_div_feature_1" in df_ratio.columns
        # Should not have NaN or inf
        assert not df_ratio["feature_0_div_feature_1"].isna().any()
        assert not np.isinf(df_ratio["feature_0_div_feature_1"]).any()

    def test_create_ratio_features_multiple(self, sample_df):
        """Test ratio features with multiple columns."""
        creator = FeatureCreator()
        df_ratio = creator.create_ratio_features(
            sample_df,
            numerator_cols=["feature_0", "feature_1"],
            denominator_cols=["feature_2", "feature_3"],
        )

        # Should have 2*2=4 new features
        assert "feature_0_div_feature_2" in df_ratio.columns
        assert "feature_0_div_feature_3" in df_ratio.columns
        assert "feature_1_div_feature_2" in df_ratio.columns
        assert "feature_1_div_feature_3" in df_ratio.columns

    def test_create_aggregation_features(self, sample_df):
        """Test aggregation feature creation."""
        creator = FeatureCreator()
        df_agg = creator.create_aggregation_features(
            sample_df,
            columns=["feature_0", "feature_1", "feature_2"],
            operations=["mean", "std"],
        )

        expected_col = "feature_0_feature_1_feature_2_mean"
        assert expected_col in df_agg.columns

        # Verify the mean calculation
        expected_mean = sample_df[["feature_0", "feature_1", "feature_2"]].mean(axis=1)
        pd.testing.assert_series_equal(
            df_agg[expected_col], expected_mean, check_names=False
        )

    def test_create_aggregation_features_all_operations(self, sample_df):
        """Test all aggregation operations."""
        creator = FeatureCreator()
        df_agg = creator.create_aggregation_features(
            sample_df,
            columns=["feature_0", "feature_1"],
            operations=["mean", "std", "min", "max", "sum"],
        )

        base_name = "feature_0_feature_1"
        assert f"{base_name}_mean" in df_agg.columns
        assert f"{base_name}_std" in df_agg.columns
        assert f"{base_name}_min" in df_agg.columns
        assert f"{base_name}_max" in df_agg.columns
        assert f"{base_name}_sum" in df_agg.columns

    def test_create_aggregation_invalid_operation(self, sample_df):
        """Test that invalid aggregation operation raises error."""
        creator = FeatureCreator()

        with pytest.raises(ValueError, match="Unknown operation"):
            creator.create_aggregation_features(
                sample_df, columns=["feature_0", "feature_1"], operations=["invalid"]
            )


class TestFeatureSelector:
    """Tests for FeatureSelector class."""

    def test_init(self):
        """Test initialization."""
        selector = FeatureSelector()
        assert selector.selected_features is None
        assert selector.selector is None

    def test_select_k_best(self, classification_data):
        """Test k-best feature selection."""
        df, y = classification_data
        selector = FeatureSelector()

        X_selected, features = selector.select_k_best(df, y, k=5)

        assert X_selected.shape[1] == 5
        assert len(features) == 5
        assert X_selected.shape[0] == df.shape[0]

    def test_select_k_best_with_numpy(self, classification_data):
        """Test k-best with numpy arrays."""
        df, y = classification_data
        selector = FeatureSelector()

        X_selected, features = selector.select_k_best(df.values, y, k=5)

        assert X_selected.shape[1] == 5
        assert len(features) == 5

    def test_select_percentile(self, classification_data):
        """Test percentile-based feature selection."""
        df, y = classification_data
        selector = FeatureSelector()

        X_selected, features = selector.select_percentile(df, y, percentile=50)

        assert X_selected.shape[1] == 5  # 50% of 10 features
        assert len(features) == 5

    def test_select_percentile_different_values(self, classification_data):
        """Test percentile selection with different percentiles."""
        df, y = classification_data
        selector = FeatureSelector()

        X_selected_30, _ = selector.select_percentile(df, y, percentile=30)
        X_selected_70, _ = selector.select_percentile(df, y, percentile=70)

        assert X_selected_30.shape[1] < X_selected_70.shape[1]

    def test_select_by_rfe(self, classification_data):
        """Test RFE feature selection."""
        df, y = classification_data
        selector = FeatureSelector()

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        X_selected, features = selector.select_by_rfe(df, y, estimator, n_features=5)

        assert X_selected.shape[1] == 5
        assert len(features) == 5

    def test_get_feature_scores(self, classification_data):
        """Test getting feature scores."""
        df, y = classification_data
        selector = FeatureSelector()

        # Select features first
        selector.select_k_best(df, y, k=5)

        # Get scores
        scores_df = selector.get_feature_scores()

        assert scores_df is not None
        assert "feature" in scores_df.columns
        assert "score" in scores_df.columns
        assert len(scores_df) == 5

    def test_get_feature_scores_before_selection(self):
        """Test that getting scores before selection returns None."""
        selector = FeatureSelector()
        scores = selector.get_feature_scores()
        assert scores is None


class TestDimensionalityReducer:
    """Tests for DimensionalityReducer class."""

    def test_init(self):
        """Test initialization."""
        reducer = DimensionalityReducer()
        assert reducer.reducer is None
        assert reducer.n_components is None

    def test_apply_pca_with_n_components(self, classification_data):
        """Test PCA with specified number of components."""
        df, y = classification_data
        reducer = DimensionalityReducer()

        X_pca = reducer.apply_pca(df, n_components=5)

        assert X_pca.shape[1] == 5
        assert X_pca.shape[0] == df.shape[0]
        assert reducer.n_components == 5

    def test_apply_pca_with_variance_ratio(self, classification_data):
        """Test PCA with variance ratio."""
        df, y = classification_data
        reducer = DimensionalityReducer()

        X_pca = reducer.apply_pca(df, variance_ratio=0.95)

        assert X_pca.shape[1] <= df.shape[1]
        assert X_pca.shape[0] == df.shape[0]

    def test_get_explained_variance_ratio(self, classification_data):
        """Test getting explained variance ratio."""
        df, y = classification_data
        reducer = DimensionalityReducer()

        reducer.apply_pca(df, n_components=5)
        variance_ratio = reducer.get_explained_variance_ratio()

        assert variance_ratio is not None
        assert len(variance_ratio) == 5
        assert all(0 <= v <= 1 for v in variance_ratio)

    def test_get_cumulative_variance_ratio(self, classification_data):
        """Test getting cumulative variance ratio."""
        df, y = classification_data
        reducer = DimensionalityReducer()

        reducer.apply_pca(df, n_components=5)
        cum_variance = reducer.get_cumulative_variance_ratio()

        assert cum_variance is not None
        assert len(cum_variance) == 5
        # Should be monotonically increasing
        assert all(
            cum_variance[i] <= cum_variance[i + 1] for i in range(len(cum_variance) - 1)
        )

    def test_get_variance_before_pca(self):
        """Test that getting variance before PCA returns None."""
        reducer = DimensionalityReducer()

        assert reducer.get_explained_variance_ratio() is None
        assert reducer.get_cumulative_variance_ratio() is None


class TestCreateDateFeatures:
    """Tests for create_date_features function."""

    def test_create_date_features_basic(self):
        """Test basic date feature creation."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10),
                "value": np.random.randn(10),
            }
        )

        df_result = create_date_features(df, "date")

        assert "date_year" in df_result.columns
        assert "date_month" in df_result.columns
        assert "date_day" in df_result.columns
        assert "date_dayofweek" in df_result.columns
        assert "date_quarter" in df_result.columns
        assert "date_is_weekend" in df_result.columns
        assert "date" not in df_result.columns  # Should be dropped

    def test_create_date_features_keep_original(self):
        """Test keeping original date column."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10),
                "value": np.random.randn(10),
            }
        )

        df_result = create_date_features(df, "date", drop_original=False)

        assert "date" in df_result.columns
        assert "date_year" in df_result.columns

    def test_create_date_features_from_string(self):
        """Test creating date features from string dates."""
        df = pd.DataFrame(
            {"date": ["2023-01-01", "2023-01-02", "2023-01-03"], "value": [1, 2, 3]}
        )

        df_result = create_date_features(df, "date")

        assert "date_year" in df_result.columns
        assert df_result["date_year"].iloc[0] == 2023

    def test_create_date_features_weekend(self):
        """Test weekend detection."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=7),  # Starts on Sunday
                "value": range(7),
            }
        )

        df_result = create_date_features(df, "date")

        # Sunday and Saturday should be weekends
        assert df_result["date_is_weekend"].iloc[0] == 1  # Sunday
        assert df_result["date_is_weekend"].iloc[6] == 1  # Saturday
        assert df_result["date_is_weekend"].iloc[1] == 0  # Monday


class TestCreateTextFeatures:
    """Tests for create_text_features function."""

    def test_create_text_features_basic(self):
        """Test basic text feature creation."""
        df = pd.DataFrame(
            {"text": ["hello world", "this is a test", "short"], "value": [1, 2, 3]}
        )

        df_result = create_text_features(df, "text")

        assert "text_length" in df_result.columns
        assert "text_word_count" in df_result.columns
        assert "text_avg_word_length" in df_result.columns

    def test_create_text_features_values(self):
        """Test text feature values."""
        df = pd.DataFrame({"text": ["hello world", "test"], "value": [1, 2]})

        df_result = create_text_features(df, "text")

        # 'hello world' has 11 characters
        assert df_result["text_length"].iloc[0] == 11
        # 'hello world' has 2 words
        assert df_result["text_word_count"].iloc[0] == 2
        # Average word length should be 11/2 = 5.5
        assert df_result["text_avg_word_length"].iloc[0] == 5.5


class TestIntegration:
    """Integration tests for complete feature engineering workflows."""

    def test_complete_feature_engineering_workflow(self, classification_data):
        """Test complete feature engineering workflow."""
        df, y = classification_data

        # Step 1: Create features
        creator = FeatureCreator()
        df_enhanced = creator.create_interaction_features(
            df, column_pairs=[("feature_0", "feature_1")], operation="multiply"
        )

        # Step 2: Select features
        selector = FeatureSelector()
        X_selected, features = selector.select_k_best(df_enhanced, y, k=5)

        # Step 3: Apply PCA
        reducer = DimensionalityReducer()
        X_final = reducer.apply_pca(X_selected, n_components=3)

        # Verify final shape
        assert X_final.shape[0] == df.shape[0]
        assert X_final.shape[1] == 3

    def test_polynomial_then_selection(self, classification_data):
        """Test polynomial features followed by selection."""
        df, y = classification_data

        # Create polynomial features
        creator = FeatureCreator()
        df_poly = creator.create_polynomial_features(
            df[["feature_0", "feature_1", "feature_2"]], degree=2
        )

        original_features = df_poly.shape[1]

        # Select best features
        selector = FeatureSelector()
        X_selected, features = selector.select_k_best(df_poly, y, k=5)

        assert X_selected.shape[1] == 5
        assert X_selected.shape[1] < original_features

    def test_aggregation_and_pca(self, sample_df):
        """Test aggregation features with PCA."""
        # Create aggregation features
        creator = FeatureCreator()
        df_agg = creator.create_aggregation_features(
            sample_df,
            columns=["feature_0", "feature_1", "feature_2"],
            operations=["mean", "std", "min", "max"],
        )

        # Apply PCA
        reducer = DimensionalityReducer()
        X_pca = reducer.apply_pca(df_agg.drop("target", axis=1), n_components=3)

        assert X_pca.shape[0] == sample_df.shape[0]
        assert X_pca.shape[1] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
