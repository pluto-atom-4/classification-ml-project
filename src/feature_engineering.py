"""
Feature engineering module for machine learning projects.

This module provides utilities for creating, selecting, and transforming features
to improve model performance.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple, Callable
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    RFE,
    f_classif,
)
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator


class FeatureCreator:
    """Create new features from existing ones."""

    def __init__(self):
        """Initialize FeatureCreator."""
        self.created_features = []

    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        degree: int = 2,
        include_bias: bool = False,
    ) -> pd.DataFrame:
        """
        Create polynomial features.

        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features from. If None, use all numeric
            degree: Degree of polynomial features
            include_bias: Whether to include bias column

        Returns:
            DataFrame with polynomial features added
        """
        df_copy = df.copy()

        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()

        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_features = poly.fit_transform(df_copy[columns])

        # Get feature names
        feature_names = poly.get_feature_names_out(columns)

        # Create DataFrame with new features
        poly_df = pd.DataFrame(
            poly_features, columns=feature_names, index=df_copy.index
        )

        # Drop original columns and merge
        df_result = df_copy.drop(columns=columns)
        df_result = pd.concat([df_result, poly_df], axis=1)

        self.created_features.extend(feature_names.tolist())

        return df_result

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        column_pairs: List[Tuple[str, str]],
        operation: str = "multiply",
    ) -> pd.DataFrame:
        """
        Create interaction features between column pairs.

        Args:
            df: Input DataFrame
            column_pairs: List of column pairs to create interactions
            operation: Type of operation ('multiply', 'add', 'subtract', 'divide')

        Returns:
            DataFrame with interaction features added
        """
        df_copy = df.copy()

        for col1, col2 in column_pairs:
            feature_name = f"{col1}_{operation}_{col2}"

            if operation == "multiply":
                df_copy[feature_name] = df_copy[col1] * df_copy[col2]
            elif operation == "add":
                df_copy[feature_name] = df_copy[col1] + df_copy[col2]
            elif operation == "subtract":
                df_copy[feature_name] = df_copy[col1] - df_copy[col2]
            elif operation == "divide":
                # Avoid division by zero
                df_copy[feature_name] = df_copy[col1] / (df_copy[col2] + 1e-10)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            self.created_features.append(feature_name)

        return df_copy

    def create_binned_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        n_bins: int = 5,
        strategy: str = "quantile",
    ) -> pd.DataFrame:
        """
        Create binned (discretized) features.

        Args:
            df: Input DataFrame
            columns: Columns to bin
            n_bins: Number of bins
            strategy: Binning strategy ('uniform', 'quantile')

        Returns:
            DataFrame with binned features added
        """
        df_copy = df.copy()

        for col in columns:
            feature_name = f"{col}_binned"

            if strategy == "uniform":
                df_copy[feature_name] = pd.cut(
                    df_copy[col], bins=n_bins, labels=False, duplicates="drop"
                )
            elif strategy == "quantile":
                df_copy[feature_name] = pd.qcut(
                    df_copy[col], q=n_bins, labels=False, duplicates="drop"
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            self.created_features.append(feature_name)

        return df_copy

    def create_ratio_features(
        self, df: pd.DataFrame, numerator_cols: List[str], denominator_cols: List[str]
    ) -> pd.DataFrame:
        """
        Create ratio features.

        Args:
            df: Input DataFrame
            numerator_cols: Columns to use as numerators
            denominator_cols: Columns to use as denominators

        Returns:
            DataFrame with ratio features added
        """
        df_copy = df.copy()

        for num_col in numerator_cols:
            for denom_col in denominator_cols:
                feature_name = f"{num_col}_div_{denom_col}"
                # Avoid division by zero
                df_copy[feature_name] = df_copy[num_col] / (df_copy[denom_col] + 1e-10)
                self.created_features.append(feature_name)

        return df_copy

    def create_aggregation_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        operations: List[str] = ["mean", "std", "min", "max"],
    ) -> pd.DataFrame:
        """
        Create aggregation features across columns.

        Args:
            df: Input DataFrame
            columns: Columns to aggregate
            operations: List of operations ('mean', 'std', 'min', 'max', 'sum')

        Returns:
            DataFrame with aggregation features added
        """
        df_copy = df.copy()

        for operation in operations:
            feature_name = f"{'_'.join(columns)}_{operation}"

            if operation == "mean":
                df_copy[feature_name] = df_copy[columns].mean(axis=1)
            elif operation == "std":
                df_copy[feature_name] = df_copy[columns].std(axis=1)
            elif operation == "min":
                df_copy[feature_name] = df_copy[columns].min(axis=1)
            elif operation == "max":
                df_copy[feature_name] = df_copy[columns].max(axis=1)
            elif operation == "sum":
                df_copy[feature_name] = df_copy[columns].sum(axis=1)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            self.created_features.append(feature_name)

        return df_copy


class FeatureSelector:
    """Select the most important features."""

    def __init__(self):
        """Initialize FeatureSelector."""
        self.selected_features = None
        self.selector = None

    def select_k_best(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        k: int = 10,
        score_func: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Select k best features based on univariate statistical tests.

        Args:
            X: Features
            y: Target variable
            k: Number of features to select
            score_func: Scoring function (default: f_classif)

        Returns:
            Tuple of (selected features, feature names)
        """
        if score_func is None:
            score_func = f_classif

        self.selector = SelectKBest(score_func=score_func, k=k)
        X_selected = self.selector.fit_transform(X, y)

        # Get selected feature names
        if isinstance(X, pd.DataFrame):
            mask = self.selector.get_support()
            self.selected_features = X.columns[mask].tolist()
        else:
            self.selected_features = [
                f"Feature_{i}"
                for i in range(X.shape[1])
                if self.selector.get_support()[i]
            ]

        return X_selected, self.selected_features

    def select_percentile(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        percentile: int = 50,
        score_func: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Select features based on percentile of highest scores.

        Args:
            X: Features
            y: Target variable
            percentile: Percentile of features to keep
            score_func: Scoring function (default: f_classif)

        Returns:
            Tuple of (selected features, feature names)
        """
        if score_func is None:
            score_func = f_classif

        self.selector = SelectPercentile(score_func=score_func, percentile=percentile)
        X_selected = self.selector.fit_transform(X, y)

        # Get selected feature names
        if isinstance(X, pd.DataFrame):
            mask = self.selector.get_support()
            self.selected_features = X.columns[mask].tolist()
        else:
            self.selected_features = [
                f"Feature_{i}"
                for i in range(X.shape[1])
                if self.selector.get_support()[i]
            ]

        return X_selected, self.selected_features

    def select_by_rfe(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        estimator: BaseEstimator,
        n_features: int = 10,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Select features using Recursive Feature Elimination.

        Args:
            X: Features
            y: Target variable
            estimator: Model to use for feature ranking
            n_features: Number of features to select

        Returns:
            Tuple of (selected features, feature names)
        """
        self.selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = self.selector.fit_transform(X, y)

        # Get selected feature names
        if isinstance(X, pd.DataFrame):
            mask = self.selector.get_support()
            self.selected_features = X.columns[mask].tolist()
        else:
            self.selected_features = [
                f"Feature_{i}"
                for i in range(X.shape[1])
                if self.selector.get_support()[i]
            ]

        return X_selected, self.selected_features

    def get_feature_scores(self) -> Optional[pd.DataFrame]:
        """
        Get feature scores from the selector.

        Returns:
            DataFrame with features and their scores
        """
        if self.selector is None:
            return None

        if hasattr(self.selector, "scores_"):
            scores_df = pd.DataFrame(
                {
                    "feature": self.selected_features,
                    "score": self.selector.scores_[self.selector.get_support()],
                }
            )
            return scores_df.sort_values("score", ascending=False)

        return None


class DimensionalityReducer:
    """Reduce the dimensionality of features."""

    def __init__(self):
        """Initialize DimensionalityReducer."""
        self.reducer = None
        self.n_components = None

    def apply_pca(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        n_components: Optional[int] = None,
        variance_ratio: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply Principal Component Analysis.

        Args:
            X: Features
            n_components: Number of components to keep
            variance_ratio: Keep enough components to explain this ratio of variance

        Returns:
            Transformed features
        """
        if n_components is None and variance_ratio is None:
            n_components = min(X.shape[1], X.shape[0]) // 2

        if variance_ratio is not None:
            self.reducer = PCA(n_components=variance_ratio)
        else:
            self.reducer = PCA(n_components=n_components)

        X_reduced = self.reducer.fit_transform(X)
        self.n_components = self.reducer.n_components_

        return X_reduced

    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        Get the explained variance ratio of each component.

        Returns:
            Array of explained variance ratios
        """
        if self.reducer is None:
            return None

        return self.reducer.explained_variance_ratio_

    def get_cumulative_variance_ratio(self) -> Optional[np.ndarray]:
        """
        Get the cumulative explained variance ratio.

        Returns:
            Array of cumulative explained variance ratios
        """
        if self.reducer is None:
            return None

        return np.cumsum(self.reducer.explained_variance_ratio_)


def create_date_features(
    df: pd.DataFrame, date_column: str, drop_original: bool = True
) -> pd.DataFrame:
    """
    Create features from a date column.

    Args:
        df: Input DataFrame
        date_column: Name of the date column
        drop_original: Whether to drop the original date column

    Returns:
        DataFrame with date features added
    """
    df_copy = df.copy()

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])

    # Extract features
    df_copy[f"{date_column}_year"] = df_copy[date_column].dt.year
    df_copy[f"{date_column}_month"] = df_copy[date_column].dt.month
    df_copy[f"{date_column}_day"] = df_copy[date_column].dt.day
    df_copy[f"{date_column}_dayofweek"] = df_copy[date_column].dt.dayofweek
    df_copy[f"{date_column}_quarter"] = df_copy[date_column].dt.quarter
    df_copy[f"{date_column}_is_weekend"] = (
        df_copy[date_column].dt.dayofweek >= 5
    ).astype(int)

    if drop_original:
        df_copy = df_copy.drop(columns=[date_column])

    return df_copy


def create_text_features(
    df: pd.DataFrame, text_column: str, max_features: int = 100
) -> pd.DataFrame:
    """
    Create features from a text column.

    Args:
        df: Input DataFrame
        text_column: Name of the text column
        max_features: Maximum number of features to create

    Returns:
        DataFrame with text features added
    """
    df_copy = df.copy()

    # Basic text features
    df_copy[f"{text_column}_length"] = df_copy[text_column].str.len()
    df_copy[f"{text_column}_word_count"] = df_copy[text_column].str.split().str.len()
    df_copy[f"{text_column}_avg_word_length"] = (
        df_copy[f"{text_column}_length"] / df_copy[f"{text_column}_word_count"]
    )

    return df_copy


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    print("=" * 60)
    print("Feature Engineering Examples")
    print("=" * 60)

    # Generate sample data
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=3,
        n_classes=2,
        random_state=42,
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

    # Example 1: Feature Creation
    print("\n" + "=" * 60)
    print("Example 1: Feature Creation")
    print("=" * 60)

    creator = FeatureCreator()

    # Create polynomial features
    df_poly = creator.create_polynomial_features(
        df[["feature_0", "feature_1"]], degree=2
    )
    print(f"\nOriginal features: {df.shape[1]}")
    print(f"After polynomial features: {df_poly.shape[1]}")

    # Create interaction features
    df_interaction = creator.create_interaction_features(
        df,
        column_pairs=[("feature_0", "feature_1"), ("feature_2", "feature_3")],
        operation="multiply",
    )
    print(f"After interaction features: {df_interaction.shape[1]}")

    # Example 2: Feature Selection
    print("\n" + "=" * 60)
    print("Example 2: Feature Selection")
    print("=" * 60)

    selector = FeatureSelector()

    # Select k best features
    X_selected, selected_features = selector.select_k_best(df, y, k=5)
    print(f"\nOriginal features: {df.shape[1]}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Selected feature names: {selected_features}")

    # Select by RFE
    estimator = RandomForestClassifier(n_estimators=10, random_state=42)
    X_rfe, rfe_features = selector.select_by_rfe(df, y, estimator, n_features=5)
    print(f"\nRFE selected features: {rfe_features}")

    # Example 3: Dimensionality Reduction
    print("\n" + "=" * 60)
    print("Example 3: Dimensionality Reduction (PCA)")
    print("=" * 60)

    reducer = DimensionalityReducer()
    X_pca = reducer.apply_pca(df, n_components=5)

    print(f"\nOriginal dimensions: {df.shape[1]}")
    print(f"Reduced dimensions: {X_pca.shape[1]}")
    print(f"Explained variance ratio: {reducer.get_explained_variance_ratio()}")
    print(f"Cumulative variance ratio: {reducer.get_cumulative_variance_ratio()}")

    # Example 4: Binning
    print("\n" + "=" * 60)
    print("Example 4: Feature Binning")
    print("=" * 60)

    creator2 = FeatureCreator()
    df_binned = creator2.create_binned_features(
        df, columns=["feature_0", "feature_1"], n_bins=5, strategy="quantile"
    )
    print(f"\nOriginal shape: {df.shape}")
    print(f"After binning: {df_binned.shape}")
    print(f"Created features: {creator2.created_features}")

    # Example 5: Aggregation Features
    print("\n" + "=" * 60)
    print("Example 5: Aggregation Features")
    print("=" * 60)

    creator3 = FeatureCreator()
    df_agg = creator3.create_aggregation_features(
        df,
        columns=["feature_0", "feature_1", "feature_2"],
        operations=["mean", "std", "max"],
    )
    print(f"\nCreated aggregation features: {creator3.created_features}")
    print(f"Final shape: {df_agg.shape}")
