"""
Data preprocessing module for classification machine learning projects.

This module provides functions for loading, cleaning, and preprocessing data
for binary and multiclass classification tasks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_diabetes


class DataLoader:
    """Load datasets from various sources."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader.

        Args:
            data_dir: Root directory for data files
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_sklearn_dataset(
        self, dataset_name: str, save_raw: bool = True
    ) -> pd.DataFrame:
        """
        Load a built-in scikit-learn dataset.

        Args:
            dataset_name: Name of the dataset ('iris', 'breast_cancer', 'wine', 'diabetes')
            save_raw: Whether to save the dataset to raw directory

        Returns:
            DataFrame containing the dataset

        Raises:
            ValueError: If dataset_name is not recognized
        """
        datasets = {
            "iris": load_iris,
            "breast_cancer": load_breast_cancer,
            "wine": load_wine,
            "diabetes": load_diabetes,
        }

        if dataset_name not in datasets:
            raise ValueError(
                f"Dataset '{dataset_name}' not recognized. "
                f"Available datasets: {list(datasets.keys())}"
            )

        data = datasets[dataset_name](as_frame=True)
        df = data.frame

        if save_raw:
            output_path = self.raw_dir / f"{dataset_name}.csv"
            df.to_csv(output_path, index=False)
            print(f"Dataset saved to {output_path}")

        return df

    def load_csv(self, filename: str, from_raw: bool = True) -> pd.DataFrame:
        """
        Load a CSV file from raw or processed directory.

        Args:
            filename: Name of the CSV file
            from_raw: If True, load from raw directory; otherwise from processed

        Returns:
            DataFrame containing the data
        """
        directory = self.raw_dir if from_raw else self.processed_dir
        filepath = directory / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        return pd.read_csv(filepath)

    def save_processed(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save a processed DataFrame to the processed directory.

        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")


class DataPreprocessor:
    """Preprocess data for machine learning models."""

    def __init__(self):
        """Initialize DataPreprocessor."""
        self.scaler = None
        self.label_encoders = {}
        self.imputer = None
        self.feature_names = None

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "mean",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
            columns: Specific columns to impute. If None, impute all numeric columns

        Returns:
            DataFrame with missing values handled
        """
        df_copy = df.copy()

        if columns is None:
            # Get numeric columns with missing values
            columns = (
                df_copy.select_dtypes(include=[np.number])
                .columns[df_copy.select_dtypes(include=[np.number]).isnull().any()]
                .tolist()
            )

        if not columns:
            return df_copy

        self.imputer = SimpleImputer(strategy=strategy)
        df_copy[columns] = self.imputer.fit_transform(df_copy[columns])

        return df_copy

    def encode_categorical(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Encode categorical variables using LabelEncoder.

        Args:
            df: Input DataFrame
            columns: List of columns to encode. If None, encode all object columns

        Returns:
            DataFrame with encoded categorical variables
        """
        df_copy = df.copy()

        if columns is None:
            columns = df_copy.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        for col in columns:
            if col in df_copy.columns:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                self.label_encoders[col] = le

        return df_copy

    def scale_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "standard",
    ) -> pd.DataFrame:
        """
        Scale numeric features.

        Args:
            df: Input DataFrame
            columns: Columns to scale. If None, scale all numeric columns
            method: Scaling method ('standard' or 'minmax')

        Returns:
            DataFrame with scaled features

        Raises:
            ValueError: If method is not recognized
        """
        if method not in ["standard", "minmax"]:
            raise ValueError("Method must be 'standard' or 'minmax'")

        df_copy = df.copy()

        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()

        self.feature_names = columns

        if method == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        df_copy[columns] = self.scaler.fit_transform(df_copy[columns])

        return df_copy

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Remove outliers from the dataset.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers. If None, check all
                numeric columns
            method: Method to detect outliers ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
                (1.5 for IQR, 3 for z-score)

        Returns:
            DataFrame with outliers removed

        Raises:
            ValueError: If method is not recognized
        """
        if method not in ["iqr", "zscore"]:
            raise ValueError("Method must be 'iqr' or 'zscore'")

        df_copy = df.copy()

        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()

        mask = pd.Series([True] * len(df_copy), index=df_copy.index)

        for col in columns:
            if method == "iqr":
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask &= (df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)
            else:  # zscore
                z_scores = np.abs(
                    (df_copy[col] - df_copy[col].mean()) / df_copy[col].std()
                )
                mask &= z_scores < threshold

        return df_copy[mask]

    def prepare_features_target(
        self,
        df: pd.DataFrame,
        target_column: str,
        drop_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.

        Args:
            df: Input DataFrame
            target_column: Name of the target column
            drop_columns: Additional columns to drop from features

        Returns:
            Tuple of (features DataFrame, target Series)

        Raises:
            ValueError: If target_column is not in DataFrame
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        df_copy = df.copy()

        # Separate target
        y = df_copy[target_column]

        # Prepare features
        columns_to_drop = [target_column]
        if drop_columns:
            columns_to_drop.extend(drop_columns)

        X = df_copy.drop(columns=columns_to_drop)

        return X, y


def split_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.

    Args:
        X: Features
        y: Target variable
        test_size: Proportion of dataset to include in test split
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified splitting
            (recommended for classification)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get comprehensive information about a dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary containing dataset information
    """
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": ((df.isnull().sum() / len(df) * 100).to_dict()),
        "numeric_columns": (df.select_dtypes(include=[np.number]).columns.tolist()),
        "categorical_columns": df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist(),
        "duplicates": df.duplicated().sum(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }

    return info


def create_pipeline(
    df: pd.DataFrame,
    target_column: str,
    handle_missing: bool = True,
    missing_strategy: str = "mean",
    encode_categorical: bool = True,
    scale_features: bool = True,
    scaling_method: str = "standard",
    remove_outliers: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataPreprocessor]:
    """
    Complete preprocessing pipeline from raw data to train/test split.

    Args:
        df: Input DataFrame
        target_column: Name of the target column
        handle_missing: Whether to handle missing values
        missing_strategy: Strategy for handling missing values
        encode_categorical: Whether to encode categorical variables
        scale_features: Whether to scale features
        scaling_method: Method for scaling ('standard' or 'minmax')
        remove_outliers: Whether to remove outliers
        test_size: Proportion for test set
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    preprocessor = DataPreprocessor()
    df_processed = df.copy()

    # Handle missing values
    if handle_missing:
        df_processed = preprocessor.handle_missing_values(
            df_processed, strategy=missing_strategy
        )

    # Encode categorical variables
    if encode_categorical:
        df_processed = preprocessor.encode_categorical(df_processed)

    # Remove outliers (before splitting to avoid data leakage in some cases)
    if remove_outliers:
        df_processed = preprocessor.remove_outliers(df_processed)

    # Separate features and target
    X, y = preprocessor.prepare_features_target(df_processed, target_column)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features (fit on training data only to prevent data leakage)
    if scale_features:
        if scaling_method == "standard":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        preprocessor.scaler = scaler

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Example usage
    print("Loading sample dataset...")
    loader = DataLoader()

    # Load iris dataset
    df = loader.load_sklearn_dataset("iris", save_raw=True)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")

    # Get dataset info
    info = get_dataset_info(df)
    print("\nDataset info:")
    print(f"  - Numeric columns: {info['numeric_columns']}")
    print(f"  - Missing values: {sum(info['missing_values'].values())}")

    # Complete preprocessing pipeline
    X_train, X_test, y_train, y_test, preprocessor = create_pipeline(
        df, target_column="target", scale_features=True
    )

    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print("Class distribution in training set:")
    print(pd.Series(y_train).value_counts().sort_index())
