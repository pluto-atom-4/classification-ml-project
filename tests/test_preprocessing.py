"""
Tests for data preprocessing module.

Run with: pytest tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import tempfile  # noqa: E402
import shutil  # noqa: E402

from src.data_preprocessing import (  # noqa: E402
    DataLoader,
    DataPreprocessor,
    split_data,
    get_dataset_info,
    create_pipeline,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9],
            "feature3": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def sample_df_with_missing():
    """Create a sample DataFrame with missing values."""
    return pd.DataFrame(
        {
            "feature1": [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
            "feature2": [2.1, np.nan, 2.5, 2.7, np.nan, 3.1, 3.3, 3.5, 3.7, 3.9],
            "feature3": ["A", "B", "A", None, "A", "B", "A", "B", "A", "B"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def sample_df_with_outliers():
    """Create a sample DataFrame with outliers."""
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],  # 100 is an outlier
            "feature2": [2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for testing file operations."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_init(self, temp_data_dir):
        """Test DataLoader initialization."""
        loader = DataLoader(data_dir=temp_data_dir)
        assert loader.data_dir == Path(temp_data_dir)
        assert loader.raw_dir.exists()
        assert loader.processed_dir.exists()

    def test_load_sklearn_dataset_iris(self, temp_data_dir):
        """Test loading iris dataset from sklearn."""
        loader = DataLoader(data_dir=temp_data_dir)
        df = loader.load_sklearn_dataset("iris", save_raw=True)

        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 150  # Iris has 150 samples
        assert "target" in df.columns
        assert (loader.raw_dir / "iris.csv").exists()

    def test_load_sklearn_dataset_breast_cancer(self, temp_data_dir):
        """Test loading breast cancer dataset from sklearn."""
        loader = DataLoader(data_dir=temp_data_dir)
        df = loader.load_sklearn_dataset("breast_cancer", save_raw=False)

        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 569  # Breast cancer has 569 samples
        assert "target" in df.columns

    def test_load_sklearn_dataset_invalid(self, temp_data_dir):
        """Test loading invalid dataset raises error."""
        loader = DataLoader(data_dir=temp_data_dir)

        with pytest.raises(ValueError, match="not recognized"):
            loader.load_sklearn_dataset("invalid_dataset")

    def test_load_csv_from_raw(self, temp_data_dir, sample_df):
        """Test loading CSV from raw directory."""
        loader = DataLoader(data_dir=temp_data_dir)

        # Save a CSV first
        csv_path = loader.raw_dir / "test.csv"
        sample_df.to_csv(csv_path, index=False)

        # Load it
        loaded_df = loader.load_csv("test.csv", from_raw=True)

        assert isinstance(loaded_df, pd.DataFrame)
        assert loaded_df.shape == sample_df.shape
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_load_csv_from_processed(self, temp_data_dir, sample_df):
        """Test loading CSV from processed directory."""
        loader = DataLoader(data_dir=temp_data_dir)

        # Save a CSV to processed dir
        csv_path = loader.processed_dir / "test_processed.csv"
        sample_df.to_csv(csv_path, index=False)

        # Load it
        loaded_df = loader.load_csv("test_processed.csv", from_raw=False)

        assert isinstance(loaded_df, pd.DataFrame)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_load_csv_file_not_found(self, temp_data_dir):
        """Test loading non-existent CSV raises error."""
        loader = DataLoader(data_dir=temp_data_dir)

        with pytest.raises(FileNotFoundError):
            loader.load_csv("nonexistent.csv")

    def test_save_processed(self, temp_data_dir, sample_df):
        """Test saving processed data."""
        loader = DataLoader(data_dir=temp_data_dir)
        loader.save_processed(sample_df, "processed_test.csv")

        assert (loader.processed_dir / "processed_test.csv").exists()

        # Verify the saved file
        loaded_df = pd.read_csv(loader.processed_dir / "processed_test.csv")
        pd.testing.assert_frame_equal(loaded_df, sample_df)


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def test_init(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor.scaler is None
        assert preprocessor.label_encoders == {}
        assert preprocessor.imputer is None
        assert preprocessor.feature_names is None

    def test_handle_missing_values_mean(self, sample_df_with_missing):
        """Test handling missing values with mean strategy."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(
            sample_df_with_missing, strategy="mean"
        )

        assert df_clean["feature1"].isnull().sum() == 0
        assert df_clean["feature2"].isnull().sum() == 0
        assert df_clean.shape[0] == sample_df_with_missing.shape[0]

        # Original dataframe should not be modified
        assert sample_df_with_missing["feature1"].isnull().sum() > 0

    def test_handle_missing_values_median(self, sample_df_with_missing):
        """Test handling missing values with median strategy."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(
            sample_df_with_missing, strategy="median"
        )

        assert df_clean["feature1"].isnull().sum() == 0
        assert df_clean["feature2"].isnull().sum() == 0

    def test_handle_missing_values_specific_columns(self, sample_df_with_missing):
        """Test handling missing values for specific columns."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(
            sample_df_with_missing, strategy="mean", columns=["feature1"]
        )

        assert df_clean["feature1"].isnull().sum() == 0
        assert df_clean["feature2"].isnull().sum() > 0  # Not imputed

    def test_encode_categorical(self, sample_df):
        """Test encoding categorical variables."""
        preprocessor = DataPreprocessor()
        df_encoded = preprocessor.encode_categorical(sample_df)

        assert df_encoded["feature3"].dtype in [np.int32, np.int64]
        assert len(preprocessor.label_encoders) > 0
        assert "feature3" in preprocessor.label_encoders

        # Check encoding is reversible
        original_values = set(sample_df["feature3"].unique())
        encoded_values = df_encoded["feature3"].unique()
        decoded_values = set(
            preprocessor.label_encoders["feature3"].inverse_transform(encoded_values)
        )
        assert original_values == decoded_values

    def test_encode_categorical_specific_columns(self, sample_df):
        """Test encoding specific categorical columns."""
        preprocessor = DataPreprocessor()
        df_encoded = preprocessor.encode_categorical(sample_df, columns=["feature3"])

        assert df_encoded["feature3"].dtype in [np.int32, np.int64]
        assert "feature3" in preprocessor.label_encoders

    def test_scale_features_standard(self, sample_df):
        """Test standard scaling of features."""
        preprocessor = DataPreprocessor()
        df_scaled = preprocessor.scale_features(
            sample_df, columns=["feature1", "feature2"], method="standard"
        )

        # Check that mean is close to 0 and std is close to 1
        # (using ddof=0 for numpy compatibility)
        assert np.isclose(df_scaled["feature1"].mean(), 0, atol=1e-10)
        assert np.isclose(df_scaled["feature1"].std(ddof=0), 1, atol=1e-10)
        assert preprocessor.scaler is not None

    def test_scale_features_minmax(self, sample_df):
        """Test MinMax scaling of features."""
        preprocessor = DataPreprocessor()
        df_scaled = preprocessor.scale_features(
            sample_df, columns=["feature1", "feature2"], method="minmax"
        )

        # Check that values are between 0 and 1
        assert df_scaled["feature1"].min() == 0
        assert df_scaled["feature1"].max() == 1
        assert preprocessor.scaler is not None

    def test_scale_features_invalid_method(self, sample_df):
        """Test scaling with invalid method raises error."""
        preprocessor = DataPreprocessor()

        with pytest.raises(ValueError, match="Method must be"):
            preprocessor.scale_features(sample_df, method="invalid")

    def test_remove_outliers_iqr(self, sample_df_with_outliers):
        """Test removing outliers using IQR method."""
        preprocessor = DataPreprocessor()
        df_no_outliers = preprocessor.remove_outliers(
            sample_df_with_outliers, method="iqr", threshold=1.5
        )

        # Should remove the row with value 100
        assert df_no_outliers.shape[0] < sample_df_with_outliers.shape[0]
        assert 100 not in df_no_outliers["feature1"].values

    def test_remove_outliers_zscore(self, sample_df_with_outliers):
        """Test removing outliers using z-score method."""
        preprocessor = DataPreprocessor()
        df_no_outliers = preprocessor.remove_outliers(
            sample_df_with_outliers,
            method="zscore",
            threshold=2,  # Lower threshold to ensure outlier is detected
        )

        # Should remove the row with value 100
        assert df_no_outliers.shape[0] < sample_df_with_outliers.shape[0]

    def test_remove_outliers_invalid_method(self, sample_df):
        """Test removing outliers with invalid method raises error."""
        preprocessor = DataPreprocessor()

        with pytest.raises(ValueError, match="Method must be"):
            preprocessor.remove_outliers(sample_df, method="invalid")

    def test_prepare_features_target(self, sample_df):
        """Test separating features and target."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_features_target(sample_df, "target")

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "target" not in X.columns
        assert X.shape[0] == sample_df.shape[0]
        assert len(y) == sample_df.shape[0]

    def test_prepare_features_target_with_drop_columns(self, sample_df):
        """Test separating features and target with additional columns
        to drop."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_features_target(
            sample_df, "target", drop_columns=["feature3"]
        )

        assert "target" not in X.columns
        assert "feature3" not in X.columns
        assert X.shape[1] == 2  # Only feature1 and feature2

    def test_prepare_features_target_invalid_column(self, sample_df):
        """Test preparing features with invalid target column raises
        error."""
        preprocessor = DataPreprocessor()

        with pytest.raises(ValueError, match="not found"):
            preprocessor.prepare_features_target(sample_df, "invalid_column")


class TestSplitData:
    """Tests for split_data function."""

    def test_split_data_default(self, sample_df):
        """Test splitting data with default parameters."""
        X = sample_df[["feature1", "feature2"]]
        y = sample_df["target"]

        X_train, X_test, y_train, y_test = split_data(X, y)

        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_split_data_test_size(self, sample_df):
        """Test splitting data with custom test size."""
        X = sample_df[["feature1", "feature2"]]
        y = sample_df["target"]

        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)

        assert len(X_test) == 3  # 30% of 10
        assert len(X_train) == 7

    def test_split_data_no_stratify(self, sample_df):
        """Test splitting data without stratification."""
        X = sample_df[["feature1", "feature2"]]
        y = sample_df["target"]

        X_train, X_test, y_train, y_test = split_data(X, y, stratify=False)

        assert len(X_train) + len(X_test) == len(X)

    def test_split_data_random_state(self, sample_df):
        """Test that random state produces reproducible splits."""
        X = sample_df[["feature1", "feature2"]]
        y = sample_df["target"]

        X_train1, X_test1, y_train1, y_test1 = split_data(X, y, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = split_data(X, y, random_state=42)

        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)


class TestGetDatasetInfo:
    """Tests for get_dataset_info function."""

    def test_get_dataset_info(self, sample_df):
        """Test getting dataset information."""
        info = get_dataset_info(sample_df)

        assert info["shape"] == sample_df.shape
        assert len(info["columns"]) == 4
        assert "feature1" in info["numeric_columns"]
        assert "feature3" in info["categorical_columns"]
        assert info["duplicates"] == 0

    def test_get_dataset_info_with_missing(self, sample_df_with_missing):
        """Test getting dataset info with missing values."""
        info = get_dataset_info(sample_df_with_missing)

        assert info["missing_values"]["feature1"] > 0
        assert info["missing_values"]["feature2"] > 0
        assert info["missing_percentage"]["feature1"] > 0

    def test_get_dataset_info_with_duplicates(self):
        """Test getting dataset info with duplicate rows."""
        df = pd.DataFrame({"a": [1, 2, 1, 2], "b": [3, 4, 3, 4]})
        info = get_dataset_info(df)

        assert info["duplicates"] == 2


class TestCreatePipeline:
    """Tests for create_pipeline function."""

    def test_create_pipeline_basic(self, sample_df):
        """Test basic preprocessing pipeline."""
        X_train, X_test, y_train, y_test, preprocessor = create_pipeline(
            sample_df, target_column="target"
        )

        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert len(X_train) > len(X_test)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        assert isinstance(preprocessor, DataPreprocessor)

    def test_create_pipeline_with_missing_values(self, sample_df_with_missing):
        """Test pipeline with missing value handling."""
        X_train, X_test, y_train, y_test, preprocessor = create_pipeline(
            sample_df_with_missing,
            target_column="target",
            handle_missing=True,
            missing_strategy="mean",
        )

        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_create_pipeline_no_scaling(self, sample_df):
        """Test pipeline without feature scaling."""
        X_train, X_test, y_train, y_test, preprocessor = create_pipeline(
            sample_df, target_column="target", scale_features=False
        )

        # When scaling is disabled, the output remains as DataFrame
        assert isinstance(X_train, pd.DataFrame) or isinstance(X_train, np.ndarray)

        # Convert to array if needed for easier comparison
        if isinstance(X_train, pd.DataFrame):
            X_train_array = X_train.values
        else:
            X_train_array = X_train

        # Check that at least some values are > 1
        # (from the original numeric features)
        assert (X_train_array > 1).any()

    def test_create_pipeline_minmax_scaling(self, sample_df):
        """Test pipeline with MinMax scaling."""
        X_train, X_test, y_train, y_test, preprocessor = create_pipeline(
            sample_df,
            target_column="target",
            scale_features=True,
            scaling_method="minmax",
        )

        # Check values are between 0 and 1
        assert X_train.min() >= 0
        assert X_train.max() <= 1

    def test_create_pipeline_with_outlier_removal(self, sample_df_with_outliers):
        """Test pipeline with outlier removal."""
        original_size = len(sample_df_with_outliers)

        X_train, X_test, y_train, y_test, preprocessor = create_pipeline(
            sample_df_with_outliers, target_column="target", remove_outliers=True
        )

        # Total samples should be less than original after outlier removal
        total_samples = len(X_train) + len(X_test)
        assert total_samples < original_size

    def test_create_pipeline_custom_test_size(self, sample_df):
        """Test pipeline with custom test size."""
        X_train, X_test, y_train, y_test, preprocessor = create_pipeline(
            sample_df, target_column="target", test_size=0.3
        )

        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        assert np.isclose(test_ratio, 0.3, atol=0.1)


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_iris(self, temp_data_dir):
        """Test complete workflow with iris dataset."""
        # Load data
        loader = DataLoader(data_dir=temp_data_dir)
        df = loader.load_sklearn_dataset("iris", save_raw=True)

        # Get info
        info = get_dataset_info(df)
        assert info["shape"][0] == 150

        # Preprocess
        X_train, X_test, y_train, y_test, preprocessor = create_pipeline(
            df, target_column="target", scale_features=True
        )

        # Verify shapes
        assert X_train.shape[0] == 120  # 80% of 150
        assert X_test.shape[0] == 30  # 20% of 150
        assert X_train.shape[1] == 4  # 4 features in iris

        # Verify scaling
        assert np.isclose(X_train.mean(), 0, atol=0.1)

        # Save processed data
        processed_df = pd.DataFrame(X_train)
        loader.save_processed(processed_df, "iris_processed.csv")
        expected_path = Path(temp_data_dir) / "processed" / "iris_processed.csv"
        assert expected_path.exists()

    def test_full_workflow_breast_cancer(self, temp_data_dir):
        """Test complete workflow with breast cancer dataset."""
        # Load data
        loader = DataLoader(data_dir=temp_data_dir)
        df = loader.load_sklearn_dataset("breast_cancer", save_raw=False)

        # Preprocess with different settings
        X_train, X_test, y_train, y_test, preprocessor = create_pipeline(
            df,
            target_column="target",
            scale_features=True,
            scaling_method="minmax",
            test_size=0.25,
            random_state=123,
        )

        # Verify
        assert X_train.shape[0] > X_test.shape[0]
        assert X_train.min() >= 0
        # Allow for floating point precision errors
        assert X_train.max() <= 1.0 + 1e-10

        # Check class balance is maintained (stratified split)
        train_class_ratio = np.mean(y_train)
        test_class_ratio = np.mean(y_test)
        assert np.isclose(train_class_ratio, test_class_ratio, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
