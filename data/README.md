# Data Directory

This directory contains datasets used for classification machine learning experiments and tutorials.

## Directory Structure

```
data/
├── raw/           # Original, immutable datasets
├── processed/     # Cleaned and preprocessed data ready for modeling
└── README.md      # This file
```

## Datasets

### Raw Data (`raw/`)

Place your original datasets here. Common classification datasets include:

- **Binary Classification**
  - Diabetes dataset (predict diabetes occurrence)
  - Heart disease dataset (predict heart disease)
  - Customer churn dataset (predict if customer will leave)
  - Spam detection dataset (spam vs. not spam)

- **Multiclass Classification**
  - Iris dataset (classify flower species)
  - Wine quality dataset (classify wine types)
  - MNIST digits (classify handwritten digits 0-9)
  - Fashion MNIST (classify clothing items)

### Processed Data (`processed/`)

This folder contains preprocessed versions of raw data after:
- Handling missing values
- Encoding categorical variables
- Feature scaling/normalization
- Feature engineering
- Train/test splits

## Data Format

All datasets should be in one of the following formats:
- **CSV** (`.csv`) - Recommended for tabular data
- **JSON** (`.json`) - For structured data
- **Parquet** (`.parquet`) - For larger datasets
- **Excel** (`.xlsx`) - If source data is in Excel

## Adding New Datasets

1. Place original data in `raw/` folder
2. Create a preprocessing notebook in `notebooks/`
3. Save cleaned data to `processed/` folder
4. Update this README with dataset description
5. Document data sources and licenses

## Data Sources

<!-- Add your data sources here -->

**Example:**
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [scikit-learn built-in datasets](https://scikit-learn.org/stable/datasets.html)

## Important Notes

⚠️ **Do not commit large files (>100MB) to git**
- Use `.gitignore` to exclude large datasets
- Consider using Git LFS or cloud storage for large files
- Document download instructions instead

📝 **Data Privacy**
- Ensure datasets comply with privacy regulations
- Do not commit sensitive or personal data
- Anonymize data when necessary

📄 **Licensing**
- Check dataset licenses before use
- Document attribution requirements
- Respect terms of use

## Getting Started

To download sample datasets, run:

```python
from sklearn.datasets import load_iris, load_diabetes

# For multiclass classification
iris = load_iris(as_frame=True)
iris.frame.to_csv('data/raw/iris.csv', index=False)

# For binary classification
diabetes = load_diabetes(as_frame=True)
diabetes.frame.to_csv('data/raw/diabetes.csv', index=False)
```

Or use the provided data loading scripts in `src/data_preprocessing.py`.

---

*Last updated: November 2025*

