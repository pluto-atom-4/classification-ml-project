# 🤖 Classification Models in Machine Learning

This project implements and explores classification models in machine learning using Python, scikit-learn, and Jupyter Notebooks. It covers binary and multiclass classification with real-world datasets, following [Microsoft Learn's machine learning training path](https://learn.microsoft.com/en-us/training/paths/understand-machine-learning/).

## 🎯 Learning Objectives

- Understand classification concepts and use cases
- Build binary classification models (e.g., logistic regression, decision trees)
- Implement multiclass classification algorithms
- Evaluate model performance using metrics (accuracy, precision, recall, F1-score)
- Visualize decision boundaries and confusion matrices
- Compare different classification algorithms

## 🚀 Features

- **Data Preprocessing**: Handling missing values, encoding categorical variables, scaling
- **Binary Classification**: Logistic regression, SVM, decision trees
- **Multiclass Classification**: One-vs-Rest, Random Forest, Neural Networks
- **Model Evaluation**: Confusion matrices, ROC curves, precision-recall curves
- **Interactive Notebooks**: Step-by-step tutorials with visualizations
- **Automated Testing**: Pytest suite for data processing and model functions

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/classification-ml-project.git
cd classification-ml-project
pip install -r requirements.txt
```

Or use `pyproject.toml`:
```bash
pip install -e .
```

## 📊 Datasets

The project uses the following datasets:

- **Diabetes Dataset**: Binary classification (has diabetes / no diabetes)
- **Iris Dataset**: Multiclass classification (3 flower species)
- **Breast Cancer Dataset**: Binary classification (malignant / benign)
- **Wine Dataset**: Multiclass classification (wine quality)
- **Custom datasets**: Add your own in `data/raw/`

## 🧑‍💻 Usage

### Run Jupyter Notebooks

```bash
jupyter lab
```

Navigate to `notebooks/` and run:
1. `01_data_exploration.ipynb` - Explore and visualize datasets
2. `02_binary_classification.ipynb` - Build binary classifiers
3. `03_multiclass_classification.ipynb` - Implement multiclass models
4. `04_model_evaluation.ipynb` - Compare and evaluate models

### Use as Python Module

```python
from src.data_preprocessing import DataLoader, create_pipeline

# Load a dataset
loader = DataLoader()
df = loader.load_sklearn_dataset('iris')

# Complete preprocessing pipeline
X_train, X_test, y_train, y_test, preprocessor = create_pipeline(
    df,
    target_column='target',
    scale_features=True
)

# Now ready for model training!
```

### Use Data Preprocessing Module

```python
from src.data_preprocessing import DataPreprocessor, split_data

# Create preprocessor
preprocessor = DataPreprocessor()

# Handle missing values
df_clean = preprocessor.handle_missing_values(df, strategy='mean')

# Encode categorical variables
df_encoded = preprocessor.encode_categorical(df_clean)

# Scale features
df_scaled = preprocessor.scale_features(df_encoded, method='standard')

# Prepare features and target
X, y = preprocessor.prepare_features_target(df_scaled, 'target')

# Split data
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

With coverage report:

```bash
pytest --cov=src tests/
```

Run specific test file:

```bash
pytest tests/test_preprocessing.py -v
```

## 🔧 Pre-commit Hooks (Optional)

Install pre-commit hooks for code quality:

```bash
pre-commit install
```

This runs Black, Flake8, and type checking before each commit.

## 📈 Key Concepts Covered

- **Classification vs. Regression**
- Training and test data splitting
- Feature scaling and normalization
- Overfitting and underfitting
- Cross-validation
- Hyperparameter tuning
- Model interpretability

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License
MIT License

## 🔗 Resources

- [Microsoft Learn: Classification Models](https://learn.microsoft.com/en-us/training/paths/understand-machine-learning/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Machine Learning Glossary](https://ml-cheatsheet.readthedocs.io/)