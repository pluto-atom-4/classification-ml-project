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

### Core Capabilities
- **Data Preprocessing**: Handling missing values, encoding categorical variables, scaling, outlier removal
- **Feature Engineering**: Polynomial features, interactions, binning, aggregations, selection, PCA
- **9 Classification Models**: Random Forest, Logistic Regression, SVM, Gradient Boosting, and more
- **Hyperparameter Tuning**: Grid Search and Random Search with cross-validation
- **Model Evaluation**: Confusion matrices, ROC curves, precision-recall curves, 9+ metrics
- **Model Persistence**: Save and load trained models
- **Feature Selection**: K-best, Percentile, RFE methods
- **Dimensionality Reduction**: PCA with variance analysis

### Quality Assurance
- **Comprehensive Testing**: 151 tests with 79% code coverage
- **Production Ready**: Clean API, error handling, type hints
- **Well Documented**: 7 comprehensive guides + API docs
- **Best Practices**: SOLID principles, DRY code, modular design

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

---

## 🛠️ Development setup

Use a virtual environment to avoid polluting your global Python environment.

1. Create and activate a virtual environment (Windows / bash):

```bash
python -m venv .venv
source .venv/Scripts/activate  # on Windows (Git Bash / WSL), or
# .venv\Scripts\activate.bat  # on Windows Command Prompt
```

2. Upgrade pip and install the project with optional dev tools:

```bash
python -m pip install --upgrade pip
pip install -e .[dev,notebook]
```

This installs the package in editable mode plus the `dev` and `notebook` extras (testing, code-quality, and jupyter tools). If you only want the minimal runtime dependencies (including scikit-learn), run:

```bash
pip install -e .
```

Notes:
- If you prefer using `requirements.txt`, you can run `pip install -r requirements.txt` instead.
- To install only dev tools: `pip install -e .[dev]`.

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

### Complete Classification Workflow

```python
from src.data_preprocessing import DataLoader, create_pipeline
from src.models import BinaryClassifier, ModelTrainer
from src.evaluation import evaluate_model, compare_models

# 1. Load and preprocess data
loader = DataLoader()
df = loader.load_sklearn_dataset('breast_cancer')

X_train, X_test, y_train, y_test, _ = create_pipeline(
    df, target_column='target', scale_features=True
)

# 2. Train a single model
classifier = BinaryClassifier(model_type='random_forest', n_estimators=100)
classifier.fit(X_train, y_train)

# 3. Or train multiple models
trainer = ModelTrainer()
models = trainer.train_multiple_models(X_train, y_train)

# 4. Evaluate and compare
comparison = compare_models(models, X_test, y_test)
print(comparison)

# 5. Use best model for predictions
best_model = trainer.get_model('Random Forest')
predictions = best_model.predict(X_test)
```

### Quick Start Examples

**Binary Classification:**
```python
from src.models import BinaryClassifier

# Simple one-liner approach
classifier = BinaryClassifier(model_type='random_forest')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
```

**Hyperparameter Tuning:**
```python
from src.models import HyperparameterTuner
from sklearn.ensemble import RandomForestClassifier

base_model = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

tuner = HyperparameterTuner(base_model, param_grid, cv=5)
best_model = tuner.grid_search(X_train, y_train)
print(f"Best params: {tuner.get_best_params()}")
```

**Model Evaluation:**
```python
from src.evaluation import ClassificationEvaluator

evaluator = ClassificationEvaluator(model_name="My Classifier")
metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba)

# Create visualizations
fig1 = evaluator.plot_confusion_matrix(y_test, y_pred)
fig2 = evaluator.plot_roc_curve(y_test, y_pred_proba[:, 1])
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
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific module tests
pytest tests/test_preprocessing.py -v
pytest tests/test_evaluation.py -v
pytest tests/test_models.py -v
```

**Test Statistics:**
- ✅ 113 tests total (all passing)
- ✅ 81% overall coverage
- ✅ 38 preprocessing tests (89% coverage)
- ✅ 37 evaluation tests (85% coverage)
- ✅ 38 models tests (72% coverage)

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