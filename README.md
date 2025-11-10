# 🤖 Classification Models in Machine Learning

This project implements and explores classification models in machine learning using Python, scikit-learn, and Jupyter Notebooks. It covers binary and multiclass classification with real-world datasets, following Microsoft Learn's machine learning training path.

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

or use pyproject.toml
```bash
pip install -e .
```

## 📊 Datasets

The project uses the following datasets:

- Diabetes Dataset: Binary classification of diabetes presence
- Iris Dataset: Multiclass classification of iris species
- Custom datasets: Add your own datasets in the `data/raw` folder

## 🧑‍💻 Usage

Run Jupyter Notebooks

```bash
jupyter lab
```
Navigate to notebooks/ and run:
1. 01_data_exploration.ipynb - Explore and visualize datasets
2. 02_binary_classification.ipynb - Build and evaluate binary models
3. 03_multiclass_classification.ipynb - Implement multiclass models
4. 04_model_evaluation.ipynb - compare and evaluate models


Use as Python Module

```python
from src.models import BinaryClassifier
from src.evaluation import evaluate_model

# Train a model
clf = BinaryClassifier(model_type='logistic_regression')
clf.fit(X_train, y_train)

# Evaluate
metrics = evaluate_model(clf, X_test, y_test)
print(metrics)
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

## 🔧 Pre-commit Hooks

Install pre-commit hooks for code quality:

```bash
pre-commit install
```
This runs Black, Flask8, and type checking on each commit.

## 📈 Key Concepts Covered

- Classification vs. Regression
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