# 🧮 Matrix Visualizations in Python

This project explores matrix operations and visualizations using Python, NumPy, Matplotlib, and Jupyter Notebook. It includes interactive widgets, 3D plots, and animated transformations to make learning linear algebra more engaging.

## 🚀 Features

- Matrix generation with adjustable size
- Heatmap and 3D surface visualizations
- Animated matrix rotation
- Interactive matrix operations (add, multiply, transpose, inverse, eigen)
- Pytest-based test suite for core functions
- Pre-commit hooks with Black and Flake8
- Jupyter Notebook with `ipywidgets` for interactivity

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/matrix-visualizations.git
cd matrix-visualizations
pip install -r requirements.txt
```
Or use the included `pyproject.toml`:
```bash
pip install .
```

## 🧑🧪 Testing
Run tests using Pytest:
```bash
pytest
```

## 📓 Jupyter Notebook
Launch the interactive notebook:
```bash
jupyter lab
```
Open `matrix_visualizations.ipyn`b to explore matrix operations visually.

## 🔧 Pre-commit Setup
Install and activate pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```
This will automatically run Black and Flake8 before each commit.

## 📁 Project Structure
```
matrix-visualizations/
├── matrix_ops.py               # Core matrix operations
├── test_matrix_ops.py          # Pytest test suite
├── matrix_visualizations.ipynb # Interactive notebook
├── requirements.txt
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```

## 🤝 Contributing
This project is licensed under the MIT License.

---

Let me know if you'd like to add badges (e.g., build status, Python version), a logo, or a demo GIF of the notebook in action!