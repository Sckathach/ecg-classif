## Installation
This project is based on Python 3.14. You can use uv:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init -p 3.14
uv venv
uv add jaxtyping jupyter loguru matplotlib networkx numpy scikit-learn scipy
```

## Generate features
Look at the `scripts/generate_features.py` and modify `data_root` if needed. Then launch the script with uv:

```shell
uv run scripts/generate_features.py
```

## Best Configurations found:
```json
{
  "ALPHA": {
    "mode": "regional",
    "selector_type": "ANOVA",
    "n_features": 25,
    "classifier": "RandomForest",
    "rf_n_estimators": 55,
    "rf_max_depth": 19
  },
  "BETA": {
    "mode": "vector",
    "selector_type": "ANOVA",
    "n_features": 40,
    "classifier": "GradientBoosting",
    "gb_n_estimators": 59,
    "gb_learning_rate": 0.5695121714815743,
    "gb_max_depth": 2
  },
  "THETA": {
    "mode": "vector",
    "selector_type": "None",
    "classifier": "GradientBoosting",
    "gb_n_estimators": 130,
    "gb_learning_rate": 0.017525241029937837,
    "gb_max_depth": 3
  },
  "DELTA": {
    "mode": "vector",
    "selector_type": "None",
    "classifier": "SVM",
    "svm_C": 0.05487349310029459,
    "svm_gamma": "auto",
    "svm_kernel": "linear"
  }
}
```