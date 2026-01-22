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