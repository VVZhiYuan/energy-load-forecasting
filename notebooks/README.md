# Notebooks

Run notebooks in this order:

1. `01_eda.ipynb`: data quality, representative meter, and load patterns.
2. `02_baseline_models.ipynb`: 4-step and 96-step trajectory baselines.
3. `03_ml_models.ipynb`: direct per-step LightGBM models, gain, and SHAP.

Select `.venv\Scripts\python.exe` as the VS Code notebook kernel. Raw data must
exist at `data/raw/LD2011_2014.txt`.
