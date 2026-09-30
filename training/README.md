# Training

This directory separates reproducible production training from exploratory experimentation.

- `train_model.py` trains the Random Forest used by the Streamlit app and exports `models/model.npz` plus `models/model_meta.json`.
- `experimental_pipeline.py` preserves the larger experimental pipeline with clustering and multiple model variants.

Run production training from the repository root:

```bash
python training/train_model.py
```

Install training dependencies first:

```bash
pip install -r requirements-train.txt
```
