# Architecture

## Runtime path

```text
Browser
  -> Streamlit UI (`app/streamlit_app.py`)
  -> feature construction
  -> compact Random Forest arrays (`models/model.npz`)
  -> metadata / feature order (`models/model_meta.json`)
  -> expected raw engagement
```

The deployed app uses NumPy/Pandas/Streamlit only. It does not require scikit-learn at inference time.

## Training path

```text
data/raw/*.csv
  -> cleaning + feature engineering
  -> train/test split
  -> Random Forest training
  -> evaluation
  -> export model.npz + model_meta.json
```

Production training is in `training/train_model.py`. The larger experimental workflow is kept separately in `training/experimental_pipeline.py`.
