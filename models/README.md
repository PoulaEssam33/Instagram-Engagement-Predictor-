# Model artifacts

Runtime model files used by the deployed Streamlit application:

- `model.npz` — compact Random Forest tree arrays.
- `model_meta.json` — feature schema, category values, defaults, and evaluation metadata.

Treat these two files as one versioned deployment artifact. If the model is retrained, update and commit both together.
