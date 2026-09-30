# Streamlit deployment

## Streamlit Community Cloud

1. Push the production-ready repository to GitHub.
2. Open Streamlit Community Cloud and sign in with GitHub.
3. Create a new app.
4. Select this repository.
5. Select the production branch: `master`.
6. Set the main file path to:

```text
app/streamlit_app.py
```

7. Deploy.

## Runtime files

The deployed branch needs:

```text
app/streamlit_app.py
models/model.npz
models/model_meta.json
requirements.txt
.streamlit/config.toml
```

The raw datasets, notebook, and training-only dependencies are not required at runtime.

## Release flow

```text
feature/* -> develop -> release/* -> master -> Streamlit
```

Before merging a release into `master`, verify:

- the app starts locally;
- a representative prediction completes;
- the UI renders at desktop and mobile widths;
- model artifacts load successfully;
- CI passes.

If the model is retrained, commit `models/model.npz` and `models/model_meta.json` together.
