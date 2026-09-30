# Instagram Engagement Predictor

A machine-learning project that estimates the expected **raw engagement** of a planned Instagram post before it is published. The Streamlit app combines account history with planned-post characteristics such as media type, caption length, hashtags, date, and publishing time, then returns a model estimate that can be compared with historical performance.

> **Project type:** Academic / portfolio ML project with a deployable Streamlit demo.

## Live demo

After deployment, place the Streamlit URL here:

```text
https://<your-app-name>.streamlit.app
```

## What it does

The predictor uses information available before publishing:

- current follower count;
- historical median engagement;
- number of previous posts;
- media type: image, carousel, or video/reel;
- image count;
- caption length;
- hashtag count;
- publication weekday;
- weekend/holiday status;
- publishing-time bucket.

The output is an **estimated raw engagement value**, not a guaranteed future result.

## Model snapshot

The production app uses a compact **Random Forest regressor** exported to NumPy arrays so Streamlit inference does not need scikit-learn.

| Metric | Current artifact |
| --- | ---: |
| Trees | 150 |
| Test R² | 0.867 |
| Test RMSE | 181,809 |
| Test MAE | 29,838 |
| Training rows | 51,848 |
| Test rows | 12,962 |

The values above come from `models/model_meta.json` and should be updated after retraining.

## Prediction flow

```text
Account history + planned post
              ↓
      Feature engineering
              ↓
   Model-ready feature vector
              ↓
      Random Forest trees
              ↓
       Average tree output
              ↓
   Expected raw engagement
```

## Repository layout

```text
instagram-engagement-predictor/
├── app/                         # Streamlit website
├── models/                      # Runtime model artifacts
├── training/                    # Production training + experiments
├── data/raw/                    # Source datasets
├── notebooks/                   # Exploratory notebook
├── tests/                       # Artifact/layout checks
├── docs/                        # Architecture/deployment/workflow docs
├── .github/                     # CI + PR template
├── .streamlit/                  # Streamlit config
├── .devcontainer/               # Optional dev environment
├── requirements.txt
├── requirements-train.txt
├── requirements-dev.txt
├── CONTRIBUTING.md
└── README.md
```

See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) for the complete file map and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the runtime/training flow.

## Run locally

### 1. Clone

```bash
git clone https://github.com/<your-username>/instagram-engagement-predictor.git
cd instagram-engagement-predictor
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install runtime dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start Streamlit

```bash
streamlit run app/streamlit_app.py
```

The app normally opens at `http://localhost:8501`.

## Retrain the production model

```bash
pip install -r requirements-train.txt
python training/train_model.py
```

Training regenerates:

```text
models/model.npz
models/model_meta.json
```

Commit both artifacts together so model parameters and feature metadata remain synchronized.

## Experimentation

The original exploratory notebook is preserved at:

```text
notebooks/Instagram_Engagement_prediction.ipynb
```

The larger experimental pipeline is kept separately at:

```text
training/experimental_pipeline.py
```

This separation keeps experimental work out of the production runtime path.

## Deployment

For Streamlit Community Cloud, use:

```text
Branch: master
Main file: app/streamlit_app.py
```

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the deployment checklist and [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for model scope, metrics, and limitations.

## Branching workflow

```text
feature/* -> develop -> release/* -> master
                  ^                    |
                  |------ hotfix/* ----|
```

- `master` contains tested production-ready code.
- `develop` is the persistent integration branch and the source for new feature work.
- `feature/*` branches are short-lived and merge into `develop` through pull requests.
- `release/*` branches are created from `develop` for release testing.
- `bugfix/*` branches address issues found during release testing.
- `hotfix/*` branches start from `master` and merge back into both `master` and `develop`.

See [`docs/BRANCHING.md`](docs/BRANCHING.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

## CI

GitHub Actions validates pushes and pull requests to `develop` and `master` by:

1. installing runtime/test dependencies;
2. compiling the application and training scripts;
3. validating the expected project layout;
4. checking that model metadata matches the exported Random Forest arrays.

## Data

The requested project package includes both original CSV datasets under `data/raw/`.

Before publishing the repository publicly, verify that the original dataset licenses/terms permit redistribution. See [`data/README.md`](data/README.md).

## Team

The Streamlit website contains the project team section with LinkedIn profiles and GitHub links where available.
