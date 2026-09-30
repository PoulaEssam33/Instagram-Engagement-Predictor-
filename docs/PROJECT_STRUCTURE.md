# Project structure

```text
instagram-engagement-predictor/
├── app/                         # Deployable Streamlit application
│   └── streamlit_app.py
├── models/                      # Versioned runtime model artifacts
│   ├── model.npz
│   ├── model_meta.json
│   └── README.md
├── training/                    # Training and experimentation code
│   ├── train_model.py
│   ├── experimental_pipeline.py
│   └── README.md
├── data/
│   ├── raw/                     # Source datasets
│   │   ├── Grammy_IG_posts_v2.csv
│   │   └── Instagram_Analytics.csv
│   └── README.md
├── notebooks/                   # Exploratory analysis
│   ├── Instagram_Engagement_prediction.ipynb
│   └── README.md
├── tests/                       # Lightweight repository/model checks
├── docs/                        # Architecture, deployment and workflow docs
├── .github/                     # CI and PR template
├── .streamlit/                  # Streamlit configuration
├── .devcontainer/               # Optional dev-container config
├── requirements.txt             # Runtime dependencies
├── requirements-train.txt       # Training/notebook dependencies
├── requirements-dev.txt         # CI/test dependencies
├── CONTRIBUTING.md
├── .gitignore
└── README.md
```

The root intentionally contains only repository-level configuration and documentation. Application code, model artifacts, data, notebooks, and training code each have one clear home.
