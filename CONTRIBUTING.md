# Contributing

## Development branch

Start normal development work from `develop`, not `master`.

```bash
git checkout develop
git pull origin develop
git checkout -b feature/<issue>-<short-description>
```

## Branch names

Use lowercase, hyphen-separated names:

```text
feature/42-redesign-predictor
bugfix/51-fix-date-input
hotfix/63-model-load-error
release/1.0.0
```

## Commits

Use concise, action-oriented commit messages, for example:

```text
feat: redesign predictor result card
fix: correct model artifact path
 docs: document Streamlit deployment
```

## Pull requests

A pull request should explain:

- What changed
- Why it changed
- How it was tested
- Screenshots for meaningful UI changes
- Any model or data artifacts that changed

Normal feature pull requests target `develop`.

## Before requesting review

```bash
python -m py_compile streamlit_app.py
python -m py_compile train_model.py
```

Then run the application locally and verify the primary prediction flow.
