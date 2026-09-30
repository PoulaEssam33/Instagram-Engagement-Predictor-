# Model card

## Purpose

Estimate the expected **raw engagement** of a planned Instagram post from information available before publishing.

## Production model

- Algorithm: Random Forest regression
- Trees: 150
- Random state: 42
- Maximum depth: 12
- Minimum samples per leaf: 2
- Maximum features per split: 0.8

## Evaluation snapshot

From the current `models/model_meta.json` artifact:

| Metric | Value |
| --- | ---: |
| Test R² | 0.8671 |
| Test RMSE | 181,809 |
| Test MAE | 29,838 |
| Training rows | 51,848 |
| Test rows | 12,962 |

## Inputs

The production feature vector contains account, content, and timing signals, including follower count, previous engagement, media type, caption length, hashtag count, weekday, weekend/holiday status, and posting-time bucket.

## Output

The app returns an estimate of raw engagement. It should be interpreted as a planning signal rather than a guarantee of future post performance.

## Limitations

- The model can only learn patterns present in the training datasets.
- Performance can differ across accounts, audiences, markets, and time periods not represented by the training data.
- Platform behavior and recommendation systems can change after training.
- The current holiday logic uses U.S. federal holidays.
- Evaluation metrics summarize performance on the held-out test split and do not guarantee identical performance on new accounts.

## Reproducibility

Use `training/train_model.py` to regenerate the production model and metadata from `data/raw/`.
