# classifier-trainer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![CI](https://github.com/HSJplayz/classifier-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/HSJplayz/classifier-trainer/actions/workflows/ci.yml)
[![Streak Day 17](https://img.shields.io/badge/Streak-Day%2017-green.svg)](https://github.com/HSJplayz/streak)

Three classifiers **from scratch** — logistic regression, Gaussian naive Bayes,
and k-nearest neighbours — with confusion matrices, precision/recall/F1,
stratified cross-validation and model comparison. A CLI and a web UI with a
decision-boundary plot. Only numpy (plus Flask for the web app).

Built with numpy on **Day 17** of the daily projects streak.

## Quick start

```bash
pip install -r requirements.txt

# Train one model, held-out report + optional CV
python cli.py train iris.csv -t species -m logistic --cv 5

# Save and reuse a model
python cli.py train iris.csv -t species -m knn --k 5 --save model.json
python cli.py predict model.json --data iris.csv -o predictions.csv --proba

# Cross-validate and rank all three models
python cli.py compare iris.csv -t species --cv 5

# Self-contained demo (synthetic 3-class blobs)
python cli.py demo -m gnb

# Web UI
python app.py   # then open http://127.0.0.1:5000
```

## Models

| Model | Notes |
| --- | --- |
| `logistic` | multiclass softmax, gradient descent, auto-standardized features, optional L2 |
| `gnb` | Gaussian naive Bayes — per-class mean/variance + priors, variance smoothing |
| `knn` | k-nearest neighbours, uniform or distance weighting, configurable `k` |

## Metrics

`classification_report` returns accuracy, a confusion matrix, and per-class
precision / recall / F1 / support plus **macro** and **weighted** averages —
the same shape you would expect from a real ML library.

```
Model:   knn   Features: sepal_len, sepal_wid, petal_len, petal_wid
Split:   112 train / 38 test (stratified)
Held-out test (38 rows):
Accuracy: 0.9474   (n=38)
  class             prec  recall      f1  support
  setosa          1.0000  1.0000  1.0000       13
  versicolor      0.9231  0.9231  0.9231       13
  virginica       0.9167  0.9167  0.9167       12
  macro avg       0.9466  0.9466  0.9466       38
```

## Library use

```python
import numpy as np
from classifiers import GaussianNB, LogisticRegression, classification_report, compare_models

X, y = ...  # features (n, d), labels (n,)
model = LogisticRegression().fit(X, y, feature_names=["a", "b"])
print(classification_report(y, model.predict(X)))

ranking = compare_models({
    "logistic": lambda: LogisticRegression(),
    "gnb": lambda: GaussianNB(),
}, X, y, k=5)["ranking"]
```

## Methods, honestly

- **Logistic regression** standardizes features internally (raw scales stall
  gradient descent) and reports `converged` so a bad learning rate is visible,
  not silent. Softmax handles any number of classes.
- **Gaussian naive Bayes** assumes conditional feature independence — fast and
  strong on blobs, weaker when features are correlated. Variance smoothing
  avoids zero-variance blow-ups.
- **k-NN** stores the training set (models serialize their data) and predicts by
  neighbour vote; it is sensitive to feature scaling and the choice of `k`.
- The web decision boundary is a coarse grid (48×32) of model predictions, so
  edges look slightly stair-stepped by design.

## Project layout

```
classifiers.py      models, metrics, splitting, CV, comparison, persistence
cli.py              train / predict / demo / compare commands
app.py              Flask JSON API (/health, /api/train)
templates/index.html self-contained UI: confusion heatmap + decision boundary
test_classifiers.py 31 unittest cases
```

## Future scope

- One-vs-rest SVM and a small decision tree
- Feature scaling report + class-balance warnings
- ROC/AUC for binary problems and a precision-recall curve
- k selection sweep and learning curves
- CSV/Excel input helpers and a downloadable classification report

## Tests

```bash
python -m unittest test_classifiers -v
```

31 tests: metric math on hand-computed examples, each classifier on separable
blobs, persistence round-trips, stratified splits, CV, CLI exit codes, and
Flask-client API tests. No network.

## License

MIT — see [LICENSE](LICENSE).