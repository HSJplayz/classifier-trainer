# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-10-11

### Added
- `classifiers.py` — three from-scratch classifiers (numpy only):
  - `LogisticRegression` — multiclass softmax trained with gradient descent, internal feature standardization, L2 option, convergence tracking.
  - `GaussianNB` — per-class Gaussian likelihoods with variance smoothing and class priors.
  - `KNNClassifier` — k-nearest neighbours with uniform or distance weighting.
  - Evaluation: `accuracy_score`, `confusion_matrix`, `classification_report` (per-class precision/recall/F1/support + macro and weighted averages).
  - `stratified_split` and stratified `cross_val_accuracy`; `compare_models` to rank several models by CV accuracy.
  - JSON persistence for all three models (`save_model` / `load_model`) including feature names.
- `cli.py` — `train` / `predict` / `demo` / `compare` subcommands, feature selection, `--cv`, model save/load, exit codes (0 ok, 1 error).
- `app.py` + `templates/index.html` — Flask UI: paste CSV, choose model, train; renders a confusion-matrix heatmap, per-class metrics, a decision-boundary grid with test points, and a cross-validated model comparison. All charts are inline SVG, no CDN.
- 31 unit tests (`test_classifiers.py`) covering metric math, every classifier, persistence round-trips, stratified splitting, CV, CLI exit codes, and Flask-client API paths.
- GitHub Actions CI (Python 3.11 / 3.13 matrix).

[1.0.0]: https://github.com/HSJplayz/classifier-trainer/releases/tag/v1.0.0