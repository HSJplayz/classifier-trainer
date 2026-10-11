"""Web front-end for classifier-trainer (Flask). Trains classifiers on pasted CSV
and returns chart-ready JSON: confusion matrix, per-class metrics, and (for two
features) a decision-boundary grid. The UI renders everything as inline SVG."""

from __future__ import annotations

import csv
import io

import numpy as np
from flask import Flask, jsonify, render_template, request

from classifiers import (
    GaussianNB,
    KNNClassifier,
    LogisticRegression,
    classification_report,
    compare_models,
    cross_val_accuracy,
    make_factories,
    stratified_split,
)

app = Flask(__name__)

MODELS = {"logistic": LogisticRegression, "gnb": GaussianNB, "knn": lambda: KNNClassifier(k=5)}


def _make(model: str, k: int):
    factory = MODELS.get(model, LogisticRegression)
    if model == "knn":
        return KNNClassifier(k=k)
    return factory()


def _parse_csv(text: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("CSV has no rows (or headers are missing)")
    return rows


def _numeric_cols(rows: list[dict]) -> list[str]:
    cols = []
    for name in rows[0].keys():
        try:
            [float(r[name]) for r in rows]
            cols.append(name)
        except (TypeError, ValueError):
            continue
    return cols


def _boundary(model, X, classes, nx=48, ny=32) -> dict:
    xmin, xmax = float(X[:, 0].min()), float(X[:, 0].max())
    ymin, ymax = float(X[:, 1].min()), float(X[:, 1].max())
    px, py = (xmax - xmin) * 0.08 or 0.5, (ymax - ymin) * 0.08 or 0.5
    xmin, xmax, ymin, ymax = xmin - px, xmax + px, ymin - py, ymax + py
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    grid = []
    for yv in gy:
        row = []
        XX = np.column_stack([gx, np.full(nx, yv)])
        preds = model.predict(XX)
        index = {str(c): i for i, c in enumerate(classes)}
        row = [index[str(p)] for p in preds]
        grid.append(row)
    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
            "nx": nx, "ny": ny, "grid": grid}


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/train")
def api_train():
    try:
        body = request.get_json(force=True, silent=True) or {}
        rows = _parse_csv(body.get("csv", ""))
        target = body.get("target")
        features = body.get("features") or []
        model_name = body.get("model", "logistic")
        test_size = float(body.get("test_size", 0.25))
        seed = int(body.get("seed", 42))
        k = int(body.get("k", 5))

        if target not in rows[0]:
            raise ValueError(f"target column '{target}' not found in {list(rows[0].keys())}")
        if not features:
            features = [c for c in _numeric_cols(rows) if c != target]
        if not features:
            raise ValueError("no numeric feature columns found")
        missing = [f for f in features if f not in rows[0]]
        if missing:
            raise ValueError(f"feature column(s) missing: {missing}")
        if len(rows) < 10:
            raise ValueError("need at least 10 rows to train")

        X = np.asarray([[float(r[f]) for f in features] for r in rows])
        y = np.asarray([r[target] for r in rows])
        if len(np.unique(y)) < 2:
            raise ValueError("target needs at least two classes")

        train_idx, test_idx = stratified_split(y, test_size, seed)
        model = _make(model_name, k).fit(X[train_idx], y[train_idx], feature_names=features)
        report_test = classification_report(y[test_idx], model.predict(X[test_idx]))
        report_train = classification_report(y[train_idx], model.predict(X[train_idx]))

        result = {
            "ok": True,
            "overview": {"rows": int(len(rows)), "features": features, "model": model_name,
                         "classes": [str(c) for c in model.classes_], "n_train": int(len(train_idx)),
                         "n_test": int(len(test_idx))},
            "metrics_train": report_train,
            "metrics_test": report_test,
        }
        if len(features) == 2:
            result["boundary"] = _boundary(model, X, model.classes_)
            result["points"] = {
                "x": [round(float(v), 4) for v in X[test_idx, 0]],
                "y": [round(float(v), 4) for v in X[test_idx, 1]],
                "pred": [report_test["labels"].index(str(p)) for p in model.predict(X[test_idx])],
                "true": [report_test["labels"].index(str(t)) for t in y[test_idx]],
            }
        if body.get("compare"):
            ccv = int(body.get("cv", 5))
            result["compare"] = {name: cross_val_accuracy(f, X, y, k=ccv, seed=seed)
                                 for name, f in make_factories().items()}
        return jsonify(result)
    except Exception as err:
        return jsonify({"ok": False, "error": str(err)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)