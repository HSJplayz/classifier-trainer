"""From-scratch classifiers and evaluation: logistic regression, Gaussian naive
Bayes, and k-nearest neighbours, plus confusion matrix, precision/recall/F1,
stratified splitting, and cross-validated model comparison.

Depends only on numpy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EPS = 1e-12
DEFAULT_SEED = 42


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _encode(y):
    """Map arbitrary labels to 0..k-1 indices. Returns (indices, classes)."""
    y = np.asarray(y)
    classes = np.unique(y)
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index[v] for v in y], dtype=int), classes


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _logsumexp(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True))).ravel()


class LogisticRegression:
    """Multiclass logistic regression (softmax) trained with gradient descent.
    Features are standardized internally for stable convergence."""

    def __init__(self, *, lr: float = 0.5, max_iters: int = 2000, l2: float = 0.0,
                 tol: float = 1e-7, seed: int = DEFAULT_SEED, standardize: bool = True):
        self.lr = lr
        self.max_iters = max_iters
        self.l2 = l2
        self.tol = tol
        self.seed = seed
        self.standardize = standardize
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.scaler: dict | None = None
        self.converged: bool | None = None
        self.n_iter_: int = 0
        self.feature_names: list[str] | None = None

    def fit(self, X, y, *, feature_names=None) -> "LogisticRegression":
        X = _as_2d(X)
        y_idx, self.classes_ = _encode(y)
        self.feature_names = list(feature_names) if feature_names else None
        n, d = X.shape
        k = len(self.classes_)
        if n < 2 or k < 2:
            raise ValueError("need at least two classes and two samples")
        if self.standardize:
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std < EPS] = 1.0
            Xs = (X - mean) / std
            self.scaler = {"mean": mean.tolist(), "std": std.tolist()}
        else:
            Xs = X
            self.scaler = None

        Y = np.eye(k)[y_idx]
        rng = np.random.default_rng(self.seed)
        W = rng.standard_normal((d, k)) * 0.01
        b = np.zeros(k)
        prev = float("inf")
        for it in range(self.max_iters):
            P = _softmax(Xs @ W + b)
            error = P - Y
            gradW = (Xs.T @ error) / n + self.l2 * W
            gradb = error.mean(axis=0)
            W -= self.lr * gradW
            b -= self.lr * gradb
            with np.errstate(divide="ignore"):
                loss = float(-np.mean(np.log(np.clip(P[np.arange(n), y_idx], EPS, 1.0))))
            if not np.isfinite(loss):
                self.converged = False
                break
            if abs(prev - loss) <= self.tol:
                self.converged = True
                break
            prev = loss
        else:
            self.converged = False
        self.n_iter_ = it + 1
        self.weights = W
        self.bias = b
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("model has not been fitted")
        X = _as_2d(X)
        if self.scaler is not None:
            X = (X - np.asarray(self.scaler["mean"])) / np.asarray(self.scaler["std"])
        return _softmax(X @ self.weights + self.bias)

    def predict(self, X):
        idx = self.predict_proba(X).argmax(axis=1)
        return self.classes_[idx]

    def params(self) -> dict:
        return {
            "_model": "LogisticRegression",
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "classes": [str(c) for c in self.classes_],
            "scaler": self.scaler,
            "n_features": int(self.weights.shape[0]),
            "feature_names": self.feature_names,
            "converged": self.converged,
            "n_iter": self.n_iter_,
        }

    @classmethod
    def from_params(cls, p: dict) -> "LogisticRegression":
        model = cls()
        model.weights = np.asarray(p["weights"], dtype=float)
        model.bias = np.asarray(p["bias"], dtype=float)
        model.classes_ = np.asarray(p["classes"])
        model.scaler = p.get("scaler")
        model.feature_names = p.get("feature_names")
        model.converged = p.get("converged")
        model.n_iter_ = p.get("n_iter", 0)
        return model


class GaussianNB:
    """Gaussian naive Bayes: per-class feature means/variances with a smoothing
    term and class priors."""

    def __init__(self, *, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.prior_: np.ndarray | None = None
        self.feature_names: list[str] | None = None

    def fit(self, X, y, *, feature_names=None) -> "GaussianNB":
        X = _as_2d(X)
        y_idx, self.classes_ = _encode(y)
        self.feature_names = list(feature_names) if feature_names else None
        k = len(self.classes_)
        d = X.shape[1]
        self.mean_ = np.zeros((k, d))
        self.var_ = np.zeros((k, d))
        self.prior_ = np.zeros(k)
        eps = self.var_smoothing * X.var() if X.var() > EPS else self.var_smoothing
        for c in range(k):
            Xc = X[y_idx == c]
            if Xc.shape[0] == 0:
                raise ValueError(f"class {self.classes_[c]!r} has no samples")
            self.mean_[c] = Xc.mean(axis=0)
            self.var_[c] = Xc.var(axis=0) + eps
            self.prior_[c] = Xc.shape[0] / X.shape[0]
        return self

    def _log_likelihood(self, X) -> np.ndarray:
        # (n, k)
        ll = np.zeros((X.shape[0], len(self.classes_)))
        for c in range(len(self.classes_)):
            var = self.var_[c]
            ll[:, c] = (-0.5 * np.sum(np.log(2 * np.pi * var))
                        - 0.5 * np.sum((X - self.mean_[c]) ** 2 / var, axis=1))
        return ll + np.log(self.prior_)

    def predict_proba(self, X) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("model has not been fitted")
        X = _as_2d(X)
        ll = self._log_likelihood(X)
        return np.exp(ll - _logsumexp(ll)[:, None])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def params(self) -> dict:
        return {
            "_model": "GaussianNB",
            "classes": [str(c) for c in self.classes_],
            "mean": self.mean_.tolist(),
            "var": self.var_.tolist(),
            "prior": self.prior_.tolist(),
            "n_features": int(self.mean_.shape[1]),
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_params(cls, p: dict) -> "GaussianNB":
        model = cls()
        model.classes_ = np.asarray(p["classes"])
        model.mean_ = np.asarray(p["mean"], dtype=float)
        model.var_ = np.asarray(p["var"], dtype=float)
        model.prior_ = np.asarray(p["prior"], dtype=float)
        model.feature_names = p.get("feature_names")
        return model


class KNNClassifier:
    """k-nearest neighbours with optional distance weighting."""

    def __init__(self, *, k: int = 5, weights: str = "uniform"):
        if k < 1:
            raise ValueError("k must be >= 1")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.k = k
        self.weights = weights
        self.classes_: np.ndarray | None = None
        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None
        self.feature_names: list[str] | None = None

    def fit(self, X, y, *, feature_names=None) -> "KNNClassifier":
        X = _as_2d(X)
        self.y_, self.classes_ = _encode(y)
        self.X_ = X
        self.feature_names = list(feature_names) if feature_names else None
        if X.shape[0] < self.k:
            raise ValueError(f"k={self.k} exceeds number of samples ({X.shape[0]})")
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.X_ is None:
            raise RuntimeError("model has not been fitted")
        X = _as_2d(X)
        k = len(self.classes_)
        out = np.zeros((X.shape[0], k))
        for i, row in enumerate(X):
            dist = np.sqrt(((self.X_ - row) ** 2).sum(axis=1))
            nn = np.argsort(dist)[: self.k]
            if self.weights == "distance":
                w = 1.0 / (dist[nn] + 1e-9)
            else:
                w = np.ones(self.k)
            for idx, wi in zip(self.y_[nn], w):
                out[i, idx] += wi
            out[i] /= out[i].sum()
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def params(self) -> dict:
        return {
            "_model": "KNNClassifier",
            "k": self.k,
            "weights": self.weights,
            "classes": [str(c) for c in self.classes_],
            "X": self.X_.tolist(),
            "y": self.y_.tolist(),
            "n_features": int(self.X_.shape[1]),
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_params(cls, p: dict) -> "KNNClassifier":
        model = cls(k=p["k"], weights=p["weights"])
        model.classes_ = np.asarray(p["classes"])
        model.X_ = np.asarray(p["X"], dtype=float)
        model.y_ = np.asarray(p["y"], dtype=int)
        model.feature_names = p.get("feature_names")
        return model


# --- evaluation ---------------------------------------------------------------

def accuracy_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred must be equal-length non-empty arrays")
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred, labels=None) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    index = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    return cm, labels


def classification_report(y_true, y_pred, labels=None) -> dict:
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    y_true = np.asarray(y_true)
    support = cm.sum(axis=1)
    per_class = []
    for i, c in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append({
            "label": str(c),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support[i]),
        })
    total = int(support.sum())
    macro = {
        "precision": round(float(np.mean([p["precision"] for p in per_class])), 4),
        "recall": round(float(np.mean([p["recall"] for p in per_class])), 4),
        "f1": round(float(np.mean([p["f1"] for p in per_class])), 4),
        "support": total,
    }
    weighted = {
        k: round(float(np.average([p[k] for p in per_class], weights=support)), 4)
        for k in ("precision", "recall", "f1")
    }
    weighted["support"] = total
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "classes": per_class,
        "macro": macro,
        "weighted": weighted,
        "matrix": cm.tolist(),
        "labels": [str(c) for c in labels],
        "n": total,
    }


def stratified_split(y, test_size: float = 0.2, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Return boolean-free index arrays (train_idx, test_idx) split per class."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * (1.0 - test_size))))
        if cut >= len(idx):
            cut = len(idx) - 1
        train_idx.extend(idx[:cut].tolist())
        test_idx.extend(idx[cut:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def cross_val_accuracy(model_factory, X, y, k: int = 5, seed: int = DEFAULT_SEED) -> dict:
    """Stratified k-fold cross-validation returning accuracy per fold + mean/std."""
    X = _as_2d(X)
    y = np.asarray(y)
    if k < 2 or k > len(y):
        raise ValueError("k must satisfy 2 <= k <= n")
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(int(i))
    scores = []
    for j in range(k):
        test_idx = np.array(folds[j])
        train_idx = np.array([i for m, f in enumerate(folds) if m != j for i in f])
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
    arr = np.asarray(scores)
    return {"scores": [round(float(s), 4) for s in arr], "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4), "k": k}


def compare_models(factories: dict, X, y, k: int = 5, seed: int = DEFAULT_SEED) -> dict:
    out = {}
    for name, factory in factories.items():
        out[name] = cross_val_accuracy(factory, X, y, k=k, seed=seed)
    ranked = sorted(out.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    return {"results": out, "ranking": [name for name, _ in ranked]}


# --- persistence --------------------------------------------------------------

_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "GaussianNB": GaussianNB,
    "KNNClassifier": KNNClassifier,
}


def save_model(model, path) -> str:
    data = model.params()
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def load_model(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tag = data.get("_model")
    if tag not in _REGISTRY:
        raise ValueError(f"unknown model type: {tag!r}")
    return _REGISTRY[tag].from_params(data)


def make_factories() -> dict:
    return {
        "logistic": lambda: LogisticRegression(),
        "gaussian_nb": lambda: GaussianNB(),
        "knn": lambda: KNNClassifier(k=5),
    }