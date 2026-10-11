"""CLI for classifier-trainer: train, predict, demo, compare."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from classifiers import (
    GaussianNB,
    KNNClassifier,
    LogisticRegression,
    classification_report,
    compare_models,
    cross_val_accuracy,
    load_model,
    make_factories,
    save_model,
    stratified_split,
)

MODELS = {
    "logistic": lambda: LogisticRegression(),
    "gnb": lambda: GaussianNB(),
    "knn": lambda: KNNClassifier(k=5),
}


def _read_csv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"'{path}' has no data rows")
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


def _extract(rows: list[dict], target: str, features: list[str]):
    X = [[float(r[f]) for f in features] for r in rows]
    y = [r[target] for r in rows]
    return np.asarray(X, dtype=float), np.asarray(y)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="classifier-trainer",
        description="Train and compare classification models from CSV data (numpy only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python cli.py train iris.csv -t species\n"
            "  python cli.py train iris.csv -t species -m knn --cv 5 --save model.json\n"
            "  python cli.py compare iris.csv -t species --cv 5\n"
            "  python cli.py predict model.json --data iris.csv -o out.csv\n"
            "  python cli.py demo -m gnb"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="fit one model and report metrics")
    t.add_argument("input", help="input CSV")
    t.add_argument("-t", "--target", required=True, help="target (class) column")
    t.add_argument("-f", "--features", action="append", help="feature column(s); default: all numeric except target")
    t.add_argument("-m", "--model", choices=list(MODELS), default="logistic", help="model (default logistic)")
    t.add_argument("--test-size", type=float, default=0.25)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--k", type=int, default=5, help="k for knn (default 5)")
    t.add_argument("--cv", type=int, help="k-fold cross-validation accuracy")
    t.add_argument("--save", help="save the fitted model to this JSON file")
    t.add_argument("-q", "--quiet", action="store_true")

    pr = sub.add_parser("predict", help="predict classes with a saved model")
    pr.add_argument("model", help="model JSON file")
    pr.add_argument("--data", required=True)
    pr.add_argument("-o", "--output", help="write predictions to CSV")
    pr.add_argument("--proba", action="store_true", help="include class probabilities")
    pr.add_argument("-q", "--quiet", action="store_true")

    d = sub.add_parser("demo", help="train on synthetic 3-class blobs")
    d.add_argument("-m", "--model", choices=list(MODELS), default="logistic")
    d.add_argument("--k", type=int, default=5, help="k for knn (default 5)")
    d.add_argument("--n", type=int, default=300)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--save", help="save the fitted model")
    d.add_argument("-q", "--quiet", action="store_true")

    c = sub.add_parser("compare", help="cross-validate and rank all models")
    c.add_argument("input")
    c.add_argument("-t", "--target", required=True)
    c.add_argument("-f", "--features", action="append")
    c.add_argument("--cv", type=int, default=5)
    c.add_argument("--seed", type=int, default=42)
    return p


def _make_model(args):
    if args.model == "knn":
        return KNNClassifier(k=args.k)
    if args.model == "gnb":
        return GaussianNB()
    return LogisticRegression()


def _print_report(report: dict, title: str = "") -> None:
    if title:
        print(title)
    print(f"Accuracy: {report['accuracy']:.4f}   (n={report['n']})")
    print(f"  {'class':<14}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for c in report["classes"]:
        print(f"  {c['label']:<14}{c['precision']:>8.4f}{c['recall']:>8.4f}{c['f1']:>8.4f}{c['support']:>9}")
    print(f"  {'macro avg':<14}{report['macro']['precision']:>8.4f}{report['macro']['recall']:>8.4f}{report['macro']['f1']:>8.4f}{report['macro']['support']:>9}")


def cmd_train(args, rows):
    features = args.features or [c for c in _numeric_cols(rows) if c != args.target]
    if not features:
        raise ValueError("no numeric feature columns found (use -f)")
    if args.target not in rows[0]:
        raise ValueError(f"target column '{args.target}' not found")
    X, y = _extract(rows, args.target, features)
    train_idx, test_idx = stratified_split(y, args.test_size, args.seed)
    model = _make_model(args).fit(X[train_idx], y[train_idx], feature_names=features)
    report = classification_report(y[test_idx], model.predict(X[test_idx]))

    if not args.quiet:
        print(f"Model:   {args.model}   Features: {', '.join(features)}")
        print(f"Split:   {len(train_idx)} train / {len(test_idx)} test (stratified)")
        _print_report(report, f"Held-out test ({len(test_idx)} rows):")
        if args.cv:
            cv = cross_val_accuracy(lambda: _make_model(args), X, y, k=args.cv, seed=args.seed)
            print(f"CV k={args.cv}: acc={cv['mean']:.4f} +/- {cv['std']:.4f}")
    if args.save:
        save_model(model, args.save)
        if not args.quiet:
            print(f"Saved model -> {args.save}")
    return report


def cmd_predict(args):
    model = load_model(args.model)
    rows = _read_csv(args.data)
    data = model.params()
    names = data.get("feature_names")
    nf = data["n_features"]
    if names:
        cols = names
    else:
        cols = _numeric_cols(rows)[:nf]
        if len(cols) != nf:
            raise ValueError(f"model expects {nf} feature(s), found {len(cols)}")
    X = np.asarray([[float(r[c]) for c in cols] for r in rows])
    preds = model.predict(X)
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = list(cols) + ["prediction"]
            proba = model.predict_proba(X) if args.proba else None
            if proba is not None:
                header += [f"p_{c}" for c in data["classes"]]
            w.writerow(header)
            for i, r in enumerate(rows):
                line = [r[c] for c in cols] + [preds[i]]
                if proba is not None:
                    line += [round(float(p), 4) for p in proba[i]]
                w.writerow(line)
        if not args.quiet:
            print(f"Wrote {len(preds)} predictions -> {args.output}")
    if not args.quiet:
        uniq, counts = np.unique(preds, return_counts=True)
        print("Prediction counts: " + ", ".join(f"{u}={n}" for u, n in zip(uniq, counts)))
    return 0


def cmd_demo(args):
    rng = np.random.default_rng(args.seed)
    centers = np.array([[0.0, 0.0], [4.0, 0.5], [2.0, 4.0]])
    labels = np.array(["red", "green", "blue"])
    per = args.n // 3
    X, y = [], []
    for c, cen in zip(labels, centers):
        pts = rng.normal(cen, 0.9, size=(per, 2))
        X.append(pts)
        y.extend([c] * per)
    X = np.vstack(X)
    y = np.array(y)
    model = _make_model(args).fit(X, y, feature_names=["x1", "x2"])
    report = classification_report(y, model.predict(X))
    if not args.quiet:
        print(f"Demo: {args.n} points, 3 Gaussian blobs, model={args.model}")
        _print_report(report, "Training-set report:")
    if args.save:
        save_model(model, args.save)
        if not args.quiet:
            print(f"Saved model -> {args.save}")
    return 0


def cmd_compare(args):
    rows = _read_csv(args.input)
    features = args.features or [c for c in _numeric_cols(rows) if c != args.target]
    X, y = _extract(rows, args.target, features)
    result = compare_models(make_factories(), X, y, k=args.cv, seed=args.seed)
    print(f"Cross-validated accuracy (k={args.cv}, stratified), {len(features)} feature(s):")
    for name in result["ranking"]:
        r = result["results"][name]
        print(f"  {name:<12} acc={r['mean']:.4f} +/- {r['std']:.4f}  folds={r['scores']}")
    print(f"Best: {result['ranking'][0]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            cmd_train(args, _read_csv(args.input))
        elif args.command == "predict":
            return cmd_predict(args)
        elif args.command == "demo":
            return cmd_demo(args)
        elif args.command == "compare":
            return cmd_compare(args)
    except (ValueError, OSError, KeyError, IndexError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except Exception as err:  # pragma: no cover - safety net
        print(f"error: unexpected failure: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())