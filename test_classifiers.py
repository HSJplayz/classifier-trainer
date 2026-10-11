"""Tests for classifier-trainer (stdlib unittest, numpy + Flask only)."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from classifiers import (
    GaussianNB,
    KNNClassifier,
    LogisticRegression,
    accuracy_score,
    classification_report,
    compare_models,
    confusion_matrix,
    cross_val_accuracy,
    load_model,
    make_factories,
    save_model,
    stratified_split,
)
from cli import build_parser, main


def blobs(n_per=40, seed=1, k=3):
    rng = np.random.default_rng(seed)
    centers = [(0, 0), (4, 0.5), (2, 4)][:k]
    X, y = [], []
    for i, c in enumerate(centers):
        X.append(rng.normal(c, 0.7, size=(n_per, 2)))
        y.extend([f"c{i}"] * n_per)
    return np.vstack(X), np.array(y)


def write_csv(rows, header, tmpdir) -> str:
    path = Path(tmpdir) / "data.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return str(path)


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        self.assertAlmostEqual(accuracy_score([0, 1, 1, 0], [0, 1, 0, 0]), 0.75)

    def test_confusion_matrix(self):
        cm, labels = confusion_matrix([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0])
        self.assertEqual(list(labels), [0, 1, 2])
        self.assertEqual(cm.tolist(), [[1, 1, 0], [0, 2, 0], [1, 0, 1]])

    def test_report_precision_recall_f1(self):
        r = classification_report([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0])
        self.assertAlmostEqual(r["accuracy"], 0.6667, places=3)
        by = {c["label"]: c for c in r["classes"]}
        self.assertAlmostEqual(by["0"]["precision"], 0.5)
        self.assertAlmostEqual(by["1"]["f1"], 0.8)
        self.assertAlmostEqual(by["2"]["precision"], 1.0)
        self.assertAlmostEqual(by["2"]["recall"], 0.5)
        self.assertEqual(r["macro"]["support"], 6)

    def test_misaligned_raises(self):
        with self.assertRaises(ValueError):
            accuracy_score([1, 2], [1])


class TestSplit(unittest.TestCase):
    def test_stratified_proportions(self):
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)
        tr, te = stratified_split(y, 0.2, seed=1)
        self.assertEqual(len(te), 6)
        for c in (0, 1, 2):
            self.assertEqual(int((y[te] == c).sum()), 2)
            self.assertEqual(int((y[tr] == c).sum()), 8)

    def test_deterministic(self):
        y = np.array([0] * 8 + [1] * 8)
        a = stratified_split(y, 0.25, seed=3)
        b = stratified_split(y, 0.25, seed=3)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


class TestLogistic(unittest.TestCase):
    def test_separable_binary(self):
        X = np.array([[0], [1], [2], [10], [11], [12]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        model = LogisticRegression().fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_multiclass_blobs(self):
        X, y = blobs()
        model = LogisticRegression().fit(X, y)
        self.assertGreater(accuracy_score(y, model.predict(X)), 0.9)

    def test_proba_rows_sum_to_one(self):
        X, y = blobs()
        P = LogisticRegression().fit(X, y).predict_proba(X)
        np.testing.assert_allclose(P.sum(axis=1), np.ones(len(X)), atol=1e-9)

    def test_unfitted_raises(self):
        with self.assertRaises(RuntimeError):
            LogisticRegression().predict([[1, 2]])


class TestGaussianNB(unittest.TestCase):
    def test_blobs_accuracy(self):
        X, y = blobs()
        model = GaussianNB().fit(X, y)
        self.assertGreater(accuracy_score(y, model.predict(X)), 0.9)

    def test_proba_normalized(self):
        X, y = blobs()
        P = GaussianNB().fit(X, y).predict_proba(X)
        np.testing.assert_allclose(P.sum(axis=1), np.ones(len(X)), atol=1e-9)


class TestKNN(unittest.TestCase):
    def test_blobs_accuracy(self):
        X, y = blobs()
        model = KNNClassifier(k=5).fit(X, y)
        self.assertGreater(accuracy_score(y, model.predict(X)), 0.9)

    def test_k_too_large(self):
        with self.assertRaises(ValueError):
            KNNClassifier(k=10).fit([[0, 0], [1, 1]], [0, 1])

    def test_distance_weights(self):
        X, y = blobs()
        model = KNNClassifier(k=5, weights="distance").fit(X, y)
        self.assertGreater(accuracy_score(y, model.predict(X)), 0.9)


class TestPersistence(unittest.TestCase):
    def test_round_trip_all_models(self):
        X, y = blobs()
        for name, factory in make_factories().items():
            model = factory().fit(X, y, feature_names=["a", "b"])
            with tempfile.TemporaryDirectory() as d:
                path = str(Path(d) / "m.json")
                save_model(model, path)
                loaded = load_model(path)
                np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
                self.assertEqual(loaded.feature_names, ["a", "b"])

    def test_unknown_model_type(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.json"
            path.write_text('{"_model": "Nope"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model(str(path))


class TestCV(unittest.TestCase):
    def test_cross_val(self):
        X, y = blobs()
        res = cross_val_accuracy(lambda: KNNClassifier(k=5), X, y, k=5, seed=1)
        self.assertEqual(len(res["scores"]), 5)
        self.assertGreater(res["mean"], 0.85)

    def test_compare_ranking(self):
        X, y = blobs()
        res = compare_models(make_factories(), X, y, k=5, seed=1)
        self.assertEqual(set(res["results"]), {"logistic", "gaussian_nb", "knn"})
        means = [res["results"][n]["mean"] for n in res["ranking"]]
        self.assertEqual(means, sorted(means, reverse=True))


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        X, y = blobs()
        rows = [[round(float(a), 3), round(float(b), 3), c] for (a, b), c in zip(X, y)]
        self.csv = write_csv(rows, ["a", "b", "label"], self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_models(self):
        for m in ("logistic", "gnb", "knn"):
            self.assertEqual(main(["train", self.csv, "-t", "label", "-m", m, "-q"]), 0)

    def test_train_saves_and_predicts(self):
        path = str(Path(self.tmp.name) / "m.json")
        out = str(Path(self.tmp.name) / "pred.csv")
        self.assertEqual(main(["train", self.csv, "-t", "label", "-m", "gnb", "--save", path, "-q"]), 0)
        self.assertEqual(main(["predict", path, "--data", self.csv, "-o", out, "--proba", "-q"]), 0)
        data = list(csv.DictReader(open(out, encoding="utf-8")))
        self.assertIn("prediction", data[0])
        self.assertTrue(any(k.startswith("p_") for k in data[0]))

    def test_train_cv(self):
        self.assertEqual(main(["train", self.csv, "-t", "label", "--cv", "4", "-q"]), 0)

    def test_compare(self):
        self.assertEqual(main(["compare", self.csv, "-t", "label", "--cv", "3"]), 0)

    def test_demo(self):
        for m in ("logistic", "gnb", "knn"):
            self.assertEqual(main(["demo", "-m", m, "-q"]), 0)

    def test_missing_file(self):
        self.assertEqual(main(["train", "nope.csv", "-t", "label", "-q"]), 1)

    def test_bad_target(self):
        self.assertEqual(main(["train", self.csv, "-t", "nope", "-q"]), 1)


class TestWeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app

        cls.client = app.test_client()

    @staticmethod
    def _csv():
        X, y = blobs(n_per=20)
        lines = ["a,b,label"]
        for (a, b), c in zip(X, y):
            lines.append(f"{a:.3f},{b:.3f},{c}")
        return "\n".join(lines) + "\n"

    def test_health(self):
        self.assertEqual(self.client.get("/health").get_json()["ok"], True)

    def test_index(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("classifier-trainer", r.get_data(as_text=True))

    def test_train_two_features_has_boundary(self):
        payload = {"csv": self._csv(), "target": "label", "features": ["a", "b"], "model": "logistic"}
        r = self.client.post("/api/train", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue(data["ok"])
        self.assertGreater(data["metrics_test"]["accuracy"], 0.8)
        self.assertIn("boundary", data)
        self.assertEqual(len(data["boundary"]["grid"]), data["boundary"]["ny"])
        self.assertIn("points", data)

    def test_compare_flag(self):
        payload = {"csv": self._csv(), "target": "label", "features": ["a", "b"], "compare": True, "cv": 3}
        data = self.client.post("/api/train", json=payload).get_json()
        self.assertIn("compare", data)
        self.assertEqual(set(data["compare"]), {"logistic", "gaussian_nb", "knn"})

    def test_bad_target(self):
        r = self.client.post("/api/train", json={"csv": self._csv(), "target": "zzz", "features": ["a"]})
        self.assertEqual(r.status_code, 400)
        self.assertFalse(r.get_json()["ok"])


if __name__ == "__main__":
    unittest.main()