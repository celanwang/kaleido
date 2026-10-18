"""Deterministic exact CART classification splits on dense numeric features."""

from dataclasses import dataclass

import numpy as np

from kaleido.classical._validation import (
    FittedFeatures,
    features,
    nonnegative_float,
    positive_int,
    target,
)


@dataclass
class _Node:
    probabilities: np.ndarray
    samples: int
    impurity: float
    feature: int = -1
    threshold: float = 0.0
    left: object = None
    right: object = None


class DecisionTreeClassifier(FittedFeatures):
    """Gini CART, no pruning or missing values; ties favor the first feature/split.

    Zero-gain splits are allowed (unless min_impurity_decrease > 0) so XOR is not
    incorrectly declared unsplittable. Depth is capped at 64 for bounded recursion.
    """

    def __init__(self, *, max_depth=5, min_samples_leaf=1, min_impurity_decrease=0.0):
        positive_int(max_depth, "max_depth")
        if max_depth > 64:
            raise ValueError("max_depth must not exceed 64")
        positive_int(min_samples_leaf, "min_samples_leaf")
        nonnegative_float(min_impurity_decrease, "min_impurity_decrease")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, x, y):
        x = features(x)
        y = target(y, len(x))
        classes, encoded = np.unique(y, return_inverse=True)
        n_classes, total = len(classes), len(x)
        importance = np.zeros(x.shape[1])
        leaves, deepest = 0, 0

        def grow(rows, depth):
            nonlocal leaves, deepest
            counts = np.bincount(encoded[rows], minlength=n_classes)
            probabilities = counts / len(rows)
            impurity = float(1 - probabilities @ probabilities)
            node = _Node(probabilities, len(rows), impurity)
            deepest = max(deepest, depth)
            if depth >= self.max_depth or impurity == 0 or len(rows) < 2 * self.min_samples_leaf:
                leaves += 1
                return node
            best_risk, best_split = np.inf, None
            for feature in range(x.shape[1]):
                order = rows[np.argsort(x[rows, feature], kind="stable")]
                values = x[order, feature]
                left = np.cumsum(np.eye(n_classes)[encoded[order]], axis=0)[:-1]
                sizes = np.arange(1, len(rows))
                right = counts - left
                risk = (
                    sizes
                    - (left**2).sum(axis=1) / sizes
                    + len(rows)
                    - sizes
                    - (right**2).sum(axis=1) / (len(rows) - sizes)
                ) / len(rows)
                valid = (
                    (values[:-1] < values[1:])
                    & (sizes >= self.min_samples_leaf)
                    & (len(rows) - sizes >= self.min_samples_leaf)
                )
                risk[~valid] = np.inf
                position = int(np.argmin(risk))
                if risk[position] < best_risk - 1e-15:
                    best_risk = float(risk[position])
                    best_split = feature, position + 1, order
            gain = len(rows) / total * (impurity - best_risk)
            if best_split is None or gain + 1e-15 < self.min_impurity_decrease:
                leaves += 1
                return node
            feature, cut, order = best_split
            low, high = x[order[cut - 1], feature], x[order[cut], feature]
            threshold = low / 2 + high / 2
            if threshold >= high:
                threshold = low
            node.feature, node.threshold = feature, float(threshold)
            importance[feature] += max(gain, 0)
            node.left = grow(order[:cut], depth + 1)
            node.right = grow(order[cut:], depth + 1)
            return node

        root = grow(np.arange(total), 0)
        self.root_, self.classes_, self.n_features_in_ = root, classes, x.shape[1]
        self.n_leaves_, self.depth_ = leaves, deepest
        self.feature_importances_ = (
            importance / importance.sum() if importance.sum() else importance
        )
        return self

    def predict_proba(self, x):
        x = self._predict_x(x)
        result = np.empty((len(x), len(self.classes_)))
        for index, row in enumerate(x):
            node = self.root_
            while node.feature >= 0:
                node = node.left if row[node.feature] <= node.threshold else node.right
            result[index] = node.probabilities
        return result

    def predict(self, x):
        probability = self.predict_proba(x)
        return self.classes_[np.argmax(probability, axis=1)]
