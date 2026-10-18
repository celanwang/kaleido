"""Linear least squares and binary logistic regression without sklearn imports."""

import warnings

import numpy as np

from kaleido.classical._validation import (
    FittedFeatures,
    features,
    nonnegative_float,
    positive_int,
    target,
)
from kaleido.numerics import sigmoid


class LinearRegression(FittedFeatures):
    """Centered least squares; SVD/lstsq also handles rank-deficient designs."""

    def fit(self, x, y):
        x = features(x)
        y = target(y, len(x), numeric=True)
        center, target_center = x.mean(axis=0), y.mean()
        coef, _, rank, singular = np.linalg.lstsq(x - center, y - target_center, rcond=None)
        self.coef_, self.intercept_ = coef, float(target_center - center @ coef)
        self.rank_, self.singular_values_ = int(rank), singular
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, x):
        return self._predict_x(x) @ self.coef_ + self.intercept_


def logistic_objective(theta, design, y, l2):
    """Mean Bernoulli NLL + l2/2 ||w||²; final theta entry is unpenalized bias."""
    logits = design @ theta
    probability = sigmoid(logits)
    loss = np.mean(np.logaddexp(0, logits) - y * logits) + l2 / 2 * (theta[:-1] @ theta[:-1])
    gradient = design.T @ (probability - y) / len(y)
    gradient[:-1] += l2 * theta[:-1]
    curvature = probability * (1 - probability)
    hessian = design.T @ (curvature[:, None] * design) / len(y)
    hessian[:-1, :-1] += l2 * np.eye(len(theta) - 1)
    return float(loss), gradient, hessian


class LogisticRegression(FittedFeatures):
    """Binary Newton solver with Armijo backtracking and an explicit objective.

    sklearn equivalence for n unweighted samples: C=1/(n*l2). For l2=0 use C=inf.
    Not a multiclass solver; unregularized separable data may have no finite optimum.
    """

    def __init__(self, *, l2=0.01, max_iter=100, tol=1e-8):
        nonnegative_float(l2, "l2")
        positive_int(max_iter, "max_iter")
        nonnegative_float(tol, "tol", positive=True)
        self.l2, self.max_iter, self.tol = l2, max_iter, tol

    def fit(self, x, y):
        x = features(x)
        y = target(y, len(x))
        classes, encoded = np.unique(y, return_inverse=True)
        if len(classes) != 2:
            raise ValueError("logistic regression requires exactly two classes")
        design = np.column_stack([x, np.ones(len(x))])
        theta = np.zeros(design.shape[1])
        history, converged, updates = [], False, 0
        for iteration in range(self.max_iter + 1):
            loss, gradient, hessian = logistic_objective(theta, design, encoded, self.l2)
            history.append(loss)
            if np.max(np.abs(gradient)) <= self.tol:
                converged = True
                break
            if iteration == self.max_iter:
                break
            direction = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            slope = float(gradient @ direction)
            if slope <= 0:  # Degenerate curvature: retain a descent direction.
                direction, slope = gradient, float(gradient @ gradient)
            step = 1.0
            for _ in range(60):
                candidate = theta - step * direction
                candidate_loss = logistic_objective(candidate, design, encoded, self.l2)[0]
                if np.isfinite(candidate_loss) and candidate_loss <= loss - 1e-4 * step * slope:
                    theta = candidate
                    updates += 1
                    break
                step *= 0.5
            else:
                break
        self.coef_, self.intercept_ = theta[:-1], float(theta[-1])
        self.classes_, self.n_features_in_ = classes, x.shape[1]
        self.loss_history_, self.converged_, self.n_iter_ = history, converged, updates
        self.gradient_max_abs_ = float(np.max(np.abs(gradient)))
        if not converged:
            warnings.warn(
                "logistic solver did not reach gradient tolerance", RuntimeWarning, stacklevel=2
            )
        return self

    def decision_function(self, x):
        return self._predict_x(x) @ self.coef_ + self.intercept_

    def predict_proba(self, x):
        positive = sigmoid(self.decision_function(x))
        return np.column_stack([1 - positive, positive])

    def predict(self, x):
        scores = self.decision_function(x)
        return self.classes_[(scores > 0).astype(int)]
