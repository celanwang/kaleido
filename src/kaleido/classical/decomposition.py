import numpy as np

from kaleido.classical._validation import FittedFeatures, features, positive_int


class PCA(FittedFeatures):
    """Centered SVD, sample-covariance eigenvalues, no whitening."""

    def __init__(self, n_components):
        positive_int(n_components, "n_components")
        self.n_components = n_components

    def fit(self, x):
        x = features(x, min_samples=2)
        if self.n_components > min(x.shape):
            raise ValueError("n_components must not exceed min(n_samples, n_features)")
        center = x.mean(axis=0)
        _, singular, vectors = np.linalg.svd(x - center, full_matrices=False)
        signs = np.sign(vectors[np.arange(len(vectors)), np.argmax(np.abs(vectors), axis=1)])
        vectors *= np.where(signs == 0, 1, signs)[:, None]
        variances = singular**2 / (len(x) - 1)
        self.components_, self.mean_ = vectors[: self.n_components], center
        self.singular_values_ = singular[: self.n_components]
        self.explained_variance_ = variances[: self.n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / variances.sum()
            if variances.sum()
            else np.zeros(self.n_components)
        )
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x):
        return (self._predict_x(x) - self.mean_) @ self.components_.T

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, coordinates):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("estimator must be fitted first")
        coordinates = features(coordinates)
        if coordinates.shape[1] != self.n_components:
            raise ValueError("coordinate count differs from n_components")
        return coordinates @ self.components_ + self.mean_
