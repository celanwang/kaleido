import numpy as np

from kaleido.classical._validation import FittedFeatures, features


class StandardScaler(FittedFeatures):
    """Population variance (ddof=0); constant features get scale one."""

    def fit(self, x):
        x = features(x)
        with np.errstate(over="raise", invalid="raise"):
            center, variance = x.mean(axis=0), x.var(axis=0)
        self.mean_, self.var_ = center, variance
        self.scale_ = np.where(self.var_ == 0, 1, np.sqrt(self.var_))
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x):
        return (self._predict_x(x) - self.mean_) / self.scale_

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        return self._predict_x(x) * self.scale_ + self.mean_
