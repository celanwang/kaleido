"""Small NumPy estimators with fit/predict and fit/transform interfaces."""

from kaleido.classical.linear import LinearRegression, LogisticRegression
from kaleido.classical.preprocessing import StandardScaler

__all__ = ["LinearRegression", "LogisticRegression", "StandardScaler"]
