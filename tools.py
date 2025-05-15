from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X, y=None):
        for col in self.cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds[col] = Q1 - 1.5 * IQR
            self.upper_bounds[col] = Q3 + 1.5 * IQR
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.cols:
            X[col] = np.clip(X[col], self.lower_bounds[col], self.upper_bounds[col])
        return X