
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegressionCustom(ClassifierMixin, BaseEstimator):

    def __init__(self, learning_rate=0.1, n_iterations=2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("Only binary classification is supported.")

        y_binary = (y == self.classes_[1]).astype(float)

        n_samples, n_features = X.shape

        # Standardize input features
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0

        X_scaled = (X - self.mean_) / self.scale_

        # Initialize parameters
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0

        # Gradient descent
        for _ in range(self.n_iterations):
            linear_model = X_scaled @ self.weights_ + self.bias_
            predictions = self.sigmoid(linear_model)

            errors = predictions - y_binary

            dw = (X_scaled.T @ errors) / n_samples
            db = np.mean(errors)

            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        X_scaled = (X - self.mean_) / self.scale_

        linear_model = X_scaled @ self.weights_ + self.bias_
        probabilities = self.sigmoid(linear_model)

        return np.column_stack([
            1 - probabilities,
            probabilities,
        ])

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        return self.classes_[predictions]
