
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegressionCustom(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        learning_rate=0.1,
        n_iterations=2000,
        tol=1e-6,
        n_iter_no_change=20,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, probabilities):
        epsilon = 1e-12
        probabilities = np.clip(
            probabilities,
            epsilon,
            1 - epsilon,
        )

        return -np.mean(
            y_true * np.log(probabilities)
            + (1 - y_true) * np.log(1 - probabilities)
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError(
                "Only binary classification is supported."
            )

        y_binary = (y == self.classes_[1]).astype(float)

        n_samples, n_features = X.shape

        # Standardisation des caractéristiques
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0

        X_scaled = (X - self.mean_) / self.scale_

        # Initialisation des paramètres
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0

        # Suivi de la convergence
        self.loss_history_ = []
        best_loss = np.inf
        iterations_without_improvement = 0

        # Descente de gradient
        for iteration in range(self.n_iterations):

            linear_model = X_scaled @ self.weights_ + self.bias_
            predictions = self.sigmoid(linear_model)

            errors = predictions - y_binary

            dw = (X_scaled.T @ errors) / n_samples
            db = np.mean(errors)

            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            # Calcul de la fonction de coût après la mise à jour
            updated_model = X_scaled @ self.weights_ + self.bias_
            updated_predictions = self.sigmoid(updated_model)

            loss = self.compute_loss(
                y_binary,
                updated_predictions,
            )

            self.loss_history_.append(loss)

            # Vérification de l'amélioration
            if best_loss - loss > self.tol:
                best_loss = loss
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            # Arrêt anticipé
            if iterations_without_improvement >= self.n_iter_no_change:
                break

        self.n_iter_ = iteration + 1
        self.final_loss_ = self.loss_history_[-1]
        self.converged_ = (
            iterations_without_improvement >= self.n_iter_no_change
        )

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
