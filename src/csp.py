
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from src.math_utils import covariance_matrix
from src.eigen_solver import generalized_eigenvalue_decomposition


class CSP(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns for binary EEG classification."""

    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y):
        """
        Learn spatial filters from EEG epochs.

        X shape: (n_epochs, n_channels, n_times)
        y shape: (n_epochs,)
        """

        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("CSP requires exactly two classes.")

        if self.n_components % 2 != 0:
            raise ValueError("n_components must be even.")

        if self.n_components > X.shape[1]:
            raise ValueError("Too many components for the number of channels.")

        covariances = []

        # Calculate one normalized covariance matrix per epoch
        for epoch in X:
            cov = covariance_matrix(epoch)
            cov /= np.trace(cov)
            covariances.append(cov)

        covariances = np.asarray(covariances)

        # Average covariance matrices for each class
        cov_class_1 = covariances[y == self.classes_[0]].mean(axis=0)
        cov_class_2 = covariances[y == self.classes_[1]].mean(axis=0)

        # Solve the generalized eigenvalue problem
        eigenvalues, eigenvectors = generalized_eigenvalue_decomposition(
            cov_class_1,
            cov_class_1 + cov_class_2,
        )

        # Sort eigenvectors by decreasing eigenvalue
        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, order]

        # Keep the most discriminative filters from both extremes
        n_filters = self.n_components // 2
        selected = np.concatenate([
            np.arange(n_filters),
            np.arange(-n_filters, 0),
        ])

        self.filters_ = eigenvectors[:, selected]

        return self

    def transform(self, X):
        """Project epochs and extract log-variance features."""

        X = np.asarray(X)

        features = []

        for epoch in X:
            projected = self.filters_.T @ epoch

            variances = np.var(projected, axis=1)

            normalized_variances = variances / np.sum(variances)

            features.append(np.log(normalized_variances + 1e-10))

        return np.asarray(features)


#tmp
if __name__ == "__main__":
    from src.preprocessing import load_eeg, filter_eeg, create_epochs

    raw = load_eeg(subject=1, run=4)
    filtered = filter_eeg(raw)
    epochs = create_epochs(filtered)

    X = epochs.get_data()
    y = epochs.events[:, -1]

    csp = CSP(n_components=4)

    X_transformed = csp.fit_transform(X, y)

    print("\n===== CSP RESULTS =====")
    print("Original shape:", X.shape)
    print("Transformed shape:", X_transformed.shape)
    print("Features:")
    print(X_transformed[:3])
