
import numpy as np


def covariance_matrix(X):
    """
    Compute the covariance matrix of a dataset.

    X shape:
        (n_features, n_samples)

    Returns:
        (n_features, n_features)
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    n_features, n_samples = X.shape

    if n_samples < 2:
        raise ValueError("At least two samples are required.")

    # Center each feature by subtracting its mean
    centered = X - np.mean(X, axis=1, keepdims=True)

    # Compute covariance matrix
    covariance = (centered @ centered.T) / (n_samples - 1)

    return covariance
