
import numpy as np
from scipy.linalg import eigh

from src.eigen_solver import (
    jacobi_eigenvalue_decomposition,
    generalized_eigenvalue_decomposition,
)


def main():
    rng = np.random.default_rng(42)

    # Test 1: standard symmetric eigenvalue problem
    M = rng.normal(size=(8, 8))
    A = M @ M.T

    custom_values, custom_vectors = (
        jacobi_eigenvalue_decomposition(A)
    )

    scipy_values, scipy_vectors = np.linalg.eigh(A)

    print("\n===== STANDARD EIGENVALUE TEST =====")
    print(
        "Maximum eigenvalue difference:",
        np.max(np.abs(custom_values - scipy_values)),
    )

    residual = np.linalg.norm(
        A @ custom_vectors
        - custom_vectors * custom_values
    )

    print("Eigenvalue equation residual:", residual)

    # Test 2: generalized eigenvalue problem
    M1 = rng.normal(size=(8, 8))
    M2 = rng.normal(size=(8, 8))

    A = M1 @ M1.T
    B = M2 @ M2.T + np.eye(8)

    custom_values, custom_vectors = (
        generalized_eigenvalue_decomposition(A, B)
    )

    scipy_values, scipy_vectors = eigh(A, B)

    print("\n===== GENERALIZED EIGENVALUE TEST =====")
    print(
        "Maximum eigenvalue difference:",
        np.max(np.abs(custom_values - scipy_values)),
    )

    residual = np.linalg.norm(
        A @ custom_vectors
        - (B @ custom_vectors) * custom_values
    )

    print("Generalized eigenvalue residual:", residual)


if __name__ == "__main__":
    main()
