
import numpy as np


def jacobi_eigenvalue_decomposition(
    matrix,
    tolerance=1e-10,
    max_sweeps=100,
):
    """
    Compute eigenvalues and eigenvectors of a symmetric matrix
    using the Jacobi rotation method.
    """
    A = np.asarray(matrix, dtype=float).copy()

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("Matrix must be symmetric.")

    n = A.shape[0]
    eigenvectors = np.eye(n)

    for _ in range(max_sweeps):

        max_off_diagonal = np.max(
            np.abs(A - np.diag(np.diag(A)))
        )

        if max_off_diagonal < tolerance:
            break

        for p in range(n - 1):
            for q in range(p + 1, n):

                if abs(A[p, q]) < tolerance:
                    continue

                tau = (A[q, q] - A[p, p]) / (2 * A[p, q])

                if tau >= 0:
                    t = 1 / (tau + np.sqrt(1 + tau ** 2))
                else:
                    t = -1 / (-tau + np.sqrt(1 + tau ** 2))

                c = 1 / np.sqrt(1 + t ** 2)
                s = t * c

                app = A[p, p]
                aqq = A[q, q]
                apq = A[p, q]

                A[p, p] = app - t * apq
                A[q, q] = aqq + t * apq
                A[p, q] = 0.0
                A[q, p] = 0.0

                for k in range(n):
                    if k != p and k != q:
                        akp = A[k, p]
                        akq = A[k, q]

                        A[k, p] = c * akp - s * akq
                        A[p, k] = A[k, p]

                        A[k, q] = s * akp + c * akq
                        A[q, k] = A[k, q]

                    vkp = eigenvectors[k, p]
                    vkq = eigenvectors[k, q]

                    eigenvectors[k, p] = c * vkp - s * vkq
                    eigenvectors[k, q] = s * vkp + c * vkq

    eigenvalues = np.diag(A).copy()

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return eigenvalues, eigenvectors


def generalized_eigenvalue_decomposition(A, B):
    """
    Solve A v = lambda B v for symmetric A and positive
    definite symmetric B.

    Uses Cholesky decomposition followed by the custom
    Jacobi eigenvalue solver.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape.")

    # Cholesky: B = L @ L.T
    L = np.linalg.cholesky(B)

    # Convert the generalized problem into a standard one:
    # C = inv(L) @ A @ inv(L.T)
    left = np.linalg.solve(L, A)
    C = np.linalg.solve(L, left.T).T

    # Remove small numerical asymmetry
    C = (C + C.T) / 2

    # Solve the standard symmetric eigenvalue problem
    eigenvalues, eigenvectors = (
        jacobi_eigenvalue_decomposition(C)
    )

    # Convert eigenvectors back to the original problem
    generalized_vectors = np.linalg.solve(
        L.T,
        eigenvectors,
    )

    return eigenvalues, generalized_vectors
