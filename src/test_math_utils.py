
import numpy as np

from src.math_utils import covariance_matrix
from src.preprocessing import load_eeg, filter_eeg, create_epochs


def main():
    raw = load_eeg(subject=1, runs=(4,))
    filtered = filter_eeg(raw)
    epochs = create_epochs(filtered)

    X = epochs.get_data()

    # Select the first epoch
    epoch = X[0]

    # Compare our implementation with NumPy
    custom_covariance = covariance_matrix(epoch)
    numpy_covariance = np.cov(epoch)

    difference = np.max(
        np.abs(custom_covariance - numpy_covariance)
    )

    print("\n===== COVARIANCE IMPLEMENTATION TEST =====")
    print("Custom matrix shape:", custom_covariance.shape)
    print("NumPy matrix shape:", numpy_covariance.shape)
    print(f"Maximum difference: {difference:.12e}")

    print(
        "Matrices equivalent:",
        np.allclose(custom_covariance, numpy_covariance),
    )


if __name__ == "__main__":
    main()
