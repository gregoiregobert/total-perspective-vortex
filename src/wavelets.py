
import numpy as np
import matplotlib.pyplot as plt
import pywt

from src.preprocessing import load_eeg, filter_eeg, create_epochs


def decompose_epoch(epoch, channel_index=0, wavelet="db4", level=4):
    """
    Decompose one EEG epoch using the Discrete Wavelet Transform.
    """
    signal = epoch[channel_index]

    coefficients = pywt.wavedec(
        signal,
        wavelet,
        level=level,
    )

    return signal, coefficients


def extract_wavelet_features(epoch, wavelet="db4", level=4):
    """
    Extract wavelet energy features from all EEG channels.

    Returns 5 features per channel:
    A4, D4, D3, D2, D1.
    """
    features = []

    # Process each EEG channel
    for channel_signal in epoch:

        # Decompose the signal
        coefficients = pywt.wavedec(
            channel_signal,
            wavelet,
            level=level,
        )

        # Calculate the energy of each component
        energies = [
            np.log(np.sum(coef ** 2) + 1e-10)
            for coef in coefficients
        ]

        features.extend(energies)

    return np.asarray(features)


def compare_wavelet_classes(X, y):
    """
    Compare average wavelet energies between left and right hand imagery.
    """

    # Extract features for every epoch
    all_features = np.array([
        extract_wavelet_features(epoch)
        for epoch in X
    ])

    # Reshape: epochs x channels x 5 wavelet components
    all_features = all_features.reshape(
        len(X), 64, 5
    )

    components = ["A4", "D4", "D3", "D2", "D1"]

    print("\n===== WAVELET CLASS COMPARISON =====")

    for class_id, class_name in [(1, "Left hand"), (2, "Right hand")]:
        class_features = all_features[y == class_id]

        # Average across epochs and channels
        mean_energies = np.mean(
            class_features,
            axis=(0, 1),
        )

        print(f"\n{class_name}:")

        for component, energy in zip(components, mean_energies):
            print(f"{component}: {energy:.4f}")




def plot_decomposition(signal, coefficients, wavelet="db4"):
    """
    Display the original signal and its wavelet coefficients.
    """
    labels = [
        "Approximation (A4)",
        "Detail (D4)",
        "Detail (D3)",
        "Detail (D2)",
        "Detail (D1)",
    ]

    fig, axes = plt.subplots(
        len(coefficients) + 1,
        1,
        figsize=(12, 10),
    )

    axes[0].plot(signal)
    axes[0].set_title("Original EEG signal")
    axes[0].set_ylabel("Amplitude")

    for i, coefficient in enumerate(coefficients):
        axes[i + 1].plot(coefficient)
        axes[i + 1].set_title(labels[i])
        axes[i + 1].set_ylabel("Coefficient")

    axes[-1].set_xlabel("Sample index")
    fig.suptitle(f"Wavelet decomposition - {wavelet}")
    plt.tight_layout()
    plt.show()


def plot_channel_differences(X, y):
    """
    Display the difference between left and right hand
    wavelet energies for each EEG channel.
    """

    all_features = np.array([
        extract_wavelet_features(epoch)
        for epoch in X
    ])

    all_features = all_features.reshape(
        len(X), 64, 5
    )

    components = ["A4", "D4", "D3", "D2", "D1"]

    left_features = all_features[y == 1]
    right_features = all_features[y == 2]

    left_mean = np.mean(left_features, axis=0)
    right_mean = np.mean(right_features, axis=0)

    channel_differences = right_mean - left_mean

    fig, axes = plt.subplots(
        5, 1,
        figsize=(14, 12),
    )

    for i, component in enumerate(components):
        axes[i].bar(
            range(64),
            channel_differences[:, i],
        )
        axes[i].set_title(f"{component} - Right minus Left")
        axes[i].set_ylabel("Energy difference")
        axes[i].set_xlabel("EEG channel index")

    plt.tight_layout()
    plt.show()



def main():
    # Load one EEG recording
    raw = load_eeg(subject=1, runs=(4,))

    # Apply the same preprocessing as our existing pipeline
    filtered = filter_eeg(raw)
    epochs = create_epochs(filtered)

    # Select the first epoch
    X = epochs.get_data()

    y = epochs.events[:, -1]

    compare_wavelet_classes(X, y)

    plot_channel_differences(X, y)

    epoch = X[0]

    wavelet_features = extract_wavelet_features(epoch)

    print("\n===== WAVELET FEATURES =====")
    print(f"Number of features: {len(wavelet_features)}")
    print(f"Feature shape: {wavelet_features.shape}")
    print(f"First 5 features: {wavelet_features[:5]}")

    # Select channel C3 if available
    channel_name = "C3" if "C3" in epochs.ch_names else epochs.ch_names[0]
    channel_index = epochs.ch_names.index(channel_name)

    print(f"\nSelected channel: {channel_name}")
    print(f"Epoch shape: {epoch.shape}")

    signal, coefficients = decompose_epoch(
        epoch,
        channel_index=channel_index,
        wavelet="db4",
        level=4,
    )

    print("\n===== WAVELET INFORMATION =====")
    for name, coefficient in zip(
        ["A4", "D4", "D3", "D2", "D1"],
        coefficients,
    ):
        print(f"{name}: {len(coefficient)} coefficients")

    plot_decomposition(signal, coefficients)


if __name__ == "__main__":
    main()
