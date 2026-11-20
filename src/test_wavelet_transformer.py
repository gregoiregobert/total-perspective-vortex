
from src.preprocessing import load_eeg, filter_eeg, create_epochs
from src.wavelet_transformer import WaveletTransformer


def main():
    raw = load_eeg(subject=1, runs=(4,))
    filtered = filter_eeg(raw)
    epochs = create_epochs(filtered)

    X = epochs.get_data()
    y = epochs.events[:, -1]

    transformer = WaveletTransformer()

    X_features = transformer.fit_transform(X, y)

    print("\n===== WAVELET TRANSFORMER TEST =====")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {X_features.shape}")
    print(f"Number of labels: {len(y)}")
    print(f"First epoch features: {X_features[0].shape}")


if __name__ == "__main__":
    main()
