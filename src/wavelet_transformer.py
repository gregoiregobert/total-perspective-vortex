
import numpy as np
import pywt

from sklearn.base import BaseEstimator, TransformerMixin


class WaveletTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, wavelet="db4", level=4):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        """
        No learning is required for the wavelet decomposition.
        The wavelet and decomposition level are fixed in advance.
        """
        return self

    def transform(self, X):
        """
        Transform each EEG epoch into wavelet energy features.
        """
        X = np.asarray(X)

        all_features = []

        for epoch in X:
            epoch_features = []

            # Process each EEG channel
            for channel_signal in epoch:

                coefficients = pywt.wavedec(
                    channel_signal,
                    self.wavelet,
                    level=self.level,
                )

                # Calculate the log-energy of each component
                energies = [
                    np.log(np.sum(coef ** 2) + 1e-10)
                    for coef in coefficients
                ]

                epoch_features.extend(energies)

            all_features.append(epoch_features)

        return np.asarray(all_features)
