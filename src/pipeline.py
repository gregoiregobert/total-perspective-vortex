
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from mne.decoding import CSP as MNE_CSP

from src.config import (
    CSP_TYPE,
    FEATURE_TYPE,
    CLASSIFIER_TYPE,
    N_CSP_COMPONENTS,
    WAVELET,
    WAVELET_LEVEL,
    LOGISTIC_LEARNING_RATE,
    LOGISTIC_ITERATIONS,
)

from src.csp import CSP
from src.wavelet_transformer import WaveletTransformer
from src.logistic_regression import LogisticRegressionCustom


def create_csp():

    if CSP_TYPE == "custom":

        return CSP(
            n_components=N_CSP_COMPONENTS,
        )

    elif CSP_TYPE == "mne":

        return MNE_CSP(
            n_components=N_CSP_COMPONENTS,
            log=True,
            norm_trace=False,
        )

    raise ValueError(f"Unknown CSP type: {CSP_TYPE}")


def create_classifier():

    if CLASSIFIER_TYPE == "lda":

        return LinearDiscriminantAnalysis()

    elif CLASSIFIER_TYPE == "logistic":

        return LogisticRegression(
            max_iter=2000,
        )

    elif CLASSIFIER_TYPE == "custom_logistic":

        return LogisticRegressionCustom(
            learning_rate=LOGISTIC_LEARNING_RATE,
            n_iterations=LOGISTIC_ITERATIONS,
        )

    raise ValueError(
        f"Unknown classifier: {CLASSIFIER_TYPE}"
    )


def create_pipeline():

    classifier = create_classifier()

    if FEATURE_TYPE == "csp":

        return Pipeline([
            ("csp", create_csp()),
            ("classifier", classifier),
        ])

    elif FEATURE_TYPE == "wavelets":

        return Pipeline([
            ("wavelets", WaveletTransformer(
                wavelet=WAVELET,
                level=WAVELET_LEVEL,
            )),
            ("classifier", classifier),
        ])

    elif FEATURE_TYPE == "combined":

        features = FeatureUnion([
            ("csp", create_csp()),
            ("wavelets", WaveletTransformer(
                wavelet=WAVELET,
                level=WAVELET_LEVEL,
            )),
        ])

        return Pipeline([
            ("features", features),
            ("classifier", classifier),
        ])

    raise ValueError(
        f"Unknown feature type: {FEATURE_TYPE}"
    )
