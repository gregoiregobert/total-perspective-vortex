
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from mne.decoding import CSP as MNE_CSP

from src.config import (
    CSP_TYPE,
    FEATURE_TYPE,
    CLASSIFIER_TYPE,
)

from src.csp import CSP
from src.wavelet_transformer import WaveletTransformer
from src.logistic_regression import LogisticRegressionCustom


# Valeurs par défaut utilisées lorsque le pipeline est appelé
# sans paramètres particuliers, notamment par train.py/predict.py.

DEFAULT_N_CSP_COMPONENTS = 4
DEFAULT_WAVELET = "db4"
DEFAULT_WAVELET_LEVEL = 4

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_ITERATIONS = 2000
DEFAULT_TOL = 1e-6
DEFAULT_PATIENCE = 20
DEFAULT_LOGISTIC_C = 1.0
DEFAULT_LDA_SHRINKAGE = "auto"


def create_csp(n_components):

    if CSP_TYPE == "custom":

        return CSP(
            n_components=n_components,
        )

    elif CSP_TYPE == "mne":

        return MNE_CSP(
            n_components=n_components,
            log=True,
            norm_trace=False,
        )

    raise ValueError(
        f"Unknown CSP type: {CSP_TYPE}"
    )


def create_classifier(
    learning_rate=None,
    iterations=None,
    tol=None,
    patience=None,
    classifier_C=None,
    lda_shrinkage=None,
):

    if CLASSIFIER_TYPE == "lda":

        if lda_shrinkage is None:
            lda_shrinkage = DEFAULT_LDA_SHRINKAGE

        return LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage=lda_shrinkage,
        )

    elif CLASSIFIER_TYPE == "logistic":

        if iterations is None:
            iterations = DEFAULT_ITERATIONS

        if classifier_C is None:
            classifier_C = DEFAULT_LOGISTIC_C

        return LogisticRegression(
            C=classifier_C,
            max_iter=iterations,
        )

    elif CLASSIFIER_TYPE == "custom_logistic":

        if learning_rate is None:
            learning_rate = DEFAULT_LEARNING_RATE

        if iterations is None:
            iterations = DEFAULT_ITERATIONS

        if tol is None:
            tol = DEFAULT_TOL

        if patience is None:
            patience = DEFAULT_PATIENCE

        return LogisticRegressionCustom(
            learning_rate=learning_rate,
            n_iterations=iterations,
            tol=tol,
            n_iter_no_change=patience,
        )

    raise ValueError(
        f"Unknown classifier: {CLASSIFIER_TYPE}"
    )


def create_pipeline(
    n_csp_components=None,
    wavelet=None,
    wavelet_level=None,
    learning_rate=None,
    iterations=None,
    tol=None,
    patience=None,
    classifier_C=None,
    lda_shrinkage=None,
):

    if n_csp_components is None:
        n_csp_components = DEFAULT_N_CSP_COMPONENTS

    if wavelet is None:
        wavelet = DEFAULT_WAVELET

    if wavelet_level is None:
        wavelet_level = DEFAULT_WAVELET_LEVEL

    classifier = create_classifier(
        learning_rate=learning_rate,
        iterations=iterations,
        tol=tol,
        patience=patience,
        classifier_C=classifier_C,
        lda_shrinkage=lda_shrinkage,
    )

    if FEATURE_TYPE == "csp":

        return Pipeline([
            (
                "csp",
                create_csp(n_csp_components),
            ),
            ("classifier", classifier),
        ])

    elif FEATURE_TYPE == "wavelets":

        return Pipeline([
            (
                "wavelets",
                WaveletTransformer(
                    wavelet=wavelet,
                    level=wavelet_level,
                ),
            ),
            ("classifier", classifier),
        ])

    elif FEATURE_TYPE == "combined":

        features = FeatureUnion([
            (
                "csp",
                create_csp(n_csp_components),
            ),
            (
                "wavelets",
                WaveletTransformer(
                    wavelet=wavelet,
                    level=wavelet_level,
                ),
            ),
        ])

        return Pipeline([
            ("features", features),
            ("classifier", classifier),
        ])

    raise ValueError(
        f"Unknown feature type: {FEATURE_TYPE}"
    )
