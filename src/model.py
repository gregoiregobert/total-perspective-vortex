
import numpy as np

from sklearn.model_selection import (
    LeaveOneGroupOut,
    cross_val_score,
    ParameterGrid,
)
from sklearn.metrics import accuracy_score

from src.config import (
    DATASET_NAME,
    FEATURE_TYPE,
    CLASSIFIER_TYPE,
    RUNS,
)

from src.preprocessing import load_dataset
from src.pipeline import create_pipeline


# ============================================================
# DATA LOADING AND SPLITTING
# ============================================================

def load_subject_data(subject):

    if DATASET_NAME == "physionet":
        X, y, groups = load_dataset(
            subject=subject,
            runs=RUNS,
        )

        cv_groups = np.asarray(RUNS[:-1])
        test_group = RUNS[-1]

    elif DATASET_NAME == "bci_competition":
        X, y, groups = load_dataset(
            subject=subject,
        )

        # Les groupes BCI sont créés par factorize()
        # dans le loader MOABB.
        available_groups = np.sort(np.unique(groups))

        if len(available_groups) < 2:
            raise ValueError(
                f"Subject {subject}: at least two BCI runs "
                "are required."
            )

        # Tous les runs sauf le dernier servent à la validation.
        cv_groups = available_groups[:-1]
        test_group = available_groups[-1]

    else:
        raise ValueError(
            f"Unknown dataset: {DATASET_NAME}"
        )

    # Vérification des groupes attendus.
    available_groups = np.sort(np.unique(groups))
    expected_groups = np.sort(
        np.append(cv_groups, test_group)
    )

    if not np.array_equal(
        available_groups,
        expected_groups,
    ):
        raise ValueError(
            f"Subject {subject}: unexpected groups. "
            f"Expected {expected_groups}, "
            f"found {available_groups}."
        )

    # Séparation des données de validation et de test.
    cv_mask = np.isin(groups, cv_groups)
    test_mask = groups == test_group

    X_cv = X[cv_mask]
    y_cv = y[cv_mask]
    groups_cv = groups[cv_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    if len(X_cv) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Subject {subject}: empty CV or test data."
        )

    if len(np.unique(y_cv)) != 2:
        raise ValueError(
            f"Subject {subject}: both classes are required "
            "in the CV data."
        )

    if len(np.unique(y_test)) != 2:
        raise ValueError(
            f"Subject {subject}: both classes are required "
            "in the test data."
        )

    return (
        X_cv,
        y_cv,
        groups_cv,
        X_test,
        y_test,
        cv_groups,
        test_group,
    )


# ============================================================
# PARAMETER GRID
# ============================================================

def create_parameter_grid():

    parameters = {}

    # CSP dimensionality reduction.
    if FEATURE_TYPE in ("csp", "combined"):
        parameters["n_csp_components"] = [2, 4, 6]

    # Wavelet feature extraction.
    if FEATURE_TYPE in ("wavelets", "combined"):
        parameters["wavelet"] = [
            "db4",
            "sym4",
            "coif1",
        ]
        parameters["wavelet_level"] = [3, 4]

    # Classifier hyperparameters.
    if CLASSIFIER_TYPE == "lda":
        parameters["lda_shrinkage"] = [
            None,
            "auto",
            0.1,
            0.5,
        ]

    elif CLASSIFIER_TYPE == "logistic":
        parameters["classifier_C"] = [
            0.1,
            1.0,
            10.0,
        ]
        parameters["iterations"] = [
            1000,
            2000,
        ]

    elif CLASSIFIER_TYPE == "custom_logistic":
        parameters["learning_rate"] = [
            0.01,
            0.1,
        ]
        parameters["iterations"] = [
            1000,
            2000,
        ]
        parameters["tol"] = [
            1e-4,
            1e-6,
        ]
        parameters["patience"] = [
            10,
            20,
        ]

    else:
        raise ValueError(
            f"Unknown classifier: {CLASSIFIER_TYPE}"
        )

    return list(ParameterGrid(parameters))


# ============================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================

def optimize_subject(subject, parameter_grid=None):

    if parameter_grid is None:
        parameter_grid = create_parameter_grid()

    (
        X,
        y,
        groups,
        X_test,
        y_test,
        cv_groups,
        test_group,
    ) = load_subject_data(subject)

    cv = LeaveOneGroupOut()

    best_accuracy = -np.inf
    best_parameters = None
    best_scores = None

    print(f"CV groups: {cv_groups}")
    print(f"Reserved test group: {test_group}")

    for index, parameters in enumerate(
        parameter_grid,
        start=1,
    ):

        pipeline = create_pipeline(**parameters)

        scores = cross_val_score(
            pipeline,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring="accuracy",
            error_score="raise",
        )

        mean_accuracy = scores.mean() * 100

        print(
            f"Configuration {index}/{len(parameter_grid)}: "
            f"{mean_accuracy:.2f}%"
        )

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_parameters = parameters.copy()
            best_scores = scores.copy()

    return {
        "subject": subject,
        "cv_accuracy": float(best_accuracy),
        "cv_scores": best_scores,
        "best_parameters": best_parameters,
    }


# ============================================================
# TRAIN BEST PIPELINE
# ============================================================

def train_best_pipeline(subject, best_parameters):

    (
        X,
        y,
        groups,
        X_test,
        y_test,
        cv_groups,
        test_group,
    ) = load_subject_data(subject)

    pipeline = create_pipeline(
        **best_parameters,
    )

    # Entraînement sur tous les runs de validation.
    # Le run réservé au test n'est jamais utilisé ici.
    pipeline.fit(
        X,
        y,
    )

    return pipeline


# ============================================================
# FINAL TEST
# ============================================================

def test_subject(subject, pipeline):

    (
        X,
        y,
        groups,
        X_test,
        y_test,
        cv_groups,
        test_group,
    ) = load_subject_data(subject)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    return float(accuracy * 100)
