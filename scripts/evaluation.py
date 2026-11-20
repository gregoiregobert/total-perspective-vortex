
import numpy as np

from sklearn.model_selection import (
    LeaveOneGroupOut,
    cross_val_score,
)

from src.config import (
    DATASET_NAME,
    FEATURE_TYPE,
    CSP_TYPE,
    CLASSIFIER_TYPE,
    MAX_SUBJECTS,
)

from src.preprocessing import load_dataset
from src.pipeline import create_pipeline


def evaluate_subject(subject):

    X, y, groups = load_dataset(subject=subject)

    if len(np.unique(groups)) < 2:
        raise ValueError(
            f"Subject {subject}: at least two runs are required."
        )

    pipeline = create_pipeline()
    cv = LeaveOneGroupOut()

    scores = cross_val_score(
        pipeline,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="accuracy",
        error_score="raise",
    )

    return scores


def main():

    if DATASET_NAME == "physionet":
        available_subjects = 109

    elif DATASET_NAME == "bci_competition":
        available_subjects = 9

    else:
        raise ValueError(
            f"Unknown dataset: {DATASET_NAME}"
        )

    n_subjects = min(MAX_SUBJECTS, available_subjects)
    subjects = range(1, n_subjects + 1)
    
    all_scores = []

    print("\n===== MODEL EVALUATION =====")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Features: {FEATURE_TYPE}")

    if FEATURE_TYPE in ("csp", "combined"):
        print(f"CSP: {CSP_TYPE}")

    print(f"Classifier: {CLASSIFIER_TYPE}")

    for subject in subjects:

        scores = evaluate_subject(subject)

        all_scores.extend(scores)

        print(
            f"Subject {subject:02d}: "
            f"{scores.mean() * 100:.2f}%"
        )

    all_scores = np.asarray(all_scores)

    print("\n===== GLOBAL RESULTS =====")
    print(f"Number of evaluations: {len(all_scores)}")
    print(f"Mean accuracy: {all_scores.mean() * 100:.2f}%")
    print(
        f"Standard deviation: "
        f"{all_scores.std() * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
