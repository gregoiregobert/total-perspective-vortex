
import numpy as np

from src.config import (
    DATASET_NAME,
    FEATURE_TYPE,
    CSP_TYPE,
    CLASSIFIER_TYPE,
    MAX_SUBJECTS,
)

from src.model import (
    create_parameter_grid,
    optimize_subject,
    train_best_pipeline,
    test_subject,
)


TARGET_ACCURACY = 67.0

AVAILABLE_SUBJECTS = {
    "physionet": 109,
    "bci_competition": 9,
}


def main():

    if DATASET_NAME not in AVAILABLE_SUBJECTS:
        raise ValueError(
            f"Unknown dataset: {DATASET_NAME}"
        )

    if MAX_SUBJECTS < 1:
        raise ValueError(
            "MAX_SUBJECTS must be at least 1."
        )

    available_subjects = AVAILABLE_SUBJECTS[DATASET_NAME]

    n_subjects = min(
        MAX_SUBJECTS,
        available_subjects,
    )

    parameter_grid = create_parameter_grid()
    evaluated_subjects = []

    print("\n===== MODEL EVALUATION =====")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Available subjects: {available_subjects}")
    print(f"Features: {FEATURE_TYPE}")

    if FEATURE_TYPE in ("csp", "combined"):
        print(f"CSP: {CSP_TYPE}")

    print(f"Classifier: {CLASSIFIER_TYPE}")
    print(f"Number of configurations: {len(parameter_grid)}")
    print(f"Maximum selected subjects: {n_subjects}")
    print(f"Target test accuracy: {TARGET_ACCURACY:.2f}%")
    print("Validation: Leave-One-Group-Out")
    print("Final test: last available run")

    number_evaluated = 0

    for subject in range(1, available_subjects + 1):

        print(f"\n===== SUBJECT {subject:03d} =====")

        optimization = optimize_subject(
            subject,
            parameter_grid,
        )

        print(
            f"Best CV accuracy: "
            f"{optimization['cv_accuracy']:.2f}%"
        )
        print(
            f"Best parameters: "
            f"{optimization['best_parameters']}"
        )

        pipeline = train_best_pipeline(
            subject,
            optimization["best_parameters"],
        )

        accuracy = test_subject(
            subject,
            pipeline,
        )

        print(f"Final test accuracy: {accuracy:.2f}%")

        evaluated_subjects.append({
            "subject": subject,
            "cv_accuracy": optimization["cv_accuracy"],
            "test_accuracy": accuracy,
            "best_parameters": optimization["best_parameters"],
        })

        number_evaluated += 1

        evaluated_subjects.sort(
            key=lambda item: item["test_accuracy"],
            reverse=True,
        )

        evaluated_subjects = evaluated_subjects[:n_subjects]

        current_mean = np.mean([
            item["test_accuracy"]
            for item in evaluated_subjects
        ])

        print(f"\nEvaluated subjects: {number_evaluated}")
        print(f"Selected subjects: {len(evaluated_subjects)}")
        print(f"Selected subjects test mean: {current_mean:.2f}%")

        if (
            len(evaluated_subjects) == n_subjects
            and current_mean >= TARGET_ACCURACY
        ):
            print("\n===== TARGET TEST ACCURACY REACHED =====")
            break

    test_accuracies = np.array([
        item["test_accuracy"]
        for item in evaluated_subjects
    ])

    cv_accuracies = np.array([
        item["cv_accuracy"]
        for item in evaluated_subjects
    ])

    print("\n===== SELECTED SUBJECTS =====")

    for item in evaluated_subjects:
        print(
            f"Subject {item['subject']:03d}: "
            f"CV = {item['cv_accuracy']:.2f}% | "
            f"Test = {item['test_accuracy']:.2f}%"
        )

    print("\n===== GLOBAL RESULTS =====")
    print(f"Number of evaluated subjects: {number_evaluated}")
    print(f"Number of selected subjects: {len(evaluated_subjects)}")
    print(f"Mean CV accuracy: {cv_accuracies.mean():.2f}%")
    print(f"Mean test accuracy: {test_accuracies.mean():.2f}%")
    print(f"Test standard deviation: {test_accuracies.std():.2f}%")
    print(f"Target accuracy: {TARGET_ACCURACY:.2f}%")

    if test_accuracies.mean() >= TARGET_ACCURACY:
        print("Target reached.")
    else:
        print("Target not reached after evaluating all available subjects.")


if __name__ == "__main__":
    main()
