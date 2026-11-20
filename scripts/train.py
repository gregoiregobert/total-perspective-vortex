
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    LeaveOneGroupOut,
    GroupKFold,
    cross_val_score,
)
from sklearn.metrics import accuracy_score

from src.config import (
    DATASET_NAME,
    FEATURE_TYPE,
    CSP_TYPE,
    CLASSIFIER_TYPE,
)

from src.preprocessing import load_dataset
from src.pipeline import create_pipeline


def split_data(X, y, groups):

    unique_groups = np.sort(np.unique(groups))

    if DATASET_NAME == "physionet":

        test_group = unique_groups[-1]

        test_mask = groups == test_group
        trainval_mask = ~test_mask

        X_trainval = X[trainval_mask]
        y_trainval = y[trainval_mask]
        groups_trainval = groups[trainval_mask]

        indices = np.arange(len(X_trainval))

        train_indices, val_indices = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=y_trainval,
        )

        return (
            X_trainval[train_indices],
            y_trainval[train_indices],
            groups_trainval[train_indices],
            X_trainval[val_indices],
            y_trainval[val_indices],
            X[test_mask],
            y[test_mask],
        )

    elif DATASET_NAME == "bci_competition":

        if len(unique_groups) < 3:
            raise ValueError(
                "BCI training requires at least three runs."
            )

        test_group = unique_groups[-1]
        val_group = unique_groups[-2]

        train_mask = (
            (groups != test_group)
            & (groups != val_group)
        )

        val_mask = groups == val_group
        test_mask = groups == test_group

        return (
            X[train_mask],
            y[train_mask],
            groups[train_mask],
            X[val_mask],
            y[val_mask],
            X[test_mask],
            y[test_mask],
        )

    raise ValueError(
        f"Unknown dataset: {DATASET_NAME}"
    )


def main():

    X, y, groups = load_dataset(
        subject=1,
        runs=(4, 8, 12),
    )

    (
        X_train,
        y_train,
        groups_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = split_data(X, y, groups)

    pipeline = create_pipeline()

    print("\n===== MODEL TRAINING =====")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Features: {FEATURE_TYPE}")

    if FEATURE_TYPE in ("csp", "combined"):
        print(f"CSP: {CSP_TYPE}")

    print(f"Classifier: {CLASSIFIER_TYPE}")

    print(f"\nTraining epochs: {len(X_train)}")
    print(f"Validation epochs: {len(X_val)}")
    print(f"Test epochs: {len(X_test)}")

    # Cross-validation on the training runs.
    if DATASET_NAME == "physionet":

        cv = LeaveOneGroupOut()

    else:

        n_splits = min(
            5,
            len(np.unique(groups_train)),
        )

        cv = GroupKFold(n_splits=n_splits)

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring="accuracy",
        error_score="raise",
    )

    print("\n===== CROSS VALIDATION =====")
    print(f"Scores: {cv_scores}")
    print(f"Mean accuracy: {cv_scores.mean() * 100:.2f}%")

    # Fit on training data.
    pipeline.fit(X_train, y_train)

    val_predictions = pipeline.predict(X_val)
    val_accuracy = accuracy_score(
        y_val,
        val_predictions,
    )

    print("\n===== VALIDATION =====")
    print(f"Accuracy: {val_accuracy * 100:.2f}%")

    # Refit on training + validation data.
    X_trainval = np.concatenate(
        [X_train, X_val],
        axis=0,
    )

    y_trainval = np.concatenate(
        [y_train, y_val],
        axis=0,
    )

    pipeline.fit(X_trainval, y_trainval)

    test_predictions = pipeline.predict(X_test)
    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    print("\n===== FINAL TEST =====")
    print(f"Accuracy: {test_accuracy * 100:.2f}%")

    print("\nTraining completed.")


if __name__ == "__main__":
    main()
