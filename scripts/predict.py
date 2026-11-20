
import time
import numpy as np

from sklearn.metrics import accuracy_score

from src.config import (
    DATASET_NAME,
    FEATURE_TYPE,
    CSP_TYPE,
    CLASSIFIER_TYPE,
)

from src.preprocessing import load_dataset
from src.pipeline import create_pipeline


def main():

    X, y, groups = load_dataset(
        subject=2,
        runs=(4, 8, 12),
    )

    unique_groups = np.sort(np.unique(groups))

    if len(unique_groups) < 2:
        raise ValueError(
            "At least two runs are required for prediction."
        )

    test_group = unique_groups[-1]

    train_mask = groups != test_group
    test_mask = groups == test_group

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    pipeline = create_pipeline()

    print("\n===== MODEL PREDICTION =====")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Features: {FEATURE_TYPE}")

    if FEATURE_TYPE in ("csp", "combined"):
        print(f"CSP: {CSP_TYPE}")

    print(f"Classifier: {CLASSIFIER_TYPE}")

    print(f"\nTraining epochs: {len(X_train)}")
    print(f"Test epochs: {len(X_test)}")

    pipeline.fit(X_train, y_train)

    predictions = []
    latencies = []

    for epoch in X_test:

        start = time.perf_counter()

        prediction = pipeline.predict(
            epoch[np.newaxis, :, :]
        )[0]

        elapsed = time.perf_counter() - start

        predictions.append(prediction)
        latencies.append(elapsed * 1000)

    predictions = np.asarray(predictions)
    latencies = np.asarray(latencies)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("\n===== PREDICTION RESULTS =====")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(
        f"Average prediction latency: "
        f"{latencies.mean():.3f} ms"
    )
    print(
        f"Maximum prediction latency: "
        f"{latencies.max():.3f} ms"
    )

    print("\n===== PREDICTIONS =====")

    for index, (prediction, expected) in enumerate(
        zip(predictions, y_test)
    ):

        print(
            f"Epoch {index + 1:03d}: "
            f"Predicted={prediction}, "
            f"Expected={expected}"
        )


if __name__ == "__main__":
    main()
