
import time
import numpy as np
import mne

from sklearn.metrics import accuracy_score

from src.config import (
    DATASET_NAME,
    RUNS,
)

from src.datasets.physionet import (
    load_eeg as load_physionet,
)

from src.preprocessing import (
    filter_eeg,
)

from src.model import (
    optimize_subject,
    train_best_pipeline,
    load_subject_data,
)


def simulate_realtime(
    subject=1,
    chunk_duration=1.0,
    playback_speed=1.0,
    tmin=1,
    tmax=4,
):
    """
    Simule une réception progressive du signal EEG.

    PhysioNet :
        Lecture progressive d'un signal continu.

    BCI Competition :
        Lecture progressive des epochs déjà préparées par MOABB.
    """

    if DATASET_NAME not in (
        "physionet",
        "bci_competition",
    ):
        raise ValueError(
            f"Unsupported dataset: {DATASET_NAME}"
        )

    if playback_speed <= 0:
        raise ValueError(
            "playback_speed must be positive."
        )

    if chunk_duration <= 0:
        raise ValueError(
            "chunk_duration must be positive."
        )

    # ==================================================
    # 1. OPTIMIZE AND TRAIN THE MODEL
    # ==================================================

    print("\n===== MODEL OPTIMIZATION =====")

    optimization = optimize_subject(subject)

    print(
        f"Best CV accuracy: "
        f"{optimization['cv_accuracy']:.2f}%"
    )

    print(
        f"Best parameters: "
        f"{optimization['best_parameters']}"
    )

    print("\n===== TRAINING =====")

    pipeline = train_best_pipeline(
        subject,
        optimization["best_parameters"],
    )

    print("Model successfully trained.")

    # Retrieve the held-out test data and its labels.
    (
        X_train,
        y_train,
        groups_train,
        X_test,
        y_test,
        cv_groups,
        test_group,
    ) = load_subject_data(subject)

    print(f"Training epochs: {len(X_train)}")
    print(f"Test epochs: {len(X_test)}")
    print(f"Training groups: {cv_groups}")
    print(f"Test group: {test_group}")

    # ==================================================
    # 2. INITIALIZE SIMULATION
    # ==================================================

    predictions = []
    expected_labels = []
    latencies = []

    print("\n===== REAL-TIME SIMULATION =====")

    # ==================================================
    # 3. PHYSIONET: CONTINUOUS SIGNAL
    # ==================================================

    if DATASET_NAME == "physionet":

        test_run = RUNS[-1]

        print("\nLoading continuous PhysioNet recording...")

        raw_list = load_physionet(
            subject=subject,
            runs=(test_run,),
        )

        raw = raw_list[0]

        # Apply the same filtering as during training.
        raw = filter_eeg(raw)

        sfreq = raw.info["sfreq"]
        total_samples = raw.n_times

        chunk_samples = max(
            1,
            int(round(chunk_duration * sfreq)),
        )

        print(f"Sampling frequency: {sfreq} Hz")
        print(
            f"Recording duration: "
            f"{raw.times[-1]:.2f} seconds"
        )
        print(f"Chunk duration: {chunk_duration} seconds")

        # Retrieve motor imagery events.
        events, _ = mne.events_from_annotations(
            raw,
            event_id={
                "T1": 1,
                "T2": 2,
            },
            verbose=False,
        )

        print(f"Number of events: {len(events)}")

        eeg_data = raw.get_data(picks="eeg")

        event_index = 0

        # Progressive playback of the continuous signal.
        for chunk_start in range(
            0,
            total_samples,
            chunk_samples,
        ):

            chunk_end = min(
                chunk_start + chunk_samples,
                total_samples,
            )

            actual_chunk_duration = (
                (chunk_end - chunk_start) / sfreq
            )

            time.sleep(
                actual_chunk_duration / playback_speed
            )

            # Process events when their complete epochs are available.
            while event_index < len(events):

                event_sample = events[event_index, 0]
                event_label = events[event_index, 2]

                epoch_start = event_sample + int(
                    round(tmin * sfreq)
                )

                epoch_end = event_sample + int(
                    round(tmax * sfreq)
                )

                # chunk_end is an exclusive index.
                if chunk_end < epoch_end + 1:
                    break

                epoch = eeg_data[
                    :,
                    epoch_start:epoch_end + 1,
                ]

                expected_samples = (
                    int(round((tmax - tmin) * sfreq)) + 1
                )

                if epoch.shape[1] != expected_samples:
                    event_index += 1
                    continue

                # Make prediction and measure inference latency.
                start_time = time.perf_counter()

                prediction = pipeline.predict(
                    epoch[np.newaxis, :, :]
                )[0]

                latency = time.perf_counter() - start_time

                predictions.append(prediction)
                expected_labels.append(event_label)
                latencies.append(latency)

                print(
                    f"Event {event_index + 1}: "
                    f"prediction={prediction}, "
                    f"expected={event_label}, "
                    f"latency={latency * 1000:.2f} ms"
                )

                if latency >= 2.0:
                    print(
                        "WARNING: Prediction exceeded 2 seconds!"
                    )

                event_index += 1

    # ==================================================
    # 4. BCI: PRE-EPOCHED SIGNAL
    # ==================================================

    elif DATASET_NAME == "bci_competition":

        print("\nSimulating BCI epoch playback...")

        sfreq = 160.0
        n_epochs = len(X_test)

        n_channels = X_test.shape[1]
        n_samples = X_test.shape[2]

        chunk_samples = max(
            1,
            int(round(chunk_duration * sfreq)),
        )

        print(f"Sampling frequency: {sfreq} Hz")
        print(f"Number of epochs: {n_epochs}")
        print(f"Number of channels: {n_channels}")
        print(
            f"Epoch duration: "
            f"{n_samples / sfreq:.2f} seconds"
        )
        print(f"Chunk duration: {chunk_duration} seconds")

        # Simulate the arrival of each test epoch.
        for epoch_index, epoch in enumerate(X_test):

            print(
                f"\nReceiving epoch "
                f"{epoch_index + 1}/{n_epochs}..."
            )

            # Receive the epoch progressively.
            for chunk_start in range(
                0,
                n_samples,
                chunk_samples,
            ):

                chunk_end = min(
                    chunk_start + chunk_samples,
                    n_samples,
                )

                actual_chunk_duration = (
                    (chunk_end - chunk_start) / sfreq
                )

                time.sleep(
                    actual_chunk_duration / playback_speed
                )

            # The complete epoch is now available.
            start_time = time.perf_counter()

            prediction = pipeline.predict(
                epoch[np.newaxis, :, :]
            )[0]

            latency = time.perf_counter() - start_time

            expected_label = y_test[epoch_index]

            predictions.append(prediction)
            expected_labels.append(expected_label)
            latencies.append(latency)

            print(
                f"Epoch {epoch_index + 1}: "
                f"prediction={prediction}, "
                f"expected={expected_label}, "
                f"latency={latency * 1000:.2f} ms"
            )

            if latency >= 2.0:
                print(
                    "WARNING: Prediction exceeded 2 seconds!"
                )

    # ==================================================
    # 5. DISPLAY RESULTS
    # ==================================================

    print("\n===== REAL-TIME RESULTS =====")

    if not predictions:
        print("No predictions were generated.")
        return

    accuracy = accuracy_score(
        expected_labels,
        predictions,
    )

    average_latency = np.mean(latencies)
    max_latency = np.max(latencies)

    print(f"Number of predictions: {len(predictions)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(
        f"Average prediction latency: "
        f"{average_latency * 1000:.2f} ms"
    )

    print(
        f"Maximum prediction latency: "
        f"{max_latency * 1000:.2f} ms"
    )

    if max_latency < 2.0:
        print("LATENCY REQUIREMENT: PASSED")
    else:
        print("LATENCY REQUIREMENT: FAILED")


if __name__ == "__main__":

    simulate_realtime(
        subject=2,
        chunk_duration=1.0,
        playback_speed=1.0,
    )
