
import time
import numpy as np
import mne

from sklearn.metrics import accuracy_score

from src.config import DATASET_NAME
from src.datasets.physionet import load_eeg as load_physionet
from src.preprocessing import load_dataset, filter_eeg
from src.pipeline import create_pipeline


def simulate_realtime(
    subject=1,
    train_runs=(4, 8),
    test_run=12,
    chunk_duration=1.0,
    playback_speed=1.0,
    tmin=1,
    tmax=4,
):
    """
    Simulate real-time EEG playback from a continuous PhysioNet recording.

    The recording is read progressively in chunks.
    A prediction is made when the complete epoch is available.
    """

    if DATASET_NAME != "physionet":
        raise ValueError(
            "Real-time simulation is currently available "
            "only for PhysioNet."
        )

    if playback_speed <= 0:
        raise ValueError("playback_speed must be positive.")

    # --------------------------------------------------
    # 1. Train the model on the training runs
    # --------------------------------------------------

    print("\n===== TRAINING =====")

    X_train, y_train, _ = load_dataset(
        subject=subject,
        runs=train_runs,
    )

    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    print(f"Training samples: {len(X_train)}")

    # --------------------------------------------------
    # 2. Load the continuous test recording
    # --------------------------------------------------

    print("\n===== LOADING TEST RECORDING =====")

    raw_list = load_physionet(
        subject=subject,
        runs=(test_run,),
    )

    raw = raw_list[0]

    # Apply the same frequency filter as during training.
    raw = filter_eeg(raw)

    sfreq = raw.info["sfreq"]
    total_samples = raw.n_times

    chunk_samples = max(
        1,
        int(round(chunk_duration * sfreq)),
    )

    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Recording duration: {raw.times[-1]:.2f} seconds")
    print(f"Chunk duration: {chunk_duration} seconds")

    # --------------------------------------------------
    # 3. Retrieve the motor imagery events
    # --------------------------------------------------

    events, _ = mne.events_from_annotations(
        raw,
        event_id={
            "T1": 1,
            "T2": 2,
        },
        verbose=False,
    )

    print(f"Number of events: {len(events)}")

    # --------------------------------------------------
    # 4. Prepare the EEG channels
    # --------------------------------------------------

    eeg_data = raw.get_data(picks="eeg")

    predictions = []
    expected_labels = []
    latencies = []

    event_index = 0

    # --------------------------------------------------
    # 5. Simulate progressive playback
    # --------------------------------------------------

    print("\n===== REAL-TIME SIMULATION =====")

    for chunk_start in range(
        0,
        total_samples,
        chunk_samples,
    ):

        chunk_end = min(
            chunk_start + chunk_samples,
            total_samples,
        )

        # Simulate the time needed for this chunk to arrive.
        actual_chunk_duration = (
            (chunk_end - chunk_start) / sfreq
        )

        time.sleep(
            actual_chunk_duration / playback_speed
        )

        # Process every event whose complete epoch is now available.
        while event_index < len(events):

            event_sample = events[event_index, 0]
            event_label = events[event_index, 2]

            epoch_start = event_sample + int(
                round(tmin * sfreq)
            )

            epoch_end = event_sample + int(
                round(tmax * sfreq)
            )

            # The complete epoch is not available yet.
            # We wait for the next incoming chunk.
            if chunk_end <= epoch_end:
                break

            # Extract the 3-second EEG epoch.
            # The final sample is included.
            epoch = eeg_data[
                :,
                epoch_start:epoch_end + 1,
            ]

            if epoch.shape[1] != int(
                round((tmax - tmin) * sfreq)
            ) + 1:
                event_index += 1
                continue

            # --------------------------------------------------
            # 6. Make the prediction and measure its latency
            # --------------------------------------------------

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
                print("WARNING: Prediction exceeded 2 seconds!")

            event_index += 1

    # --------------------------------------------------
    # 7. Display the results
    # --------------------------------------------------

    print("\n===== RESULTS =====")

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
        subject=1,
        train_runs=(4, 8),
        test_run=12,
        chunk_duration=1.0,
        playback_speed=1.0,
    )
