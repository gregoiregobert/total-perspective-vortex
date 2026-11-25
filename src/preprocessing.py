
import numpy as np
import mne

from src.config import (
    DATASET_NAME,
    RUNS,
)

from src.datasets.physionet import (
    load_eeg as load_physionet,
)

from src.datasets.bci_competition import (
    load_eeg as load_bci,
)


def filter_eeg(raw, low_freq=8, high_freq=30):

    raw.filter(
        l_freq=low_freq,
        h_freq=high_freq,
        verbose=False,
    )

    return raw


def create_epochs(raw, tmin=1, tmax=4):

    events, _ = mne.events_from_annotations(
        raw,
        event_id={
            "T1": 1,
            "T2": 2,
        },
        verbose=False,
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id={
            "T1": 1,
            "T2": 2,
        },
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        picks="eeg",
        reject_by_annotation=True,
        verbose=False,
    )

    X = epochs.get_data()
    y = epochs.events[:, -1]

    return X, y


def load_dataset(subject=1, runs = RUNS, session="T"):

    if DATASET_NAME == "physionet":

        raw_list = load_physionet(
            subject=subject,
            runs=runs,
        )

        X_list = []
        y_list = []
        groups_list = []

        for run, raw in zip(runs, raw_list):

            raw = filter_eeg(raw)

            X, y = create_epochs(raw)

            X_list.append(X)
            y_list.append(y)

            groups_list.append(
                np.full(len(y), run, dtype=int)
            )

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        groups = np.concatenate(groups_list, axis=0)

        return X, y, groups

    elif DATASET_NAME == "bci_competition":

        return load_bci(subject=subject)

    else:
        raise ValueError(
            f"Unknown dataset: {DATASET_NAME}"
        )


# Backward compatibility for scripts that still need Raw data.
def load_eeg(subject=1, runs=RUNS):

    if DATASET_NAME != "physionet":
        raise ValueError(
            "load_eeg() returning Raw objects is only available "
            "for PhysioNet. Use load_dataset() for BCI."
        )

    raw_list = load_physionet(
        subject=subject,
        runs=runs,
    )

    return mne.concatenate_raws(raw_list)


def visualize_preprocessing(raw, low_freq=8, high_freq=30):
    """
    Affiche le signal EEG brut et le signal filtré.
    """

    # On conserve le signal brut en créant une copie.
    raw_filtered = raw.copy()

    # On applique le filtre uniquement sur la copie.
    filter_eeg(
        raw_filtered,
        low_freq=low_freq,
        high_freq=high_freq,
    )

    # Affichage du signal brut.
    # Affichage du signal brut.
    fig_raw = raw.plot(
        duration=10,
        n_channels=10,
        title="EEG - Signal brut",
        show=True,
        block=True,
    )

    # Affichage du signal filtré.
    fig_filtered = raw_filtered.plot(
        duration=10,
        n_channels=10,
        title=f"EEG - Signal filtre ({low_freq}-{high_freq} Hz)",
        show=True,
        block=True,
    )


def main():

    print("Loading EEG data...")

    raw = load_eeg(
        subject=1,
        runs=(4,),
    )

    print("Visualizing raw and filtered signals...")

    visualize_preprocessing(raw)


if __name__ == "__main__":
    main()

