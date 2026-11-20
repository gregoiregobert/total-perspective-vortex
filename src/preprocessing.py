
import numpy as np
import mne

from src.config import DATASET_NAME

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


def load_dataset(subject=1, runs=(4, 8, 12), session="T"):

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
def load_eeg(subject=1, runs=(4, 8, 12)):

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
