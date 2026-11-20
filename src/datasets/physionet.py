
from pathlib import Path

import mne
from mne.datasets import eegbci


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "physionet"


def load_eeg(subject=1, runs=(4, 8, 12)):

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_list = []

    for run in runs:

        files = eegbci.load_data(
            subjects=subject,
            runs=[run],
            path=str(DATA_DIR),
            update_path=False,
        )

        raw = mne.io.read_raw_edf(
            files[0],
            preload=True,
            verbose=False,
        )

        eegbci.standardize(raw)

        raw.set_montage(
            "colin27_1005",
            on_missing="ignore",
        )

        raw_list.append(raw)

    return raw_list
