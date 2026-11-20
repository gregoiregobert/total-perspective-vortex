
from pathlib import Path

import numpy as np
import moabb

from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "moabb"


def load_eeg(subject=1):

    if not 1 <= subject <= 9:
        raise ValueError("Subject must be between 1 and 9.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Store downloaded data inside the project.
    moabb.set_download_dir(str(DATA_DIR))

    dataset = BNCI2014_001()

    paradigm = LeftRightImagery(
        fmin=8,
        fmax=30,
    )

    X, y, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=[subject],
    )

    # Keep only the training session.
    train_mask = (
        metadata["session"]
        .astype(str)
        .str.lower()
        .str.contains("train")
        .to_numpy()
    )

    if not np.any(train_mask):
        raise RuntimeError(
            "No training session found in BCI dataset."
        )

    X = X[train_mask]
    y = np.asarray(y)[train_mask]
    metadata = metadata.loc[train_mask].reset_index(drop=True)

    # Convert class labels to the same format as PhysioNet.
    label_mapping = {
        "left_hand": 1,
        "right_hand": 2,
    }

    try:
        y = np.array([
            label_mapping[str(label).lower()]
            for label in y
        ])
    except KeyError as error:
        raise ValueError(
            f"Unexpected BCI class label: {error}"
        ) from error

    # Preserve the original run of every epoch.
    groups, _ = metadata["run"].factorize()

    return (
        np.asarray(X, dtype=float),
        y,
        np.asarray(groups),
    )
