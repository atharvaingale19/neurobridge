import mne
import numpy as np
from moabb.datasets import BNCI2014_001


def load_eeg(subject_id=1):
    dataset = BNCI2014_001()
    data = dataset.get_data(subjects=[subject_id])

    sessions = data[subject_id]
    runs = sessions["0train"]

    epochs_list = []

    for raw in runs.values():
        raw = raw.copy()
        raw.filter(8, 30, fir_design="firwin")

        events, event_id = mne.events_from_annotations(raw)

        selected = {
            key: value for key, value in event_id.items()
            if key in ["left_hand", "right_hand"]
        }

        epochs = mne.Epochs(
            raw,
            events,
            event_id=selected,
            tmin=0.5,
            tmax=4.0,
            baseline=None,
            preload=True,
            picks="eeg",
            verbose=False
        )

        epochs_list.append(epochs)

    all_epochs = mne.concatenate_epochs(epochs_list)

    X = all_epochs.get_data(copy=True).astype(np.float32)
    y = all_epochs.events[:, -1]

    y = np.where(y == all_epochs.event_id["left_hand"], 0, 1)

    return X, y


if __name__ == "__main__":
    X, y = load_eeg()

    print("Data shape:", X.shape)
    print("Labels:", np.unique(y, return_counts=True))