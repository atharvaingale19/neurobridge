import numpy as np
import mne

from moabb.datasets import BNCI2014_001
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, f1_score

dataset = BNCI2014_001()
subjects = {}

for subject_id in range(1, 10):
    print(f"Loading subject {subject_id}...")

    sessions = dataset.get_data(subjects=[subject_id])[subject_id]
    epochs_list = []

    for raw in sessions["0train"].values():
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

    X = all_epochs.get_data(copy=True).astype(np.float64)
    y = all_epochs.events[:, -1]

    y = np.where(y == all_epochs.event_id["left_hand"], 0, 1)

    subjects[subject_id] = (X, y)

scores = []

for test_subject in range(1, 10):

    X_train = np.concatenate([
        subjects[s][0]
        for s in subjects if s != test_subject
    ])

    y_train = np.concatenate([
        subjects[s][1]
        for s in subjects if s != test_subject
    ])

    X_test, y_test = subjects[test_subject]

    model = Pipeline([
        ("csp", CSP(n_components=6, reg="ledoit_wolf", log=True)),
        ("lda", LinearDiscriminantAnalysis())
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = balanced_accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    scores.append(score)

    print(
        f"Subject {test_subject}: "
        f"Balanced Accuracy={score:.4f}, "
        f"Macro F1={f1:.4f}"
    )

print("\nCross-subject results")
print("Mean balanced accuracy:", np.mean(scores))
print("Standard deviation:", np.std(scores))