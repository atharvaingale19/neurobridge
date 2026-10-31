import numpy as np
import mne

from moabb.datasets import BNCI2014_001
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

dataset = BNCI2014_001()
subjects = {}

for subject_id in range(1, 10):
    sessions = dataset.get_data(subjects=[subject_id])[subject_id]
    epochs_list = []

    for raw in sessions["0train"].values():
        raw = raw.copy()
        raw.filter(8, 30, fir_design="firwin")

        events, event_id = mne.events_from_annotations(raw)

        selected = {
            k: v for k, v in event_id.items()
            if k in ["left_hand", "right_hand"]
        }

        epochs = mne.Epochs(
            raw, events,
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


def create_model():
    return Pipeline([
        ("csp", CSP(n_components=6, reg="ledoit_wolf", log=True)),
        ("lda", LinearDiscriminantAnalysis())
    ])


target_X, target_y = subjects[9]

X_pool, X_test, y_pool, y_test = train_test_split(
    target_X,
    target_y,
    test_size=0.30,
    random_state=42,
    stratify=target_y
)

X_population = np.concatenate([
    subjects[s][0] for s in subjects if s != 9
])

y_population = np.concatenate([
    subjects[s][1] for s in subjects if s != 9
])

results = {}

for n in [0, 5, 10, 20]:
    rng = np.random.default_rng(42)

    calibration_indices = []

    for label in [0, 1]:
        indices = np.where(y_pool == label)[0]
        selected = rng.choice(indices, size=n, replace=False)
        calibration_indices.extend(selected)

    calibration_indices = np.array(calibration_indices, dtype=int)

    X_cal = X_pool[calibration_indices]
    y_cal = y_pool[calibration_indices]

    if n > 0:
        X_train = np.concatenate([X_population, X_cal])
        y_train = np.concatenate([y_population, y_cal])
    else:
        X_train = X_population
        y_train = y_population

    model = create_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = balanced_accuracy_score(y_test, predictions)
    results[n] = score

    print(f"{n} calibration trials/class: {score:.4f}")


subject_model = create_model()
subject_model.fit(X_pool, y_pool)

subject_predictions = subject_model.predict(X_test)

subject_score = balanced_accuracy_score(y_test, subject_predictions)

print(f"\nSubject-only reference: {subject_score:.4f}")
print("\nCalibration results:", results)