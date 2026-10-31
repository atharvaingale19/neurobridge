import mne
import numpy as np
import joblib

from moabb.datasets import BNCI2014_001
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix

dataset = BNCI2014_001()
data = dataset.get_data(subjects=[1])

runs = data[1]["0train"]
epochs_list = []

for raw in runs.values():
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

left_code = all_epochs.event_id["left_hand"]
y = np.where(y == left_code, 0, 1)

model = Pipeline([
    ("csp", CSP(n_components=6, reg="ledoit_wolf", log=True)),
    ("lda", LinearDiscriminantAnalysis())
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

predictions = cross_val_predict(model, X, y, cv=cv)

print("Total trials:", len(y))
print("Balanced accuracy:", balanced_accuracy_score(y, predictions))
print("\nClassification report:")
print(classification_report(y, predictions, target_names=["Left hand", "Right hand"]))
print("Confusion matrix:")
print(confusion_matrix(y, predictions))

model.fit(X, y)
joblib.dump(model, "models/csp_lda.joblib")